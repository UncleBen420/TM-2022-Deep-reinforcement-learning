import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

# https://github.com/schneimo/ddpg-pytorch/blob/master
# https://arxiv.org/pdf/1711.08946.pdf

# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

class CategoricalNet(nn.Module):
    def __init__(self, classes=5):
        super(CategoricalNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = classes

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7, stride=3),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.iou_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

        self.cat_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, X):
        x = self.backbone(X)
        return self.iou_head(x), self.cat_head(x)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)


class PolicyNet(nn.Module):
    def __init__(self, nb_actions=8, n_kernels=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(nb_actions)
        self.nb_actions = nb_actions

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7, stride=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            #torch.nn.Dropout(0.1),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(64),
            torch.nn.Dropout(0.1),
            torch.nn.Flatten(),
        )

        self.middle = torch.nn.Sequential(
            torch.nn.Linear(576, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 100),
            torch.nn.ReLU()
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.v_head = torch.nn.Sequential(
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )

        self.backbone.to(self.device)
        self.middle.to(self.device)

        self.middle.apply(self.init_weights)


    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs)

    def e_greedy(self, probs):
        p = np.random.random()
        if p < 0.1:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(probs)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        x = self.backbone(state)
        x = self.middle(x)
        preds = self.policy_head(x)
        v = self.v_head(x)
        return preds, v

class TOD:

    def __init__(self, environment, learning_rate=0.0001, gamma=0.1,
                 entropy_coef=0.01, beta_coef=0.01,
                 lr_gamma=0.8, batch_size=64, pa_dataset_size=1024, pa_batch_size=100, img_res=64):

        self.gamma = gamma
        self.environment = environment
        self.environment.agent = self

        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.policy = PolicyNet()
        self.catnet = CategoricalNet()

        self.batch_size = batch_size
        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.cat_optimizer = torch.optim.Adam(self.catnet.parameters(), lr=0.001)
        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.TDE_pa_batch = None
        self.G_pa_batch = None

        self.X_bad = None
        self.X_good = None
        self.Y_bad = None
        self.Y_good = None
        self.Y_label = None

    def pred_class(self, X):
        # Data preparation
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0).to(self.catnet.device)
        X = self.catnet.prepare_data(X)

        # prediction
        iou, cat_pred = self.catnet(X)

        # prediction to numpy
        cat_pred = torch.nn.functional.softmax(cat_pred, dim=1)
        cat_pred = cat_pred.detach().cpu().numpy()[0]
        cat_pred = np.argmax(cat_pred)
        return iou.item(), cat_pred

    def add_to_ds(self, X, Y, label):

        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0).to(self.catnet.device)
        X = self.catnet.prepare_data(X)

        if Y == 1.:
            Y = torch.FloatTensor([Y]).to(self.catnet.device)
            label = torch.LongTensor([label]).to(self.catnet.device)
            #label = torch.nn.functional.one_hot(label, self.catnet.classes)

            if self.X_good is None:
                self.X_good = X
                self.Y_good = Y
                self.Y_label = label
            else:
                self.X_good = torch.cat((self.X_good, X), 0)
                self.Y_good = torch.cat((self.Y_good, Y), 0)
                self.Y_label = torch.cat((self.Y_label, label), 0)
        else:
            Y = torch.FloatTensor([Y]).to(self.catnet.device)
            if self.X_bad is None:
                self.X_bad = X
                self.Y_bad = Y
            else:
                self.X_bad = torch.cat((self.X_bad, X), 0)
                self.Y_bad = torch.cat((self.Y_bad, Y), 0)

    def trim_ds(self, X, Y, label=None):
        if X is not None and len(X) > self.pa_dataset_size:
            # shuffling the batch
            shuffle_index = torch.randperm(len(X))
            X = X[shuffle_index]
            Y = Y[shuffle_index]
            if label is not None:
                label = label[shuffle_index]


            # dataset clipping
            max_size = min(self.pa_dataset_size, len(self.X_good), len(self.X_bad))
            surplus = len(X) - max_size
            _, X = torch.split(X, [surplus, max_size])
            _, Y = torch.split(Y, [surplus, max_size])
            if label is not None:
                _, label = torch.split(label, [surplus, max_size])

        return X, Y, label

    def train_classification(self):
        self.cat_optimizer.zero_grad()
        X = torch.cat((self.X_good, self.X_bad), dim=0)
        Y = torch.cat((self.Y_good, self.Y_bad), dim=0)
        pred, cat_pred = self.catnet(X)
        loss = torch.nn.functional.binary_cross_entropy(pred.squeeze(), Y)
        print(loss)
        loss.backward(retain_graph=True)

        cat_pred = cat_pred[Y == 1]

        cat_loss = torch.nn.functional.cross_entropy(cat_pred, self.Y_label)

        cat_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.catnet.parameters(), 10)
        self.cat_optimizer.step()

        return loss.item() + cat_loss.item()

    def save(self, file):
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print(self.catnet)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())
                                         + sum(p.numel() for p in self.catnet.parameters())))

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G, TDE = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, V = self.policy(S)

        log_probs = torch.log(action_probs)
        log_probs = torch.nan_to_num(log_probs)

        selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))

        #entropy_loss = - self.entropy_coef * (action_probs * log_probs).sum(1).mean()
        value_loss = torch.nn.functional.mse_loss(V.squeeze(), G.detach().squeeze())
        #value_loss = self.beta_coef * torch.nn.functional.mse_loss(V.squeeze(), G)
        value_loss.backward(retain_graph=True)
        policy_loss = - (G.unsqueeze(1) * selected_log_probs).mean()
        loss = policy_loss #+ entropy_loss# + value_loss
        loss.backward()

        self.optimizer.step()

        return loss.item(), 0

    def fit_one_episode(self, S):

        # ------------------------------------------------------------------------------------------------------
        # EPISODE PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = []
        R_batch = []
        A_batch = []
        V_batch = []

        # ------------------------------------------------------------------------------------------------------
        # EPISODE REALISATION
        # ------------------------------------------------------------------------------------------------------
        counter = 0
        sum_v = 0
        sum_reward = 0

        counter += 1
        # State preprocess

        while True:
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)
            with torch.no_grad():
                action_probs, v = self.policy(S)
                action_probs = action_probs.detach().cpu().numpy()[0]

                A = self.policy.follow_policy(action_probs)

            S_prime, R, is_terminal = self.environment.take_action_tod(A, v.item())

            sum_v += v.item()

            S_batch.append(S)
            A_batch.append(A)
            V_batch.append(v.item())
            R_batch.append(R)
            sum_reward += R

            S = S_prime

            if is_terminal:
                break
        TDE_batch = R_batch.copy()
        for i in reversed(range(1, len(R_batch))):
            R_batch[i - 1] += self.gamma * R_batch[i]
            TDE_batch[i - 1] += self.gamma * R_batch[i]
            TDE_batch[i] - V_batch[i]

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)
        TDE_batch = torch.FloatTensor(TDE_batch).to(self.policy.device)
        G_batch = torch.FloatTensor(R_batch).to(self.policy.device)
        #tde_min = torch.min(TDE_batch)
        #tde_max = torch.max(TDE_batch)
        #TDE_batch = (TDE_batch - tde_min) / (tde_max - tde_min)
        TDE_batch = torch.nan_to_num(TDE_batch)


        # ------------------------------------------------------------------------------------------------------
        # PAST ACTION DATASET PREPARATION
        # ------------------------------------------------------------------------------------------------------
        # Append the past action batch to the current batch if possible

        if self.A_pa_batch is not None and len(self.A_pa_batch) > self.pa_batch_size:
            batch = (torch.cat((self.S_pa_batch[0:self.pa_batch_size], S_batch), 0),
                     torch.cat((self.A_pa_batch[0:self.pa_batch_size], A_batch), 0),
                     torch.cat((self.G_pa_batch[0:self.pa_batch_size], G_batch), 0),
                     torch.cat((self.TDE_pa_batch[0:self.pa_batch_size], TDE_batch), 0))
        else:
            batch = (S_batch, A_batch, G_batch, TDE_batch)

        # Add some experiences to the buffer with respect of TD error
        nb_new_memories = min(5, counter)

        idx = torch.randperm(len(A_batch))[:nb_new_memories]

        #idx = torch.multinomial(TDE_batch, nb_new_memories, replacement=True)

        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
            self.TDE_pa_batch = TDE_batch[idx]
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)
            self.TDE_pa_batch = torch.cat((self.TDE_pa_batch, TDE_batch[idx]), 0)

        # clip the buffer if it's to big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            # shuffling the batch
            shuffle_index = torch.randperm(len(self.A_pa_batch))
            self.A_pa_batch = self.A_pa_batch[shuffle_index]
            self.G_pa_batch = self.G_pa_batch[shuffle_index]
            self.S_pa_batch = self.S_pa_batch[shuffle_index]
            self.TDE_pa_batch = self.TDE_pa_batch[shuffle_index]

            # dataset clipping
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])
            _, self.TDE_pa_batch = torch.split(self.TDE_pa_batch, [surplus, self.pa_dataset_size])

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------
        if len(self.A_pa_batch) < self.batch_size:
            print("nope")
            loss = 0
        else:
            loss, value_loss = self.update_policy(batch)

        return loss, sum_reward

    def exploit_one_episode(self, S):
        sum_reward = 0
        sum_V = 0

        while True:
            # State preprocess

            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs, V = self.policy(S)
                action_probs = action_probs.detach().cpu().numpy()[0]
                A = np.argmax(action_probs)

            S_prime, R, is_terminal = self.environment.take_action_tod(A, V.item())

            S = S_prime
            sum_reward += R
            sum_V += 0
            if is_terminal:
                break

        return sum_reward

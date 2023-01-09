import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

# https://github.com/schneimo/ddpg-pytorch/blob/master
# https://arxiv.org/pdf/1711.08946.pdf

# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py


class PolicyNet(nn.Module):
    def __init__(self, nb_actions=8, classes=5, n_kernels=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(nb_actions)
        self.nb_actions = nb_actions
        self.classes = classes

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

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(576, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(576, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, classes)
        )

        self.backbone.to(self.device)

    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs)

    def get_class(self, class_preds):
        proba = torch.nn.functional.softmax(class_preds, dim=1).squeeze()
        pred = torch.argmax(proba).item()
        conf = proba[pred]
        return pred, conf.item()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        x = self.backbone(state)
        preds = self.policy_head(x)
        class_preds = self.classification_head(x)
        return preds, class_preds

class TOD:

    def __init__(self, environment, learning_rate=0.0001, gamma=0.1,
                 entropy_coef=0.01, beta_coef=0.01,
                 lr_gamma=0.8, batch_size=64, pa_dataset_size=1024, pa_batch_size=100, img_res=64):

        self.gamma = gamma
        self.environment = environment
        self.environment.tod = self

        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.policy = PolicyNet()
        self.nb_per_class = np.zeros(self.policy.classes)

        self.batch_size = batch_size
        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.TDE_pa_batch = None
        self.G_pa_batch = None

        self.X = None
        self.Y = None

    def add_to_ds(self, X, Y):

        if np.argmax(self.nb_per_class) == Y and np.std(self.nb_per_class) > 5.:
            return

        self.nb_per_class[Y] += 1

        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0).to(self.policy.device)
        X = self.policy.prepare_data(X)

        Y = torch.LongTensor([Y]).to(self.policy.device)

        if self.X is None:
            self.X = X
            self.Y = Y
        else:
            self.X = torch.cat((self.X, X), 0)
            self.Y = torch.cat((self.Y, Y), 0)


    def trim_ds(self):
        if self.X is not None and len(self.X) > self.pa_dataset_size:
            # shuffling the batch
            shuffle_index = torch.randperm(len(self.X))
            X = self.X[shuffle_index]
            Y = self.Y[shuffle_index]

            # dataset clipping
            max_size = min(self.pa_dataset_size, len(self.X))
            surplus = len(self.X) - max_size
            _, self.X = torch.split(self.X, [surplus, max_size])
            released, self.Y = torch.split(self.Y, [surplus, max_size])

            _, count = released.unique(return_counts=True)
            for i in range(len(count)):
                self.nb_per_class[i] -= count[i].item()

    def train_classification(self):
        if len(self.X) < self.batch_size:
            return 0
        self.optimizer.zero_grad()

        _, class_preds = self.policy(self.X)
        loss = torch.nn.functional.cross_entropy(class_preds, self.Y)
        loss.backward()
        self.optimizer.step()
        print(loss.item())

        return loss.item()

    def save(self, file):
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G, TDE = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, _ = self.policy(S)

        log_probs = torch.log(action_probs)
        log_probs = torch.nan_to_num(log_probs)

        selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))

        policy_loss = - (G.unsqueeze(1) * selected_log_probs).mean()
        loss = policy_loss
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
                action_probs, class_preds = self.policy(S)
                #class_preds = self.class_net(S)

                action_probs = action_probs.detach().cpu().numpy()[0]
                A = self.policy.follow_policy(action_probs)

                label, conf = self.policy.get_class(class_preds)

            S_prime, R, is_terminal = self.environment.take_action_tod(A, conf, label)

            sum_v += conf

            S_batch.append(S)
            A_batch.append(A)
            V_batch.append(conf)
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
            loss = 0
        else:
            loss, value_loss = self.update_policy(batch)

        return loss, sum_reward

    def exploit_one_episode(self, S):
        sum_reward = 0
        while True:
            # State preprocess

            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs, class_preds = self.policy(S)
                #class_preds = self.class_net(S)

                action_probs = action_probs.detach().cpu().numpy()[0]
                A = self.policy.follow_policy(action_probs)

                label, conf = self.policy.get_class(class_preds)

            S_prime, R, is_terminal = self.environment.take_action_tod(A, conf, label)

            S = S_prime
            sum_reward += R
            if is_terminal:
                break

        return sum_reward

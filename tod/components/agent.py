import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR


# https://arxiv.org/pdf/1711.08946.pdf

class CategoricalNet(nn.Module):
    def __init__(self, classes=10):
        super(CategoricalNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=6, stride=3),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.iou_head = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

        self.cat_head = torch.nn.Sequential(
            torch.nn.Linear(64, classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, X):
        x = self.backbone(X)
        return self.iou_head(x)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

class PolicyNet(nn.Module):
    def __init__(self, img_res=200, n_hidden_nodes=256, n_kernels=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(10)
        self.nb_actions = 10

        self.img_res = img_res

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Flatten(),
        )

        self.middle = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU()
        )

        self.headx = torch.nn.Sequential(
            torch.nn.Linear(32, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.heady = torch.nn.Sequential(
            torch.nn.Linear(32, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.headw = torch.nn.Sequential(
            torch.nn.Linear(32, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.headh = torch.nn.Sequential(
            torch.nn.Linear(32, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.backbone.to(self.device)
        self.middle.to(self.device)
        self.headx.to(self.device)

        self.middle.apply(self.init_weights)
        self.headx.apply(self.init_weights)

    def follow_policy(self, probs):
        x = probs[0].detach().cpu().numpy()[0]
        y = probs[1].detach().cpu().numpy()[0]
        w = probs[2].detach().cpu().numpy()[0]
        h = probs[3].detach().cpu().numpy()[0]
        return np.random.choice(self.action_space, p=x), \
            np.random.choice(self.action_space, p=y), \
            np.random.choice(self.action_space, p=w), \
            np.random.choice(self.action_space, p=h)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        x = self.backbone(state)
        x = self.middle(x)
        return (self.headx(x), self.heady(x), self.headw(x), self.headh(x))


class PolicyGradient:

    def __init__(self, environment, learning_rate=0.0001, gamma=0.6,
                 entropy_coef=0.1, beta_coef=0.1,
                 lr_gamma=0.5, batch_size=64, pa_dataset_size=256, pa_batch_size=30, img_res=64):

        self.gamma = gamma
        self.environment = environment
        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.policy = PolicyNet(img_res=img_res)
        self.catnet = CategoricalNet()
        self.action_space = 4
        self.batch_size = batch_size
        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.cat_optimizer = torch.optim.Adam(self.catnet.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)
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

    def add_to_ds(self, X, Y):

        if Y == 1.:
            Y = torch.FloatTensor([Y]).to(self.catnet.device)
            if self.X_good is None:
                self.X_good = X
                self.Y_good = Y
            else:
                self.X_good = torch.cat((self.X_good, X), 0)
                self.Y_good = torch.cat((self.Y_good, Y), 0)
        else:
            Y = torch.FloatTensor([Y]).to(self.catnet.device)
            if self.X_bad is None:
                self.X_bad = X
                self.Y_bad = Y
            else:
                self.X_bad = torch.cat((self.X_bad, X), 0)
                self.Y_bad = torch.cat((self.Y_bad, Y), 0)

    def trim_ds(self, X, Y):
        if len(X) > self.pa_dataset_size:
            # shuffling the batch
            shuffle_index = torch.randperm(len(X))
            X = X[shuffle_index]
            Y = Y[shuffle_index]

            # dataset clipping
            max_size = min(self.pa_dataset_size, len(self.X_good), len(self.X_bad))
            surplus = len(X) - max_size
            _, X = torch.split(X, [surplus, max_size])
            _, Y = torch.split(Y, [surplus, max_size])

        return X, Y

    def train_classification(self):
        self.cat_optimizer.zero_grad()
        X = torch.cat((self.X_good, self.X_bad), dim=0)
        Y = torch.cat((self.Y_good, self.Y_bad), dim=0)
        Y_ = self.catnet(X)
        loss = torch.nn.functional.cross_entropy(Y, Y_.squeeze())
        loss.backward()
        self.cat_optimizer.step()
        return loss.item()

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

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def a2c(self, advantages, rewards, action_probs, log_probs, selected_log_probs, values):
        # entropy_loss = - self.entropy_coef * (action_probs * log_probs).mean(0).sum(1).mean()
        # entropy_loss = torch.nan_to_num(entropy_loss)
        # value_loss = self.beta_coef * torch.nn.functional.mse_loss(values.squeeze(), rewards)

        policy_loss = - (advantages * selected_log_probs.squeeze()).mean()

        # policy_loss = torch.nn.functional.mse_loss(selected_log_probs.squeeze(), advantages.repeat(4, 1).permute(1, 0))
        policy_loss = torch.nan_to_num(policy_loss)
        loss = policy_loss  # + entropy_loss #+ value_loss

        loss.backward(retain_graph=True)

        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 10.)

        return loss.item()

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G, TD = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs = self.policy(S)

        TD = TD.detach()
        for i, probs in enumerate(action_probs):
            log_probs = torch.log(probs)
            log_probs = torch.nan_to_num(log_probs)

            selected_log_probs = torch.gather(log_probs, 1, A[:, i].unsqueeze(1))

            sum_loss += self.a2c(TD, G, action_probs, log_probs, selected_log_probs, 0)

            counter += 1

        self.optimizer.step()
        self.scheduler.step()

        return sum_loss / counter, 0.

    def fit_one_episode(self, S):

        # ------------------------------------------------------------------------------------------------------
        # EPISODE PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = []
        R_batch = []
        A_batch = []
        TDE_batch = []

        # ------------------------------------------------------------------------------------------------------
        # EPISODE REALISATION
        # ------------------------------------------------------------------------------------------------------
        counter = 0
        sum_v = 0
        sum_reward = 0

        counter += 1
        # State preprocess

        state_change = True
        while True:
            if state_change:
                S = torch.from_numpy(S).float()
                S = S.unsqueeze(0).to(self.policy.device)
                S = self.policy.prepare_data(S)
                with torch.no_grad():
                    action_probs = self.policy(S)

            As = self.policy.follow_policy(action_probs)

            S_prime, R, is_terminal, state_change, cat = self.environment.take_action(As)
            bbox, X, Y = cat
            X = torch.from_numpy(X).float()
            X = X.unsqueeze(0).to(self.catnet.device)
            X = self.catnet.prepare_data(X)
            self.add_to_ds(X, Y)

            iou = self.catnet(X).item()
           # print(iou)
            self.environment.add_to_history(bbox, iou)

            sum_v += 0

            S_batch.append(S)
            A_batch.append(As)
            TDE_batch.append(R)
            R_batch.append(R)
            sum_reward += R

            if state_change:
                S = S_prime

            if is_terminal:
                break

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)

        G_batch = torch.FloatTensor(R_batch).to(self.policy.device)
        TDE_batch = torch.FloatTensor(R_batch).to(self.policy.device)

        # TD error is scaled to ensure no exploding gradient
        # also it stabilise the learning : https://arxiv.org/pdf/2105.05347.pdf
        self.min_r = torch.min(TDE_batch)
        self.max_r = torch.max(TDE_batch)
        if len(TDE_batch) > 1:
            TDE_batch = self.minmax_scaling(TDE_batch)

        # if len(TDE_batch) > 1:
        #    mean, std = torch.mean(TDE_batch), torch.std(TDE_batch)
        #    TDE_batch = (TDE_batch - mean) / std

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
        nb_new_memories = min(10, counter)

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
        loss, value_loss = self.update_policy(batch)

        return loss, sum_reward, value_loss, torch.sum(TDE_batch).item()

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
                As = self.policy.follow_policy(action_probs)
                V = V.item()

            S_prime, R, is_terminal = self.environment.take_action(As)

            S = S_prime
            sum_reward += R
            sum_V += V
            if is_terminal:
                break

        return sum_reward, sum_V

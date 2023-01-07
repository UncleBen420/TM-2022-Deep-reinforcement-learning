import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

# https://github.com/schneimo/ddpg-pytorch/blob/master
# https://arxiv.org/pdf/1711.08946.pdf

# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py


class PolicyNet(nn.Module):
    def __init__(self, img_res=200, n_hidden_nodes=256, n_kernels=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(2)
        self.nb_actions = 2

        self.img_res = img_res

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
            torch.nn.BatchNorm2d(64),
            #torch.nn.Dropout(0.1),
            torch.nn.Flatten(),
        )

        self.middle = torch.nn.Sequential(
            torch.nn.Linear(576, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.Softmax(dim=1)
        )


        self.backbone.to(self.device)
        self.middle.to(self.device)

        self.middle.apply(self.init_weights)


    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        x = self.backbone(state)
        x = self.middle(x)
        return x

class DOT:

    def __init__(self, environment, learning_rate=0.001, gamma=0.1,
                 entropy_coef=0.1, beta_coef=0.1,
                 lr_gamma=0.8, batch_size=64, pa_dataset_size=256, pa_batch_size=100, img_res=64):

        self.gamma = gamma
        self.environment = environment
        self.environment.agent = self

        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1

        self.policy = PolicyNet(img_res=img_res)

        self.batch_size = batch_size
        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)
        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.TDE_pa_batch = None
        self.G_pa_batch = None

    def save(self, file):
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def a2c(self, advantages, rewards, action_probs, log_probs, selected_log_probs, values):

        policy_loss = - (advantages.unsqueeze(1) * selected_log_probs).mean()
        loss = policy_loss
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.)
        self.optimizer.step()
        return loss.item()

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs = self.policy(S)

        log_probs = torch.log(action_probs)
        log_probs = torch.nan_to_num(log_probs)

        selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))

        loss = - (G.unsqueeze(1) * selected_log_probs).mean()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), 0

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

        while True:
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)
            with torch.no_grad():
                action_probs = self.policy(S)
                action_probs = action_probs.detach().cpu().numpy()[0]

                A = self.policy.follow_policy(action_probs)

            S_prime, R, is_terminal = self.environment.take_action(A)

            sum_v += 0

            S_batch.append(S)
            A_batch.append(A)
            TDE_batch.append(R)
            R_batch.append(R)
            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        for i in reversed(range(1, len(R_batch))):
            R_batch[i - 1] += self.gamma * R_batch[i]

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)

        G_batch = torch.FloatTensor(R_batch).to(self.policy.device)


        # ------------------------------------------------------------------------------------------------------
        # PAST ACTION DATASET PREPARATION
        # ------------------------------------------------------------------------------------------------------
        # Append the past action batch to the current batch if possible

        if self.A_pa_batch is not None and len(self.A_pa_batch) > self.pa_batch_size:
            batch = (torch.cat((self.S_pa_batch[0:self.pa_batch_size], S_batch), 0),
                     torch.cat((self.A_pa_batch[0:self.pa_batch_size], A_batch), 0),
                     torch.cat((self.G_pa_batch[0:self.pa_batch_size], G_batch), 0))
        else:
            batch = (S_batch, A_batch, G_batch)

        # Add some experiences to the buffer with respect of TD error
        nb_new_memories = min(10, counter)

        idx = torch.randperm(len(A_batch))[:nb_new_memories]

        #idx = torch.multinomial(TDE_batch, nb_new_memories, replacement=True)

        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)

        # clip the buffer if it's to big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            # shuffling the batch
            shuffle_index = torch.randperm(len(self.A_pa_batch))
            self.A_pa_batch = self.A_pa_batch[shuffle_index]
            self.G_pa_batch = self.G_pa_batch[shuffle_index]
            self.S_pa_batch = self.S_pa_batch[shuffle_index]

            # dataset clipping
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------
        loss, value_loss = self.update_policy(batch)

        return loss, sum_reward, value_loss, torch.sum(G_batch).item()

    def exploit_one_episode(self, S):
        sum_reward = 0
        sum_V = 0

        while True:
            # State preprocess

            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs = self.policy(S)
                A = np.argmax(action_probs)

            S_prime, R, is_terminal = self.environment.take_action(A)

            S = S_prime
            sum_reward += R
            sum_V += 0
            if is_terminal:
                break

        return sum_reward, sum_V

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
# https://arxiv.org/pdf/1711.08946.pdf


class PolicyNet(nn.Module):
    def __init__(self, img_res=200, n_hidden_nodes=256, n_kernels=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(10)
        self.nb_actions = 10

        self.img_res = img_res

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(7, 7), padding=3),
            #torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 3), padding=1),
            torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(5, 5), padding=2),
            # torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 3), padding=1),
            torch.nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 3), padding=1),
            #torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 3), padding=1),
            torch.nn.Flatten(),

        )

        self.middle = torch.nn.Sequential(
            torch.nn.Linear(640, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 20),
            torch.nn.ReLU(),
        )

        self.head1 = torch.nn.Sequential(
            torch.nn.Linear(20, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.head2 = torch.nn.Sequential(
            torch.nn.Linear(self.nb_actions + 20, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.head3 = torch.nn.Sequential(
            torch.nn.Linear(self.nb_actions * 2 + 20, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.head4 = torch.nn.Sequential(
            torch.nn.Linear(self.nb_actions * 2 + 20, self.nb_actions),
            torch.nn.Softmax(dim=1)
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(20, 1)
        )

        self.backbone.to(self.device)
        self.middle.to(self.device)
        self.head1.to(self.device)
        self.head2.to(self.device)
        self.head3.to(self.device)
        self.head4.to(self.device)
        self.value_head.to(self.device)

        self.middle.apply(self.init_weights)
        self.head1.apply(self.init_weights)
        self.head2.apply(self.init_weights)
        self.head3.apply(self.init_weights)
        self.head4.apply(self.init_weights)
        self.value_head.apply(self.init_weights)

    def follow_policy(self, probs):
        actions = []
        for proba in probs:
            proba = proba.detach().cpu().numpy()[0]
            actions.append(np.random.choice(self.action_space, p=proba))

        return actions

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        x = self.backbone(state)
        x = self.middle(x)
        x_head1 = self.head1(x)
        x_head2 = self.head2(torch.cat((x_head1, x), 1))
        return (x_head1,
                x_head2,
                self.head3(torch.cat((x_head1, x_head2, x), 1)),
                self.head4(torch.cat((x_head1, x_head2, x), 1))), self.value_head(x)


class PolicyGradient:

    def __init__(self, environment, learning_rate=0.0001, gamma=0.6,
                 entropy_coef=0.1, beta_coef=0.1,
                 lr_gamma=0.5, batch_size=64, pa_dataset_size=256, pa_batch_size=50, img_res=64):

        self.gamma = gamma
        self.environment = environment
        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.policy = PolicyNet(img_res=img_res)
        self.action_space = 4
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

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def a2c(self, advantages, rewards, action_probs, log_probs, selected_log_probs, values):
        entropy_loss = - self.entropy_coef * (action_probs * log_probs).mean(0).sum(1).mean()
        entropy_loss = torch.nan_to_num(entropy_loss)
        #value_loss = self.beta_coef * torch.nn.functional.mse_loss(values.squeeze(), rewards)

        policy_loss = - (advantages * selected_log_probs.squeeze().mean(-1)).mean()

        #policy_loss = torch.nn.functional.mse_loss(selected_log_probs.squeeze(), advantages.repeat(4, 1).permute(1, 0))
        policy_loss = torch.nan_to_num(policy_loss)
        loss = policy_loss + entropy_loss #+ value_loss

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 10.)
        self.optimizer.step()
        return loss.item()

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G, TD = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, V = self.policy(S)

        action_probs = torch.stack((action_probs[0], action_probs[1], action_probs[2], action_probs[3]))

        log_probs = torch.log(action_probs)
        log_probs = torch.nan_to_num(log_probs)

        selected_log_probs = torch.gather(log_probs.permute(1, 0, 2), 2, A.unsqueeze(2))

        sum_loss += self.a2c(TD, G, action_probs, log_probs, selected_log_probs, V)

        counter += 1
        self.scheduler.step()

        return sum_loss / counter

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
                    action_probs, V = self.policy(S)
                    V = V.item()

            sum_v += V
            As = self.policy.follow_policy(action_probs)
            S_prime, R, is_terminal, state_change = self.environment.take_action(As)

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
        TDE_batch = torch.FloatTensor(TDE_batch).to(self.policy.device)

        # TD error is scaled to ensure no exploding gradient
        # also it stabilise the learning : https://arxiv.org/pdf/2105.05347.pdf
        self.min_r = min(torch.min(TDE_batch), self.min_r)
        self.max_r = max(torch.max(TDE_batch), self.max_r)
        #if len(TDE_batch) > 1:
        #    TDE_batch = self.minmax_scaling(TDE_batch)
        if len(TDE_batch) > 1:
            mean, std = torch.mean(TDE_batch), torch.std(TDE_batch)
            TDE_batch = (TDE_batch - mean) / std

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
        #idx = torch.multinomial(1 - TDE_batch, nb_new_memories, replacement=True)
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
        loss = self.update_policy(batch)

        return loss, sum_reward, sum_v, torch.sum(TDE_batch).item()

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

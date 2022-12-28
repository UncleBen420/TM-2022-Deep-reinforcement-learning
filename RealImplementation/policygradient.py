import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.autograd.grad_mode import F
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
from torchvision import models
from torchvision import transforms
from tqdm import tqdm


class PolicyNet(nn.Module):
    def __init__(self, img_res=100, n_hidden_nodes=64, n_kernels=32):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("RUNNING ON {0}".format(self.device))

        self.action_space = np.arange(4)

        self.img_res = img_res
        self.sub_img_res = int(self.img_res / 2)

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=3, out_channels=n_kernels >> 3, kernel_size=(1, 9, 9)),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=n_kernels >> 3, out_channels=n_kernels >> 2, kernel_size=(1, 7, 7)),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((1, 2, 2)),
            torch.nn.Conv3d(in_channels=n_kernels >> 2, out_channels=n_kernels >> 1, kernel_size=(1, 5, 5)),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((1, 2, 2)),
            torch.nn.Conv3d(in_channels=n_kernels >> 1, out_channels=n_kernels, kernel_size=(1, 3, 3)),
            torch.nn.Flatten(),
        )

        self.middle = torch.nn.Sequential(
            torch.nn.Linear(img_res * n_kernels, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes >> 2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes >> 2, n_hidden_nodes >> 3),
            torch.nn.ReLU()
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_nodes >> 3, 4)
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_nodes >> 3, 1)
        )

        self.backbone.to(self.device)
        self.middle.to(self.device)
        self.head.to(self.device)
        self.value_head.to(self.device)

        self.middle.apply(self.init_weights)
        self.head.apply(self.init_weights)
        self.value_head.apply((self.init_weights))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        img = state.permute(0, 3, 1, 2)
        patches = img.unfold(1, 3, 3).unfold(2, self.sub_img_res, self.sub_img_res).unfold(3, self.sub_img_res,
                                                                                           self.sub_img_res)
        patches = patches.contiguous().view(1, 4, -1, self.sub_img_res, self.sub_img_res)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches

    def forward(self, state):
        x = self.backbone(state)
        x = self.middle(x)
        return self.head(x), self.value_head(x)

    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs.detach().cpu().numpy()[0])


class PolicyGradient:

    def __init__(self, environment, learning_rate=0.0005,
                 episodes=100, val_episode=100, gamma=0.6,
                 entropy_coef=0.01, beta_coef=0.05,
                 lr_gamma=0.5, batch_size=64, pa_dataset_size=1000, pa_batch_size=5):

        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.val_episode = val_episode
        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.policy = PolicyNet(img_res=environment.img_res)
        self.action_space = environment.nb_action
        self.batch_size = batch_size
        self.pa_dataset_size = pa_dataset_size
        self.pa_batch_size = pa_batch_size
        print(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def a2c(self, advantages, rewards, action_probs, log_probs, selected_log_probs, values):

        entropy_loss = - self.entropy_coef * (action_probs * log_probs).sum(1).mean()
        value_loss = self.beta_coef * torch.nn.functional.mse_loss(values.squeeze(), rewards)
        policy_loss = - (advantages * selected_log_probs.squeeze()).mean()
        loss = policy_loss + entropy_loss + value_loss
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.)
        self.optimizer.step()
        return loss.item()

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G, TD = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, V = self.policy(S)
        action_probs = torch.nn.functional.softmax(action_probs, dim=1)

        log_probs = torch.log(action_probs)
        log_probs[torch.isinf(log_probs)] = 0

        selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))

        sum_loss += self.a2c(TD, G, action_probs, log_probs, selected_log_probs, V)

        counter += 1
        self.scheduler.step()

        return sum_loss / counter

    def calculate_advantage_tree(self, rewards):
        rewards = np.array(rewards)

        # calculate the discount rewards
        G = rewards[:, 3].astype(float)
        V = rewards[:, 4].astype(float)
        node_info = rewards[:, 0:3].astype(int)

        TDE = V.copy()

        for ni in node_info[::-1]:
            parent, current, child = ni
            parent_index = np.all(node_info[:, [1, 2]] == [parent, current], axis=1)
            current_index = np.all(node_info[:, [1, 2]] == [current, child], axis=1)

            if parent != -1:
                G[parent_index] += self.gamma * G[current_index]
                TDE[parent_index] += self.gamma * V[current_index]

        # calculate the TD error as A = Q(S,A) - V(S) => A + V(S') - V(S)
        return G.tolist(), (G + (TDE - 2 * V)).tolist()

    def fit(self):

        # for plotting
        losses = []
        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []
        nb_effective_action = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            S_pa_batch = None
            A_pa_batch = None
            TDE_pa_batch = None
            G_pa_batch = None
            for i in episode:
                # ------------------------------------------------------------------------------------------------------
                # EPISODE PREPARATION
                # ------------------------------------------------------------------------------------------------------

                S_batch = []
                R_batch = []
                A_batch = []

                S = self.environment.reload_env()

                # ------------------------------------------------------------------------------------------------------
                # EPISODE REALISATION
                # ------------------------------------------------------------------------------------------------------
                counter = 0
                sum_v = 0
                sum_reward = 0
                existing_proba = None
                existing_v = None
                while True:

                    counter += 1
                    # State preprocess
                    S = torch.from_numpy(S).float()
                    S = S.unsqueeze(0).to(self.policy.device)
                    S = self.policy.prepare_data(S)

                    if existing_proba is None:
                        with torch.no_grad():
                            action_probs, V = self.policy(S)
                            action_probs = torch.nn.functional.softmax(action_probs, dim=-1)
                            action_probs = action_probs.detach().cpu().numpy()[0]
                            V = V.item()
                    else:
                        action_probs = existing_proba
                        V = existing_v

                    action_probs /= action_probs.sum()  # resovle an know issue with numpy
                    A = self.environment.follow_policy(action_probs, V)

                    sum_v += V

                    S_prime, R, is_terminal, node_info, existing_pred = self.environment.take_action(A)
                    existing_proba, existing_v = existing_pred
                    parent, current, child = node_info

                    S_batch.append(S)
                    A_batch.append(A)
                    R_batch.append((parent, current, child, R, V))
                    sum_reward += R

                    S = S_prime

                    if is_terminal:
                        break

                # ------------------------------------------------------------------------------------------------------
                # CUMULATED REWARD CALCULATION AND TD ERROR
                # ------------------------------------------------------------------------------------------------------
                G_batch, TDE_batch = self.calculate_advantage_tree(R_batch)

                # ------------------------------------------------------------------------------------------------------
                # BATCH PREPARATION
                # ------------------------------------------------------------------------------------------------------
                weights = TDE_batch
                S_batch = torch.concat(S_batch).to(self.policy.device)
                A_batch = torch.LongTensor(A_batch).to(self.policy.device)
                G_batch = torch.FloatTensor(G_batch).to(self.policy.device)
                TDE_batch = torch.FloatTensor(TDE_batch).to(self.policy.device)

                # TD error is scaled to ensure no exploding gradient
                # also it stabilise the learning : https://arxiv.org/pdf/2105.05347.pdf
                self.min_r = min(torch.min(TDE_batch), self.min_r)
                self.max_r = max(torch.max(TDE_batch), self.max_r)
                TDE_batch = self.minmax_scaling(TDE_batch)

                # ------------------------------------------------------------------------------------------------------
                # PAST ACTION DATASET PREPARATION
                # ------------------------------------------------------------------------------------------------------

                if A_pa_batch is not None and len(A_pa_batch) > self.pa_batch_size:
                    batch = (torch.cat((S_pa_batch[0:self.pa_batch_size], S_batch), 0),
                             torch.cat((A_pa_batch[0:self.pa_batch_size], A_batch), 0),
                             torch.cat((G_pa_batch[0:self.pa_batch_size], G_batch), 0),
                             torch.cat((TDE_pa_batch[0:self.pa_batch_size], TDE_batch), 0))
                else:
                    batch = (S_batch, A_batch, G_batch, TDE_batch)

                # Add some experiences to the buffer with respect of TD error
                nb_new_memories = min(10, counter)

                # idx = torch.randperm(len(A_batch))[:nb_new_memories]
                idx = torch.multinomial(1 - TDE_batch, nb_new_memories, replacement=True)
                if A_pa_batch is None:
                    A_pa_batch = A_batch[idx]
                    S_pa_batch = S_batch[idx]
                    G_pa_batch = G_batch[idx]
                    TDE_pa_batch = TDE_batch[idx]
                else:
                    A_pa_batch = torch.cat((A_pa_batch, A_batch[idx]), 0)
                    S_pa_batch = torch.cat((S_pa_batch, S_batch[idx]), 0)
                    G_pa_batch = torch.cat((G_pa_batch, G_batch[idx]), 0)
                    TDE_pa_batch = torch.cat((TDE_pa_batch, TDE_batch[idx]), 0)

                # clip the buffer if it's to big
                if len(A_pa_batch) > self.pa_dataset_size:
                    # shuffling the batch
                    shuffle_index = torch.randperm(len(A_pa_batch))
                    A_pa_batch = A_pa_batch[shuffle_index]
                    G_pa_batch = G_pa_batch[shuffle_index]
                    S_pa_batch = S_pa_batch[shuffle_index]
                    TDE_pa_batch = TDE_pa_batch[shuffle_index]

                    # dataset clipping
                    surplus = len(A_pa_batch) - self.pa_dataset_size
                    _, A_pa_batch = torch.split(A_pa_batch, [surplus, self.pa_dataset_size])
                    _, G_pa_batch = torch.split(G_pa_batch, [surplus, self.pa_dataset_size])
                    _, S_pa_batch = torch.split(S_pa_batch, [surplus, self.pa_dataset_size])
                    _, TDE_pa_batch = torch.split(TDE_pa_batch, [surplus, self.pa_dataset_size])

                # ------------------------------------------------------------------------------------------------------
                # MODEL OPTIMISATION
                # ------------------------------------------------------------------------------------------------------
                mean_loss = self.update_policy(batch)

                # ------------------------------------------------------------------------------------------------------
                # METRICS RECORD
                # ------------------------------------------------------------------------------------------------------
                rewards.append(sum_reward)
                losses.append(mean_loss)
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                mz = self.environment.nb_max_zoom
                nb_action.append(st)
                nb_effective_action.append(mz)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=sum_reward, loss=mean_loss,
                                    nb_action=st, V=sum_v / counter)

        return losses, rewards, nb_action, good_choices, bad_choices, nb_effective_action

    def exploit(self):

        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []
        conv_policy = []
        time_by_episode = []
        nb_effective_action = []

        with tqdm(range(self.val_episode), unit="episode") as episode:
            for i in episode:
                sum_episode_reward = 0
                S = self.environment.reload_env()
                existing_proba = None
                existing_v = None
                start_time = time.time()
                while True:
                    # State preprocess
                    S = torch.from_numpy(S).float()
                    S = S.unsqueeze(0).to(self.policy.device)
                    S = self.policy.prepare_data(S)

                    if existing_proba is None:
                        with torch.no_grad():
                            probs, V = self.policy(S)
                            probs = torch.nn.functional.softmax(probs, dim=-1)
                            probs = probs.detach().cpu().numpy()[0]
                            V = V.item()
                    else:
                        probs = existing_proba
                        V = existing_v

                    # no need to explore, so we select the most probable action
                    probs /= probs.sum()
                    A = self.environment.exploit(probs, V)
                    S_prime, R, is_terminal, _, existing_pred = self.environment.take_action(A)

                    existing_proba, existing_v = existing_pred

                    S = S_prime
                    sum_episode_reward += R
                    if is_terminal:
                        break
                done_time = time.time()
                rewards.append(sum_episode_reward)
                conv_policy.append(self.environment.conventional_policy_nb_step)
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                mz = self.environment.nb_max_zoom
                nb_action.append(st)
                nb_effective_action.append(mz)
                time_by_episode.append(done_time - start_time)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=rewards[-1], nb_action=st)

        return rewards, nb_action, good_choices, bad_choices, conv_policy, time_by_episode, nb_effective_action

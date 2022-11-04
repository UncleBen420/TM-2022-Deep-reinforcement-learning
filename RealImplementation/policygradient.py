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
    def __init__(self, img_res=64, n_hidden_nodes=128, n_kernels=32):
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
            torch.nn.Conv3d(in_channels=n_kernels >> 1, out_channels=n_kernels, kernel_size=(1, 3, 3)),
            torch.nn.Flatten(),
        )

        self.middle = torch.nn.Sequential(
                torch.nn.Linear(n_kernels * 36, n_hidden_nodes),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden_nodes, n_hidden_nodes >> 2),
                torch.nn.ReLU()
            )

        self.head = torch.nn.Sequential(
                torch.nn.Linear(n_hidden_nodes >> 2, 4)
            )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_nodes >> 2, 1)
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
        patches = img.unfold(1, 3, 3).unfold(2, self.sub_img_res, self.sub_img_res).unfold(3, self.sub_img_res, self.sub_img_res)
        patches = patches.contiguous().view(1, 4, -1, self.sub_img_res, self.sub_img_res)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches

    def forward(self, state):
        #xs = []
        #for i in range(4):
        #    xs.append(self.backbone(state[:, i]))
        x = self.backbone(state)
        #x = torch.concat(x, 1)
        x = self.middle(x)
        return self.head(x), self.value_head(x)

    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs.detach().cpu().numpy()[0])


class PolicyGradient:

    def __init__(self, environment, learning_rate=0.001,
                 episodes=100, val_episode=100, gamma=0.7,
                 entropy_coef=0.001, beta_coef=0.02, clip=0.5,
                 lr_gamma=0.8, batch_size=256, loss_function="ppo"):

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
        self.clip = clip
        print(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)

        if loss_function == "ppo":
            self.loss_function = self.ppo
        elif loss_function == "a2c":
            self.loss_function = self.a2c
        elif loss_function == "reinforce":
            self.loss_function = self.reinforce

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def reinforce(self, advantages, rewards, action_probs, log_probs, selected_log_probs, values):
        # TD error is scaled to ensure no exploding gradient
        # also it stabilise the learning : https://arxiv.org/pdf/2105.05347.pdf
        self.min_r = min(torch.min(advantages), self.min_r)
        self.max_r = max(torch.max(advantages), self.max_r)

        rewards_n = self.minmax_scaling(rewards).unsqueeze(1)

        selected_log_probs *= rewards_n
        policy_loss = - selected_log_probs.mean()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

    def a2c(self, advantages, rewards, action_probs, log_probs, selected_log_probs, values):

        # TD error is scaled to ensure no exploding gradient
        # also it stabilise the learning : https://arxiv.org/pdf/2105.05347.pdf
        self.min_r = min(torch.min(advantages), self.min_r)
        self.max_r = max(torch.max(advantages), self.max_r)
        advantages = self.minmax_scaling(advantages).unsqueeze(1)

        entropy_loss = self.entropy_coef * (action_probs * log_probs).sum(1).mean()
        value_loss = self.beta_coef * torch.nn.functional.mse_loss(values.squeeze(), rewards)
        policy_loss = - (advantages * selected_log_probs).mean()
        loss = policy_loss + entropy_loss
        # upgrade
        value_loss.backward(retain_graph=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.)
        self.optimizer.step()
        return loss.item()

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.
        S, A, G, TD = batch
        S = S.split(self.batch_size)
        A = A.split(self.batch_size)
        G = G.split(self.batch_size)
        TD = TD.split(self.batch_size)


        # create batch of size edible by the gpu
        for i in range(len(A)):
            S_ = S[i]
            A_ = A[i]
            G_ = G[i]
            TD_ = TD[i]

            # Calculate loss
            self.optimizer.zero_grad()
            action_probs, V = self.policy(S_)
            action_probs = torch.nn.functional.softmax(action_probs, dim=1)
            #TD_error = G_ - V.detach()

            log_probs = torch.log(action_probs)
            log_probs[torch.isinf(log_probs)] = 0

            selected_log_probs = torch.gather(log_probs, 1, A_.unsqueeze(1))

            sum_loss += self.loss_function(TD_, G_, action_probs, log_probs, selected_log_probs, V)

            counter += 1
        self.scheduler.step()

        return sum_loss / counter

    def calculate_advantage_tree(self, rewards):
        rewards = np.array(rewards)

        # calculate the discount rewards
        G = rewards[:, 3].astype(float)
        V = rewards[:, 4].astype(float)
        node_info = rewards[:, 0:3].astype(int)

        for ni in node_info[::-1]:
            parent, current, child = ni
            parent_index = np.all(node_info[:, [1, 2]] == [parent, current], axis=1)
            current_index = np.all(node_info[:, [1, 2]] == [current, child], axis=1)

            if parent != -1:
                #G[parent_index] += self.gamma * G[current_index] / 4

                G[parent_index] += self.gamma * G[current_index]
        # calculate the TD error as A = Q(S,A) - V(S) => A + V(S') - V(S)
        TDE = V.copy()
        for ni in node_info[::-1]:
            parent, current, child = ni
            parent_index = np.all(node_info[:, [1, 2]] == [parent, current], axis=1)
            current_index = np.all(node_info[:, [1, 2]] == [current, child], axis=1)

            if parent != -1:
                TDE[parent_index] += self.gamma * V[current_index]
        TDE - 2 * V
        return G.tolist(), (G + (TDE - 2 * V)).tolist()

    def fit(self):

        good_behaviour_dataset = []
        # for plotting
        losses = []
        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []

        with tqdm(range(self.episodes), unit="episode") as episode:
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
                            #V = action_probs.sum().item()
                            action_probs = torch.nn.functional.softmax(action_probs, dim=-1)
                            action_probs = action_probs.detach().cpu().numpy()[0]
                            V = V.item()
                    else:
                        action_probs = existing_proba
                        V = existing_v
                    
                    action_probs /= action_probs.sum() # resovle an know issue with numpy
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
                S_batch = torch.concat(S_batch).to(self.policy.device)
                A_batch = torch.LongTensor(A_batch).to(self.policy.device)
                G_batch = torch.FloatTensor(G_batch).to(self.policy.device)
                TDE_batch = torch.FloatTensor(TDE_batch).to(self.policy.device)

                # ------------------------------------------------------------------------------------------------------
                # MODEL OPTIMISATION
                # ------------------------------------------------------------------------------------------------------
                mean_loss = self.update_policy((S_batch, A_batch, G_batch, TDE_batch))

                # ------------------------------------------------------------------------------------------------------
                # METRICS RECORD
                # ------------------------------------------------------------------------------------------------------
                rewards.append(sum_reward)
                losses.append(mean_loss)
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                nb_action.append(st)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=sum_reward, loss=mean_loss,
                                    nb_action=st, V=sum_v / counter)

        return losses, rewards, nb_action, good_choices, bad_choices

    def exploit(self):

        rewards = []
        nb_action = []
        good_choices = []
        bad_choices = []
        conv_policy = []
        time_by_episode = []

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
                            probs = probs.detach().cpu().numpy()[0]
                            V = V.item()
                    else:
                        probs = existing_proba
                        V = existing_v

                    # no need to explore, so we select the most probable action
                    probs /= probs.sum()
                    A = self.environment.exploit(probs, V)
                    S_prime, R, is_terminal, _, existing_pred = self.environment.take_action(A, V)

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
                nb_action.append(st)
                time_by_episode.append(done_time - start_time)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=rewards[-1], nb_action=st)

        return rewards, nb_action, good_choices, bad_choices, conv_policy, time_by_episode



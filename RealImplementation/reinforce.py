import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from torch.distributions import Categorical
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

class PolicyNet(nn.Module):
    def __init__(self, img_res=40, n_actions=4, n_hidden_nodes=1024, n_kernels=128):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("RUNNING ON {0}".format(self.device))

        self.action_space = np.arange(4)

        self.img_res = img_res
        self.sub_img_res = int(self.img_res / 2)

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=n_kernels >> 2, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=n_kernels >> 2, out_channels=n_kernels >> 1, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=n_kernels >> 1, out_channels=n_kernels, kernel_size=3),
            torch.nn.Flatten(),
        )
        
        self.middle = torch.nn.Sequential(
                torch.nn.Linear(n_kernels * 4, n_hidden_nodes),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden_nodes, n_hidden_nodes >> 2),
                torch.nn.ReLU()
            )

        self.head = torch.nn.Sequential(
                torch.nn.Linear(n_hidden_nodes >> 2, 4),
                torch.nn.Softmax(dim=-1)
            )

            
        self.value_head = torch.nn.Sequential(
                torch.nn.Linear(n_hidden_nodes >> 2, 1)
            )
            
        self.backbone.to(self.device)
        self.middle.to(self.device)
        self.head.to(self.device)
        self.value_head.to(self.device)

    def prepare_data(self, state):
        img = state.permute(0, 3, 1, 2)
        patches = img.unfold(1, 3, 3).unfold(2, self.sub_img_res, self.sub_img_res).unfold(3, self.sub_img_res, self.sub_img_res)
        patches = patches.contiguous().view(1, 4, -1, self.sub_img_res, self.sub_img_res)
        return patches

    def forward(self, state):
        xs = []
        for i in range(4):
            xs.append(self.backbone(state[:, i]))
        x = torch.concat(xs, 1)
        x = self.middle(x)
        return self.head(x), self.value_head(x)

    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs.detach().cpu().numpy()[0])


class Reinforce:

    def __init__(self, environment, learning_rate=0.00002,
                 episodes=100, val_episode=100, gamma=0.5,
                 entropy_coef=0.2, beta_coef=0.2,
                 early_stopping_threshold=0.0001, batch_size=256):

        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.val_episode = val_episode
        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.policy = PolicyNet()
        self.action_space = environment.nb_action
        self.batch_size = batch_size
        print(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.early_stopping_threshold = early_stopping_threshold

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        #for batch in dataset:

        S, A, G = batch
        S = S.split(self.batch_size)
        A = A.split(self.batch_size)
        G = G.split(self.batch_size)

        # create batch of size edible by the gpu
        for i in range(len(A)):
            S_ = S[i]
            A_ = A[i]
            G_ = G[i]

            # Calculate loss
            self.optimizer.zero_grad()
            action_probs, V = self.policy(S_)
            
            TD_error = torch.sub(G_.unsqueeze(1), V.detach())
            
            log_probs = torch.log(action_probs)
            selected_log_probs = TD_error * torch.gather(log_probs, 1, A_.unsqueeze(1))

            entropy_loss = self.beta_coef * (action_probs * log_probs).sum(1).mean()
            value_loss = self.entropy_coef * torch.nn.MSELoss()(V.detach(), G_.unsqueeze(1).detach())

            policy_loss = - selected_log_probs.mean()
            total_policy_loss = policy_loss + entropy_loss + value_loss
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 100.)
            self.optimizer.step()

            sum_loss += total_policy_loss.item()
            counter += 1

        return sum_loss / counter

    def cumulated_reward_tree(self, rewards):
        rewards = np.array(rewards)
        G = rewards[:, 2].astype(float)

        for reward in rewards[::-1]:
            parent, current, R = reward
            if parent != -1:
                G[rewards[:, 1] == parent] += (self.gamma * R) / len(G[rewards[:, 1] == parent])
        return G.tolist()

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
                while True:

                    counter += 1
                    # State preprocess
                    S = torch.from_numpy(S).float()
                    S = S.unsqueeze(0).to(self.policy.device)
                    S = self.policy.prepare_data(S)
                    if existing_proba is None:
                        with torch.no_grad():
                            action_probs, V = self.policy(S)
                            action_probs = action_probs.detach().cpu().numpy()[0]
                    else:
                        action_probs = existing_proba
                    A = self.environment.follow_policy(action_probs)

                    sum_v += V.item()
 
                    S_prime, R, is_terminal, parent, current, proba = self.environment.take_action(A)
                    existing_proba = proba

                    S_batch.append(S)
                    A_batch.append(A)
                    R_batch.append((parent, current, R))
                    sum_reward += R

                    S = S_prime

                    if is_terminal:
                        break

                # ------------------------------------------------------------------------------------------------------
                # CUMULATED REWARD CALCULATION
                # ------------------------------------------------------------------------------------------------------
                G_batch = self.cumulated_reward_tree(R_batch)

                # ------------------------------------------------------------------------------------------------------
                # BATCH PREPARATION
                # ------------------------------------------------------------------------------------------------------
                S_batch = torch.concat(S_batch).to(self.policy.device)
                A_batch = torch.LongTensor(A_batch).to(self.policy.device)
                G_batch = torch.FloatTensor(G_batch).to(self.policy.device)
                self.min_r = min(torch.min(G_batch), self.min_r)
                self.max_r = max(torch.max(G_batch), self.max_r)
                G_batch = self.minmax_scaling(G_batch)


                # ------------------------------------------------------------------------------------------------------
                # MODEL OPTIMISATION
                # ------------------------------------------------------------------------------------------------------
                mean_loss = self.update_policy((S_batch, A_batch, G_batch))

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

        with tqdm(range(self.val_episode), unit="episode") as episode:
            for i in episode:
                sum_episode_reward = 0
                S = self.environment.reload_env()
                existing_proba = None
                while True:
                    # State preprocess
                    S = torch.from_numpy(S).float()
                    S = S.unsqueeze(0).to(self.policy.device)
                    S = self.policy.prepare_data(S)

                    if existing_proba is None:
                        with torch.no_grad():
                            probs, V = self.policy(S)
                            probs = probs.detach().cpu().numpy()[0]
                    else:
                        probs = existing_proba
                    #    V = self.policy.V(S_v)
                    # no need to explore, so we select the most probable action
                    A = self.environment.exploit(probs)
                    S_prime, R, is_terminal, _, _, proba = self.environment.take_action(A)

                    existing_proba = proba

                    S = S_prime
                    sum_episode_reward += R
                    if is_terminal:
                        break

                rewards.append(sum_episode_reward)

                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                nb_action.append(st)
                good_choices.append(gt / (gt + bt + 0.00001))
                bad_choices.append(bt / (gt + bt + 0.00001))

                episode.set_postfix(rewards=rewards[-1], nb_action=st)

        return rewards, nb_action, good_choices, bad_choices



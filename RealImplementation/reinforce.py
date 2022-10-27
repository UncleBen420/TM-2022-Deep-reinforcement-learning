import random
from operator import itemgetter

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from torch.distributions import Categorical
from torchvision import models
from tqdm import tqdm


class PolicyNet(nn.Module):
    def __init__(self, n_actions, img_res, hist_res, n_hidden_nodes=512, n_kernels=64, n_layers=1,fine_tune=False):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("RUNNING ON {0}".format(self.device))

        self.action_space = np.arange(n_actions)
        self.task_space = np.arange(2)

        self.img_res = img_res
        self.hist_res = hist_res

        self.head = torch.nn.Sequential(
            torch.nn.Linear(1024, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, (n_hidden_nodes >> 1)),
            torch.nn.ReLU(),
            torch.nn.Linear((n_hidden_nodes >> 1), n_actions),
            torch.nn.Softmax(dim=-1)
        )

        self.task_head = torch.nn.Sequential(
            torch.nn.Linear(1024, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, (n_hidden_nodes >> 1)),
            torch.nn.ReLU(),
            torch.nn.Linear((n_hidden_nodes >> 1), 2),
            torch.nn.Softmax(dim=-1)
        )

        self.vision_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=n_kernels, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=n_kernels, out_channels=64, kernel_size=3),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Flatten(),
        )

        self.vision_backbone.to(self.device)
        self.head.to(self.device)

        self.vision_backbone.apply(self.init_weights)
        self.head.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        img = self.prepare_data(state)
        x = self.vision_backbone(img)
        task_probs = self.task_head(x)
        action_probs = self.head(x)
        return action_probs, task_probs

    def follow_policy(self, action_probs, task_probs):
        action_probs = action_probs.detach().cpu().numpy()[0]
        task_probs = task_probs.detach().cpu().numpy()[0]
        return np.random.choice(self.action_space, p=action_probs), np.random.choice(self.task_space, p=task_probs)

    def action_prob_task(self, state):
        img = self.prepare_data(state)
        x = self.vision_backbone(img)
        return self.task_head(x)

    def action_prob(self, state):
        img = self.prepare_data(state)
        x = self.vision_backbone(img)
        return self.head(x)


class Reinforce:

    def __init__(self, environment, learning_rate=0.00006,
                 episodes=100, val_episode=10, guided_episodes=50, gamma=0.2,
                 dataset_max_size=10, good_ds_max_size=20,
                 entropy_coef=0.2, img_res=32, hist_res=32, batch_size=128,
                 early_stopping_threshold=0.0001):

        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.val_episode = val_episode
        self.dataset_max_size = dataset_max_size
        self.good_ds_max_size = good_ds_max_size
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.guided_episodes = guided_episodes
        self.policy = PolicyNet(environment.nb_action, img_res, hist_res)
        print(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.early_stopping_threshold = early_stopping_threshold

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def update_policy_task(self, dataset):
        sum_loss_task = 0.
        counter = 0.

        for _, batch in dataset:

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

                task_probs = self.policy.action_prob_task(S_)
                task_log_probs = torch.log(task_probs)
                selected_task_log_probs = G_ * torch.gather(task_log_probs, 1, A_.unsqueeze(1)).squeeze()
                policy_loss_task = - selected_task_log_probs.mean()

                # Calculate gradients
                policy_loss_task.backward()

                # Apply gradients
                self.optimizer.step()

                sum_loss_task += policy_loss_task.item()
                counter += 1

        return sum_loss_task / counter

    def update_policy(self, dataset):

        sum_loss = 0.
        counter = 0.

        for _, batch in dataset:

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

                action_probs = self.policy.action_prob(S_)
                log_probs = torch.log(action_probs)
                selected_log_probs = G_ * torch.gather(log_probs, 1, A_.unsqueeze(1)).squeeze()
                policy_loss = - selected_log_probs.mean()
                policy_loss.backward()
                # Apply gradients
                self.optimizer.step()

                sum_loss += policy_loss.item()
                counter += 1

        return sum_loss / counter

    def cumulated_reward(self, rewards, t):
        Gt = 0
        pw = 0
        for R in rewards[t:]:
            Gt += self.gamma ** pw * R
            pw += 1
        return Gt

    def fit(self):

        good_behaviour_dataset = []
        good_behaviour_task_dataset = []
        # for plotting
        losses = []
        rewards = []
        nb_action = []
        nb_mark = []
        successful_marks = []
        good_choices = []
        bad_choices = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for i in episode:

                # ------------------------------------------------------------------------------------------------------
                # EPISODE PREPARATION
                # ------------------------------------------------------------------------------------------------------

                if i == self.guided_episodes:
                    print("STOP GUIDING AGENT")
                    self.environment.guided = False

                S_batch = []
                R_batch = []
                A_batch = []
                R_task_batch = []
                A_task_batch = []
                S = self.environment.reload_env()
                first_action = True

                # ------------------------------------------------------------------------------------------------------
                # EPISODE REALISATION
                # ------------------------------------------------------------------------------------------------------
                while True:

                    # casting to torch tensor
                    S = torch.from_numpy(S).float()

                    with torch.no_grad():
                        action_probs, action_probs_task = self.policy(S.unsqueeze(0).to(self.policy.device))
                    A, A_task = self.policy.follow_policy(action_probs, action_probs_task)
                    # Exploratory start for guided episodes to ensure the agent don't fall into a local minimum
                    if i <= self.guided_episodes and first_action:
                        A = 15
                    first_action = False
                    S_prime, R, is_terminal, R_task, A_task = self.environment.take_action(A, A_task)

                    S_batch.append(S)
                    A_batch.append(A)
                    R_batch.append(R)
                    A_task_batch.append(A_task)
                    R_task_batch.append(R_task)

                    S = S_prime

                    if is_terminal:
                        break

                sum_episode_reward = np.sum(R_batch)
                sum_task_reward = np.sum(R_task_batch)
                rewards.append(sum_episode_reward)
                # ------------------------------------------------------------------------------------------------------
                # CUMULATED REWARD CALCULATION
                # ------------------------------------------------------------------------------------------------------
                G_batch = []
                G_task_batch = []
                for t in range(len(R_batch)):
                    G_batch.append(self.cumulated_reward(R_batch, t))
                    G_task_batch.append(self.cumulated_reward(R_task_batch, t))

                # ------------------------------------------------------------------------------------------------------
                # BATCH PREPARATION
                # ------------------------------------------------------------------------------------------------------
                S_batch = torch.stack(S_batch).to(self.policy.device)
                A_batch = torch.LongTensor(A_batch).to(self.policy.device)
                G_batch = torch.FloatTensor(G_batch).to(self.policy.device)
                A_task_batch = torch.LongTensor(A_task_batch).to(self.policy.device)
                G_task_batch = torch.FloatTensor(G_task_batch).to(self.policy.device)
                self.min_r = min(torch.min(G_batch), self.min_r)
                self.max_r = max(torch.max(G_batch), self.max_r)
                G_batch = self.minmax_scaling(G_batch)
                m = np.mean((sum_episode_reward, self.environment.nb_actions_taken))

                # ------------------------------------------------------------------------------------------------------
                # DATASET PREPARATION
                # ------------------------------------------------------------------------------------------------------
                good_behaviour_dataset.append((self.environment.nb_good_choice,
                                               (S_batch, A_batch, G_batch)))
                good_behaviour_task_dataset.append((self.environment.marked_correctly,
                                                    (S_batch, A_task_batch, G_task_batch)))

                if len(good_behaviour_dataset) > self.good_ds_max_size:
                    good_behaviour_dataset = sorted(good_behaviour_dataset, key=itemgetter(0), reverse=True)
                    good_behaviour_task_dataset = sorted(good_behaviour_task_dataset, key=itemgetter(0), reverse=True)
                    good_behaviour_dataset.pop(-1)
                    good_behaviour_task_dataset.pop(-1)



                # ------------------------------------------------------------------------------------------------------
                # MODEL OPTIMISATION
                # ------------------------------------------------------------------------------------------------------
                mean_loss = 0.
                mean_loss_task = 0.
                if len(good_behaviour_dataset) > 3:
                    dataset = random.choices(good_behaviour_dataset, k=3)
                    mean_loss += self.update_policy(dataset)

                if len(good_behaviour_task_dataset) > 3:
                    dataset = random.choices(good_behaviour_task_dataset, k=1)
                    mean_loss_task += self.update_policy_task(dataset)

                # ------------------------------------------------------------------------------------------------------
                # METRICS RECORD
                # ------------------------------------------------------------------------------------------------------

                losses.append(mean_loss)
                nbm = self.environment.nb_mark
                nbmc = self.environment.marked_correctly
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                nb_action.append(st)
                nb_mark.append(nbm)
                successful_marks.append(nbmc)
                good_choices.append(gt / (st + 0.00001))
                bad_choices.append(bt / (st + 0.00001))

                episode.set_postfix(rewards=rewards[-1], loss=mean_loss,
                                    loss_task=mean_loss_task, nb_action=st, nb_mark=nbm,
                                    marked_correctly=nbmc, task_reward=sum_task_reward)

        return losses, rewards, nb_mark, nb_action, successful_marks, good_choices, bad_choices

    def exploit(self):

        rewards = []
        nb_action = []
        nb_mark = []
        successful_marks = []
        good_choices = []
        bad_choices = []

        with tqdm(range(self.val_episode), unit="episode") as episode:
            for i in episode:
                sum_episode_reward = 0
                S = self.environment.reload_env()
                while True:
                    # casting to torch tensor
                    S = torch.from_numpy(S).float()

                    with torch.no_grad():
                        action_probs, action_probs_task = self.policy(S.unsqueeze(0).to(self.policy.device))
                    # no need to explore, so we select the most probable action
                    A = np.argmax(action_probs)
                    A_task = np.argmax(action_probs_task)
                    S_prime, R, is_terminal, R_task, A_task = self.environment.take_action(A, A_task)

                    S = S_prime
                    sum_episode_reward += R
                    if is_terminal:
                        break

                rewards.append(sum_episode_reward)

                nbm = self.environment.nb_mark
                nbmc = self.environment.marked_correctly
                st = self.environment.nb_actions_taken
                gt = self.environment.nb_good_choice
                bt = self.environment.nb_bad_choice
                nb_action.append(st)
                nb_mark.append(nbm)
                good_choices.append(gt / (st + 0.00001))
                bad_choices.append(bt / (st + 0.00001))
                successful_marks.append(self.environment.marked_correctly)

                episode.set_postfix(rewards=rewards[-1], nb_action=st, marked_correctly=nbmc, nb_mark=nbm)

        return rewards, nb_mark, nb_action, successful_marks, good_choices, bad_choices



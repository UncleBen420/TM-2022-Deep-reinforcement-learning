import random
from operator import itemgetter

import numpy as np
import torch
from torch import nn

from torch.distributions import Categorical
from torchvision import models
from tqdm import tqdm

class PolicyNet(nn.Module):
    def __init__(self, n_actions, img_res, n_hidden_nodes=32, fine_tune=False):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_space = np.arange(n_actions)

        self.img_res = img_res
        self.split_index = (self.img_res * self.img_res * 3, self.img_res * self.img_res * 3)

        self.vision_backbone = models.mobilenet_v3_small()
        self.history_backbone = models.mobilenet_v3_small()

        self.head = torch.nn.Sequential(
            torch.nn.Linear(64, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_actions),
            torch.nn.Softmax(dim=-1)
        )

        if not fine_tune:
            for params in self.history_backbone.parameters():
                params.requires_grad = True
            for params in self.vision_backbone.parameters():
                params.requires_grad = True

        # Change the classification head.
        self.history_backbone.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=128),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=n_hidden_nodes),
            nn.ReLU()
        )
        self.vision_backbone.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=128),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=n_hidden_nodes),
            nn.ReLU()
        )

        self.vision_backbone.to(self.device)
        self.history_backbone.to(self.device)
        self.head.to(self.device)

    def prepare_data(self, state):
        img, hist = torch.split(state, self.split_index, dim=1)
        img = torch.reshape(img, (-1 , self.img_res, self.img_res, 3))
        hist = torch.reshape(hist, (-1 , self.img_res, self.img_res, 3))
        return img.permute(0, 3, 1, 2), hist.permute(0, 3, 1, 2)

    def forward(self, state):
        img, hist = self.prepare_data(state)
        x_img = self.vision_backbone(img)
        x_hist = self.history_backbone(hist)
        x = torch.cat((x_img, x_hist), 1)
        action_probs = self.head(x)
        return action_probs

    def follow_policy(self, action_probs):
        return np.random.choice(self.action_space, p=action_probs)


class Reinforce:

    def __init__(self, environment, n_actions=10, learning_rate=0.001,
                 episodes=100, guided_episodes=100, gamma=0.01, dataset_max_size=6, good_ds_max_size=50,
                 entropy_coef=0.2, img_res=224):

        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.dataset_max_size = dataset_max_size
        self.good_ds_max_size = good_ds_max_size
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.guided_episodes = guided_episodes
        self.policy = PolicyNet(n_actions, img_res)
        print(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def fit(self):

        dataset = []
        good_behaviour_dataset = []
        # for plotting
        losses = []
        rewards = []
        nb_action = []
        nb_mark = []
        successful_marks = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for i in episode:

                if self.guided_episodes == i:
                    print("stop guiding agent")
                    self.environment.guided = False

                S_batch = []
                R_batch = []
                A_batch = []

                S = self.environment.reload_env()
                while True:
                    # casting to torch tensor
                    S = torch.from_numpy(S).float()

                    with torch.no_grad():
                        action_probs = self.policy(S.unsqueeze(0).to(self.policy.device)).detach().cpu().numpy()[0]
                    A = self.policy.follow_policy(action_probs)
                    S_prime, R, is_terminal, A_tips = self.environment.take_action(A)
                    A = A_tips # the environment can give tips to the agent to help him learn

                    S_batch.append(S)
                    A_batch.append(A)
                    R_batch.append(R)

                    S = S_prime

                    if is_terminal:
                        break

                sum_episode_reward = np.sum(R_batch)
                rewards.append(sum_episode_reward)

                G_batch = []
                for t in range(len(R_batch)):
                    Gt = 0
                    pw = 0
                    for R in R_batch[t:]:
                        Gt += self.gamma ** pw * R
                        pw += 1
                    G_batch.append(Gt)

                S_batch = torch.stack(S_batch).to(self.policy.device)
                A_batch = torch.LongTensor(A_batch).to(self.policy.device)
                G_batch = torch.FloatTensor(G_batch).to(self.policy.device)
                self.min_r = min(torch.min(G_batch), self.min_r)
                self.max_r = max(torch.max(G_batch), self.max_r)
                G_batch = self.minmax_scaling(G_batch)

                dataset.append((sum_episode_reward, (S_batch, A_batch, G_batch)))
                if self.environment.nb_actions_taken < self.environment.nb_max_actions:
                    good_behaviour_dataset.append((sum_episode_reward, (S_batch, A_batch, G_batch)))

                if len(good_behaviour_dataset) > self.good_ds_max_size:
                    good_behaviour_dataset = sorted(good_behaviour_dataset, key=itemgetter(0), reverse=True)
                    good_behaviour_dataset.pop(-1)

                if len(dataset) > self.dataset_max_size:
                    dataset.pop(-1)

                counter = 0
                sum_loss = 0.
                combined_ds = dataset + good_behaviour_dataset
                random.shuffle(combined_ds)

                for _, batch in combined_ds:
                    S, A, G = batch

                    # Calculate loss
                    self.optimizer.zero_grad()

                    logprob = torch.log(self.policy(S))
                    selected_logprobs = G * torch.gather(logprob, 1, A.unsqueeze(1)).squeeze()
                    policy_loss = - selected_logprobs.mean()

                    entropy = Categorical(probs=logprob).entropy()
                    entropy_loss = - entropy.mean()

                    loss = policy_loss + self.entropy_coef * entropy_loss

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    self.optimizer.step()

                    sum_loss += loss.item()
                    counter += 1

                losses.append(sum_loss / counter)

                nbm = self.environment.nb_mark
                st = self.environment.nb_actions_taken
                nb_action.append(st)
                nb_mark.append(nbm)
                successful_marks.append(self.environment.marked_correctly)

                episode.set_postfix(rewards=rewards[-1], loss=sum_loss / counter, nb_action=st, nb_mark=nbm)

        return losses, rewards, nb_mark, nb_action, successful_marks

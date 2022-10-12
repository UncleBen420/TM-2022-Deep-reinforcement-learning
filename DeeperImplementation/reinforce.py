from operator import itemgetter

import numpy as np
import torch

from torch.distributions import Categorical
from tqdm import tqdm

from environment import Action


class Reinforce:

    def __init__(self, environment, n_actions=7, n_hidden_nodes=128, learning_rate=0.001,
                 episodes=100, gamma=0.01, dataset_max_size=4, entropy_coef=0.01, img_res=10):

        self.vision_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(2307, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_actions),
            torch.nn.Softmax(dim=-1)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_space = np.arange(n_actions)

        self.vision_backbone.to(self.device)
        self.head.to(self.device)

        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.dataset_max_size = dataset_max_size
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.img_res = img_res
        self.split_index = (self.img_res * self.img_res * 3, 3)

        self.optimizer_h = torch.optim.Adam(self.head.parameters(), lr=learning_rate)
        self.optimizer_vb = torch.optim.Adam(self.vision_backbone.parameters(), lr=learning_rate)

    def prepare_data(self, state):
        img, pos = torch.split(state, self.split_index, dim=1)
        img = torch.reshape(img, (-1 , self.img_res, self.img_res, 3))
        return img.permute(0, 3, 1, 2), pos

    def predict(self, state):
        img, pos = self.prepare_data(state)
        x = self.vision_backbone(img)
        x = torch.cat((x, pos), 1)
        action_probs = self.head(x)
        return action_probs

    def follow_policy(self, action_probs):
        return np.random.choice(self.action_space, p=action_probs)

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def fit(self):

        dataset = []
        # for plotting
        losses = []
        rewards = []
        nb_action = []
        nb_mark = []
        successful_marks = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for _ in episode:

                episode_loss = []
                S_batch = []
                R_batch = []
                A_batch = []

                S = self.environment.reload_env()

                reward = 0
                V_sum = 0

                while True:
                    # casting to torch tensor
                    S = torch.from_numpy(S).float()

                    with torch.no_grad():
                        action_probs = self.predict(S.unsqueeze(0)).detach().numpy()[0]
                    A = np.random.choice(self.action_space, p=action_probs)
                    S_prime, R, is_terminal, should_have_mark = self.environment.take_action(A)

                    # we can force the agent to learn to mark with shortcutting the action
                    if should_have_mark:
                        A = Action.MARK.value # mark

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

                S_batch = torch.stack(S_batch)
                A_batch = torch.LongTensor(A_batch)
                G_batch = torch.FloatTensor(G_batch)
                self.min_r = min(torch.min(G_batch), self.min_r)
                self.max_r = max(torch.max(G_batch), self.max_r)
                G_batch = self.minmax_scaling(G_batch)

                dataset.append((sum_episode_reward, (S_batch, A_batch, G_batch)))
                dataset = sorted(dataset, key=itemgetter(0), reverse=True)

                if len(dataset) > self.dataset_max_size:
                    dataset.pop(-1)

                counter = 0
                sum_loss = 0.
                for _, batch in dataset:
                    S, A, G = batch

                    # Calculate loss
                    self.optimizer_h.zero_grad()
                    self.optimizer_vb.zero_grad()

                    logprob = torch.log(self.predict(S))
                    selected_logprobs = G * torch.gather(logprob, 1, A.unsqueeze(1)).squeeze()
                    policy_loss = - selected_logprobs.mean()

                    entropy = Categorical(probs=logprob).entropy()
                    entropy_loss = - entropy.mean()

                    loss = policy_loss + self.entropy_coef * entropy_loss

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    self.optimizer_h.step()
                    self.optimizer_vb.step()
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

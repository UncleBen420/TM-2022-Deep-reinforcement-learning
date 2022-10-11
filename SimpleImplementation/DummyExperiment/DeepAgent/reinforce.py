from operator import itemgetter

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

class Reinforce:

    def __init__(self, environment, n_inputs, n_actions=6, n_hidden_nodes=128, learning_rate=0.0001,
                 episodes=100, gamma=0.01, dataset_max_size=4, entropy_coef=0.2):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_actions),
            torch.nn.Softmax(dim=-1)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_space = np.arange(n_actions)
        self.model.to(self.device)
        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.dataset_max_size = dataset_max_size
        self.entropy_coef = entropy_coef
        self.min_r = environment.min_reward
        self.max_r = environment.max_reward

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict(self, state):
        action_probs = self.model(state)
        return action_probs

    def follow_policy(self, action_probs):
        return np.random.choice(self.action_space, p=action_probs)

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def fit(self):

        losses = []
        rewards = []
        dataset = []
        nb_action = []
        nb_mark = []

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
                        action_probs = self.predict(S).detach().numpy()
                    A = np.random.choice(self.action_space, p=action_probs)
                    S_prime, R, is_terminal, should_have_mark = self.environment.take_action(A)

                    # we can force the agent to learn to mark with shortcutting the action
                    if should_have_mark:
                        print("caca")
                        A = 5 # mark

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
                        Gt = Gt + self.gamma ** pw * R
                        pw += 1
                    G_batch.append(Gt)

                S_batch = torch.stack(S_batch)
                A_batch = torch.LongTensor(A_batch)
                G_batch = torch.FloatTensor(G_batch)
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
                    self.optimizer.zero_grad()
                    logprob = torch.log(self.predict(S))
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

                episode.set_postfix(rewards=rewards[-1], loss=sum_loss / counter, nb_action=st, nb_mark=nbm)

        return rewards, losses, nb_mark, nb_action

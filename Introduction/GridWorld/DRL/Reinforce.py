import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from Environment.GridWorld import Action


class Reinforce:

    def __init__(self, environment, n_inputs, n_actions=4, n_hidden_nodes=32, learning_rate=0.001, episodes=100, gamma=0.01):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_actions),
            torch.nn.Softmax(dim=-1)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_space = np.arange(4)
        self.model.to(self.device)
        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict(self, state):
        action_probs = self.model(torch.FloatTensor(state))
        return action_probs

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r

    def follow_policy(self, action_probs):
        return np.random.choice(self.action_space, p=action_probs)

    def fit(self):
        all_rewards = []
        best_rolling = -99999

        loss = []
        rewards = []
        nb_action = []
        dataset = {}
        v = []
        nb_mark = []

        ds = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for _ in episode:

                episode_loss = []
                S_batch = []
                R_batch = []
                A_batch = []

                S = 0  # initial state
                Sv = self.environment.get_env_vision(S)

                reward = 0
                V_sum = 0
                counter = 0

                for _ in range(1000):
                    with torch.no_grad():
                        action_probs = self.predict(Sv).detach().numpy()
                    A = np.random.choice(self.action_space, p=action_probs)
                    S_prime = self.environment.get_next_state(S, Action(A))
                    R = self.environment.get_reward(S, Action(A), S_prime)

                    S_batch.append(Sv)
                    A_batch.append(A)
                    R_batch.append(R)

                    S = S_prime

                    counter += 1

                    if self.environment.states[S].value['is_terminal']:
                        break

                rewards.append(np.sum(R_batch))

                #G_batch = self.discount_rewards(R_batch, self.gamma)

                G_batch = []
                for t in range(len(R_batch)):
                    Gt = 0
                    pw = 0
                    for R in R_batch[t:]:
                        Gt = Gt + self.gamma ** pw * R
                        pw += 1
                    G_batch.append(Gt)

                S_batch = torch.FloatTensor(S_batch)
                A_batch = torch.LongTensor(A_batch)
                G_batch = torch.FloatTensor(G_batch)
                G_batch = (G_batch - torch.mean(G_batch)) / torch.std(G_batch)

                ds.append((S_batch, A_batch, G_batch))

                counter = 0
                sum_loss = 0.
                for S, A, G in ds:

                    # Calculate loss
                    self.optimizer.zero_grad()
                    logprob = torch.log(self.predict(S))
                    selected_logprobs = G * torch.gather(logprob, 1, A.unsqueeze(1)).squeeze()
                    loss = - selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    self.optimizer.step()
                    sum_loss += loss.item()
                    counter += 1

                episode.set_postfix(rewards=rewards[-1], loss=sum_loss / counter, step_taken=counter)

        return rewards

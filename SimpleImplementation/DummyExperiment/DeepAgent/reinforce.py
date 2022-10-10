import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


class Reinforce:

    def __init__(self, environment, n_inputs, n_actions=7, n_hidden_nodes=256, learning_rate=0.01, episodes=1000, gamma=0.1):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_actions),
            torch.nn.Softmax(dim=0)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_array = np.arange(n_actions)
        self.model.to(self.device)
        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)


    def act_prob(self, state):
        return self.model(state)

    def get_action(self, actions_probabilities):
        return np.random.choice(self.action_array, p=actions_probabilities.data.numpy())

    def fit(self):
        all_rewards = []
        best_rolling = -99999

        loss = []
        rewards = []
        nb_action = []
        dataset = {}
        v = []
        nb_mark = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            st = self.environment.nb_max_actions
            for _ in episode:

                episode_loss = []
                S_batch = []
                R_batch = []
                A_batch = []

                S = self.environment.reload_env()
                S = torch.from_numpy(S).float()

                reward = 0
                V_sum = 0
                counter = 0
                while True:
                    proba = self.act_prob(S)
                    A = self.get_action(proba)

                    S_prime, R, is_terminal = self.environment.take_action(A)

                    S_prime = torch.from_numpy(S_prime).float()

                    S_batch.append(S)
                    A_batch.append(A)
                    R_batch.append(R)

                    S = S_prime

                    if is_terminal:
                        break

                rewards.append(np.sum(R_batch))

                G_batch = []
                for t in range(len(R_batch)):
                    Gt = 0
                    pw = 0
                    for R in R_batch[t:]:
                        Gt = Gt + self.gamma ** pw * R
                        pw += 1
                    G_batch.append(Gt)

                G_batch = torch.tensor(G_batch, dtype=torch.float32, device=self.device)
                G_batch = (G_batch - torch.mean(G_batch)) / torch.std(G_batch)

                A_batch = torch.tensor(A_batch, dtype=torch.int64, device=self.device)
                S_batch = torch.stack(S_batch)

                P = self.model(S_batch)
                X = P.gather(dim=1, index=A_batch.view(-1, 1)).squeeze()

                loss = - torch.sum(torch.log(X) * G_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                nbm = self.environment.nb_mark
                st = self.environment.nb_actions_taken
                nb_action.append(st)
                nb_mark.append(nbm)
                episode.set_postfix(rewards=rewards[-1], steps_taken=st, loss=loss.item(), nb_marked=nbm)

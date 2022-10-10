import torch
from tqdm import tqdm

from DRL.Model import DummyNET

from DRL.Policy import E_Greedy_DRL
from Environment.GridWorld import Action
import numpy as np


class DQLearning:

    def __init__(self, environment, alpha=0.1, gamma=0.1, episodes=100, patience=100, dataset_size=32, dataset_max_size=64):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.model = DummyNET(environment.size * environment.size)
        self.policy = E_Greedy_DRL(0.05)
        self.policy.set_agent(self)
        self.patience = patience
        self.dataset_size = dataset_size
        self.dataset_max_size = dataset_max_size
        self.nb_action = self.environment.nb_action
        # for evaluation
        # for visualisation


    def fit(self, verbose=False):
        if verbose:
            history = []
            rewards = []
            dataset = []
            loss = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for _ in episode:
                episode_loss = []
                S = 0  # initial state
                Sv = torch.from_numpy(self.environment.get_env_vision(S)).float()
                if verbose:
                    reward = 0
                counter = 0
                V_sum = 0
                for _ in range(self.patience):
                    # for visualisation
                    Q, V = self.model.predict_no_grad(Sv)
                    A = self.policy.chose_action(Q.to("cpu").numpy().astype(dtype=int))
                    S_prime = self.environment.get_next_state(S, Action(A))
                    R = self.environment.get_reward(S, Action(A), S_prime)
                    Sv_prime = torch.from_numpy(self.environment.get_env_vision(S_prime)).float()

                    dataset.append((Sv, A, R, Sv_prime, self.environment.states[S].value['is_terminal']))

                    # Learning step:
                    if len(dataset) >= self.dataset_size:
                        loss_q, loss_v = self.model.update(dataset, self.gamma)
                        episode_loss.append(loss_q)
                        dataset.clear()

                    S = S_prime
                    V_sum += V.to("cpu").numpy()

                    S = S_prime
                    Sv = Sv_prime

                    if verbose:
                        reward += R

                    counter += 1

                    if self.environment.states[S].value['is_terminal']:
                        break

                if verbose:
                    history.append(V_sum / counter)
                    rewards.append(reward)
                    loss.append(np.mean(episode_loss))


        if verbose:
            return history, rewards, loss

    def get_policy(self):
        return np.argmax(self.Q, axis=1)
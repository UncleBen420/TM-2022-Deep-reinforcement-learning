import random

import torch
from tqdm import tqdm

from DRL.Model import DummyNET

from DRL.Policy import E_Greedy_DRL
from Environment.GridWorld import Action
import numpy as np


class DQLearning:

    def __init__(self, environment, alpha=0.001, gamma=0.5, episodes=100, patience=100, dataset_size=64):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.model = DummyNET(environment.size * environment.size, learning_rate=alpha)
        self.policy = E_Greedy_DRL(0.5)
        self.policy.set_agent(self)
        self.patience = patience
        self.dataset_size = dataset_size
        self.nb_action = self.environment.nb_action

    def fit(self, verbose=False, trajectory=False):
        """
        This method will run the learning process over n episode
        :param verbose: if the learning process must return data or not
        :return: the V function over the episode and the reward over the episode
        """
        dataset = []

        if verbose:
            rewards = []
            loss = []

        if trajectory:
            self.trajectory = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for _ in episode:
                if self.policy.e > 0.1:
                    self.policy.e -= 0.0001
                episode_loss = []
                S = 0  # initial state
                Sv = torch.from_numpy(self.environment.get_env_vision(S)).float()
                if verbose:
                    reward = 0
                counter = 0

                for i in range(self.patience):
                    # for visualisation
                    if trajectory:
                        self.trajectory.append(S)

                    Q = self.model.predict_no_grad(Sv)
                    A = self.policy.chose_action(Q.to("cpu").numpy())
                    S_prime = self.environment.get_next_state(S, Action(A))
                    R = self.environment.get_reward(S, Action(A), S_prime)
                    Sv_prime = torch.from_numpy(self.environment.get_env_vision(S_prime)).float()

                    dataset.append((Sv, A, R, Sv_prime, self.environment.states[S].value['is_terminal']))

                    # Learning step:
                    if len(dataset) >= self.dataset_size and i % 10 == 0:
                        batch = random.choices(dataset, k=self.dataset_size)
                        loss_q = self.model.update(batch, self.gamma)
                        episode_loss.append(loss_q)
                        #dataset.clear()

                    S = S_prime
                    Sv = Sv_prime

                    if verbose:
                        reward += R

                    counter += 1

                    if self.environment.states[S].value['is_terminal']:
                        break

                if verbose:
                    rewards.append(reward)
                    loss.append(np.mean(episode_loss))

                episode.set_postfix(rewards=reward, loss=np.mean(episode_loss), step_taken=counter)

        if verbose:
            return rewards, loss
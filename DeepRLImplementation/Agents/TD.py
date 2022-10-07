"""
This file implement 3 learning agent algorithme: Q-learning, N-step Sarsa and Monte Carlo on policy.
"""
import random
from abc import ABC, abstractmethod
import numpy as np
import torch
from tqdm import tqdm

class QLearning:
    """
    Implementation of the off-policy algorithme QLearning
    """

    def __init__(self, environment, model, policy, alpha=0.1, gamma=0.1,
                 episodes=100, dataset_size=32, dataset_max_size=64):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.dataset_size = dataset_size
        self.dataset_max_size = dataset_max_size
        self.model = model
        self.policy = policy
        self.policy.set_agent(self)
        self.nb_action = environment.nb_action

    def fit(self):
        """
        fit is called to train the agent on the environment
        :return: return the history of V and accumulated reward and the percent of boats left over the episodes
        """
        loss = []
        rewards = []
        boats_left = []
        dataset = []
        v = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            bl=100
            st = self.environment.nb_max_actions
            for _ in episode:
                episode_loss = []

                S = self.environment.reload_env()

                reward = 0
                V_sum = 0
                counter = 0
                while True:
                    # for visualisation
                    Q, V = self.model.predict_no_grad(S)
                    A = self.policy.chose_action(Q.to("cpu").numpy().astype(dtype=int))

                    S_prime, R, is_terminal = self.environment.take_action(A)
                    #Q_prime = self.model.predict(S_prime)
                    #Qy = R + self.gamma * ((not is_terminal) * torch.max(Q_prime))
                    #Vy = R + self.gamma * ((not is_terminal) * V_prime.squeeze())
                    img, vision = S
                    img_prime, vision_prime = S_prime
                    dataset.append((img, vision, A, R, img_prime, vision_prime, is_terminal))
                    # Learning step:
                    if len(dataset) > self.dataset_size:
                        loss_q, loss_v = self.model.update(self.model.split_random(dataset), self.gamma)
                        episode_loss.append(loss_q)

                    if len(dataset) > self.dataset_max_size:
                        dataset.pop(0)

                    S = S_prime
                    V_sum += V.to("cpu").numpy()[0][0]
                    reward += R

                    if is_terminal:
                        break

                    counter += 1

                bl = self.environment.get_marked_percent()
                rewards.append(reward)
                boats_left.append(bl)
                loss.append(np.mean(episode_loss))
                v.append(V_sum / counter)
                episode.set_postfix(Q_loss=loss[-1], V=v[-1], rewards=reward, boats_left=bl)

            return loss, v, rewards, boats_left

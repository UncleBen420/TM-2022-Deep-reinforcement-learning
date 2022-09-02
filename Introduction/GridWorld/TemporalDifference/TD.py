import random

import numpy as np
import random

from Environment import Agent
from Environment.GridWorld import Action


class Sarsa:

    def __init__(self, environment, alpha=0.1, gamma=0.1, epsilon=0.1, episodes=1000):
        self.environment = environment
        self.a = alpha
        self.e = epsilon
        self.gamma = gamma
        self.episodes = episodes
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        if verbose:
            history = []

        for _ in range(self.episodes):

            S = 0 # initial state
            A = Agent.e_greedy(self.Q[S], self.e)

            while True:

                S_prime = self.environment.get_next_state(S, Action(A))
                R = self.environment.get_reward(S, Action(A), S_prime)
                A_prime = Agent.e_greedy(self.Q[S_prime], self.e)
                self.Q[S][A] += self.a * (R + self.gamma * self.Q[S_prime][A_prime] - self.Q[S][A])
                # evaluation of V
                self.V[S] += self.a * (R + self.gamma * self.V[S_prime] - self.V[S])
                S = S_prime
                A = A_prime

                if self.environment.states[S].value['is_terminal']:
                    break
            if verbose:
                history.append(self.V.copy())
        if verbose:
            return history

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


class QLearning:

    def __init__(self, environment, alpha=0.1, gamma=0.1, epsilon=0.1, episodes=1000):
        self.environment = environment
        self.a = alpha
        self.e = epsilon
        self.gamma = gamma
        self.episodes = episodes
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        if verbose:
            history = []

        for _ in range(self.episodes):

            S = 0 # initial state

            while True:

                A = Agent.e_greedy(self.Q[S], self.e)
                S_prime = self.environment.get_next_state(S, Action(A))
                R = self.environment.get_reward(S, Action(A), S_prime)

                self.Q[S][A] += self.a * (R + self.gamma * np.max(self.Q[S_prime]) - self.Q[S][A])
                # evaluation of V
                self.V[S] += self.a * (R + self.gamma * self.V[S_prime] - self.V[S])
                S = S_prime

                if self.environment.states[S].value['is_terminal']:
                    break

            if verbose:
                history.append(self.V.copy())
        if verbose:
            return history


    def get_policy(self):
        return np.argmax(self.Q, axis=1)

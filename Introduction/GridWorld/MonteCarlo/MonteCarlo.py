import random
from abc import abstractmethod

import numpy as np

from Environment.GridWorld import Action


class MC:
    def __init__(self, environment, episodes=1000, gamma=0.1, patience=100):
        self.environment = environment
        self.episodes = episodes
        self.gamma = gamma
        self.patience = patience

        self.policy = environment.init_policy()

        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.nb_step = np.zeros((environment.size * environment.size, environment.nb_action))

    def incremental_mean(self, reward, state, action):
        """
        do the incremental mean between a reward and the espected value
        :param state:
        :param action: the number of the action taken
        :param reward: is the reward given by the environment
        :return: the mean value
        """
        self.nb_step[state][action] += 1

        self.Q[state][action] += (reward -
                                  self.Q[state][action]) / self.nb_step[state][action]

    @abstractmethod
    def fit(self, verbose=False):
        pass

    def fit_and_evaluate(self, threshold=0.001):
        V = []
        for episode in self.fit(verbose=True):
            V.append(self.environment.evaluate_policy(episode, self.gamma, threshold))

        return V


class MCES(MC):

    def fit(self, verbose=False):

        if verbose:
            history = []

        for e in range(self.episodes):

            if verbose:
                history.append(self.policy.copy())

            # No need to check if p(S0, A0) > 0 because in
            # GridWorld every pair A/S have a probability of 1
            S0 = random.randint(0, self.environment.size * self.environment.size - 1)
            A0 = random.randint(0, self.environment.nb_action - 1)

            S = []
            A = []
            R = []

            S.append(S0)
            A.append(A0)
            R.append(0)

            # Generate episode
            # Patience is to stop the generation when the agent is blocked or
            # is policy doesn't allow it to find the terminal state
            for _ in range(self.patience):

                new_state = self.environment.get_next_state(S[-1], Action(A[-1]))

                R.append(self.environment.get_reward(S[-1], Action(A[-1]), new_state))
                S.append(new_state)
                A.append(self.policy[new_state])

                if self.environment.states[new_state].value['is_terminal']:
                    break

            G = 0  # accumulated reward
            already_visited = []
            for t in reversed(range(len(S) - 1)):
                G = self.gamma * G + R[t + 1]

                if not (S[t], A[t]) in already_visited:
                    # version with incremental mean to reduce the memory cost
                    self.incremental_mean(G, S[t], A[t])
                    self.policy[S[t]] = np.argmax(self.Q[S[t]])
                    already_visited.append((S[t], A[t]))

        if verbose:
            return history

        return self.policy

class OnPolicyMC(MC):

    def __init__(self, environment, episodes=1000, gamma=0.1, patience=100, epsilon=0.1, greedy=0.9):
        self.environment = environment
        self.episodes = episodes
        self.gamma = gamma
        self.patience = patience
        self.e = greedy
        self.epsilon = epsilon

        basic_policy = environment.init_policy()
        self.policy = np.zeros((environment.size * environment.size, environment.nb_action))
        for i in range(len(basic_policy)):
            self.policy[i, basic_policy[i]] = 1.

        print(self.policy)

        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.nb_step = np.zeros((environment.size * environment.size, environment.nb_action))

    def e_greedy(self, A):
        """
        this function is used to select an action based on the
        e-greedy function.
        :return: the chosen action
        """
        if np.random.binomial(1, self.e):
            return random.randrange(self.environment.nb_action)
        return np.argmax(A)

    def fit(self, verbose=False):

        if verbose:
            history = []

        for e in range(self.episodes):

            if verbose:
                history.append(self.policy.copy())

            # No need to check if p(S0, A0) > 0 because in
            # GridWorld every pair A/S have a probability of 1
            S0 = 0
            A0 = np.argmax(self.policy[0])

            S = []
            A = []
            R = []

            S.append(S0)
            A.append(A0)
            R.append(0)

            # Generate episode
            # Patience is to stop the generation when the agent is blocked or
            # is policy doesn't allow it to find the terminal state
            for _ in range(self.patience):

                new_state = self.environment.get_next_state(S[-1], A[-1])

                R.append(self.environment.get_reward(S[-1], A[-1], new_state))
                S.append(new_state)
                A.append(np.argmax(self.policy[new_state]))

                if self.environment.states[new_state].value['is_terminal']:
                    break

            G = 0  # accumulated reward
            already_visited = []
            for t in reversed(range(len(S) - 1)):
                G = self.gamma * G + R[t + 1]

                if not (S[t], A[t]) in already_visited:
                    # version with incremental mean to reduce the memory cost
                    self.incremental_mean(G, S[t], A[t])

                    chosen_A = self.e_greedy(self.Q[S[t]])
                    for a in range(self.environment.nb_action):
                        if a is chosen_A:
                            self.policy[S[t], a] = 1 - self.epsilon + self.epsilon / np.abs(np.max(self.policy[S[t]]))
                        else:
                            self.policy[S[t], a] = self.epsilon / np.abs(np.max(self.policy[S[t]]))

                    already_visited.append((S[t], A[t]))

        if verbose:
            return history

        return self.policy

    def get_policy(self):
        return np.argmax(self.policy, axis=1)

import random
import numpy as np

from Environment import Agent
from Environment.GridWorld import Action


class MCES:

    def __init__(self, environment, episodes=1000, gamma=0.1, patience=100):
        self.environment = environment
        self.episodes = episodes
        self.gamma = gamma
        self.patience = patience
        self.policy = Agent.init_policy(environment)
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.nb_step = np.zeros((environment.size * environment.size, environment.nb_action))

    def fit(self, verbose=False):

        if verbose:
            history = []

        for _ in range(self.episodes):

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

                if new_state == S[-1]:
                    break

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
                    self.nb_step, self.Q = Agent.incremental_mean(G, S[t], A[t], self.nb_step, self.Q)
                    self.policy[S[t]] = np.argmax(self.Q[S[t]])
                    already_visited.append((S[t], A[t]))

        if verbose:
            return history

        return self.policy


class OnPolicyMC:

    def __init__(self, environment, episodes=1000, gamma=0.1, patience=100, epsilon=0.1, greedy=0.9):
        self.environment = environment
        self.episodes = episodes
        self.gamma = gamma
        self.patience = patience
        self.e = greedy
        self.epsilon = epsilon

        # Is a soft policy so every action for a state are > 0
        basic_policy = Agent.init_policy(environment)
        self.policy = np.zeros((environment.size * environment.size, environment.nb_action))
        for i in range(len(basic_policy)):
            self.policy[i, basic_policy[i]] = 1.

        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.nb_step = np.zeros((environment.size * environment.size, environment.nb_action))

    def fit(self, verbose=False):

        if verbose:
            history = []

        for _ in range(self.episodes):

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

                new_state = self.environment.get_next_state(S[-1], Action(A[-1]))

                # To stop the generation when the agent is blocked
                # is policy doesn't allow it to find the terminal state
                if new_state == S[-1]:
                    break

                R.append(self.environment.get_reward(S[-1], Action(A[-1]), new_state))
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
                    self.nb_step, self.Q = Agent.incremental_mean(G, S[t], A[t], self.nb_step, self.Q)

                    chosen_A = Agent.e_greedy(self.Q[S[t]], self.e)
                    for a in range(self.environment.nb_action):
                        if a is chosen_A:
                            self.policy[S[t], a] = 1 - self.epsilon + self.epsilon / self.environment.nb_action
                        else:
                            self.policy[S[t], a] = self.epsilon / self.environment.nb_action

                    already_visited.append((S[t], A[t]))

        if verbose:
            return history

        return self.policy

    def get_policy(self):
        return np.argmax(self.policy, axis=1)


class OffPolicyMC:

    def __init__(self, environment, episodes=1000, gamma=0.1, patience=100, greedy=0.9):
        self.environment = environment
        self.episodes = episodes
        self.gamma = gamma
        self.patience = patience
        self.e = greedy

        # Is a soft policy so every action for a state are > 0
        self.policy = Agent.init_policy(environment)
        # b is a soft policy so p > 0 for every State / Action pair
        self.b = np.full((environment.size * environment.size, environment.nb_action), 0.1)
        for i in range(len(self.policy)):
            self.b[i, self.policy[i]] = 1.

        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.C = np.zeros((environment.size * environment.size, environment.nb_action))

    def fit(self, verbose=False):

        if verbose:
            history = []

        for _ in range(self.episodes):

            if verbose:
                history.append(self.policy.copy())

            # No need to check if p(S0, A0) > 0 because in
            # GridWorld every pair A/S have a probability of 1
            S0 = 0
            A0 = Agent.e_greedy(self.b[0], self.e)

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

                # To stop the generation when the agent is blocked
                # is policy doesn't allow it to find the terminal state
                if new_state == S[-1]:
                    break

                R.append(self.environment.get_reward(S[-1], Action(A[-1]), new_state))
                S.append(new_state)
                A.append(Agent.e_greedy(self.b[new_state], self.e))

                if self.environment.states[new_state].value['is_terminal']:
                    break

            G = 0  # accumulated reward
            W = 1
            for t in reversed(range(len(S) - 1)):
                G = self.gamma * G + R[t + 1]
                self.C[S[t]][A[t]] += W
                self.Q[S[t]][A[t]] += (W / self.C[S[t]][A[t]]) * (G - self.Q[S[t]][A[t]])
                self.policy[S[t]] = Agent.e_greedy(self.Q[S[t]], self.e)
                if A[t] != self.policy[S[t]]:
                    break

                W *= 1 / self.b[S[t]][A[t]]

        if verbose:
            return history

        return self.policy

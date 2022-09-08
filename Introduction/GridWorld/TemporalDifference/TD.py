import numpy as np
from Environment.GridWorld import Action


class Sarsa:

    def __init__(self, environment, policy, alpha=0.1, gamma=0.1, episodes=100, patience=200):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.policy = policy
        self.policy.set_agent(self)
        self.patience = patience
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        if verbose:
            history = []
            rewards = []

        for _ in range(self.episodes):

            S = 0  # initial state
            if verbose:
                reward = 0
            A = self.policy.chose_action(S)

            for _ in range(self.patience):

                S_prime = self.environment.get_next_state(S, Action(A))
                R = self.environment.get_reward(S, Action(A), S_prime)
                A_prime = self.policy.chose_action(S_prime)
                self.Q[S][A] += self.a * (R + self.gamma * self.Q[S_prime][A_prime] - self.Q[S][A])
                # evaluation of V
                self.V[S] += self.a * (R + self.gamma * self.V[S_prime] - self.V[S])
                S = S_prime
                A = A_prime

                if verbose:
                    reward += R

                if self.environment.states[S].value['is_terminal']:
                    break
            if verbose:
                history.append(self.V.copy())
                rewards.append(reward)

        if verbose:
            return history, rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


class QLearning:

    def __init__(self, environment, policy, alpha=0.1, gamma=0.1, episodes=100, patience=200):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.policy = policy
        self.policy.set_agent(self)
        self.patience = patience
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        if verbose:
            history = []
            rewards = []

        for _ in range(self.episodes):

            S = 0  # initial state
            if verbose:
                reward = 0

            for _ in range(self.patience):

                A = self.policy.chose_action(S)
                S_prime = self.environment.get_next_state(S, Action(A))
                R = self.environment.get_reward(S, Action(A), S_prime)

                self.Q[S][A] += self.a * (R + self.gamma * np.max(self.Q[S_prime]) - self.Q[S][A])
                # evaluation of V
                self.V[S] += self.a * (R + self.gamma * self.V[S_prime] - self.V[S])
                S = S_prime

                if verbose:
                    reward += R

                if self.environment.states[S].value['is_terminal']:
                    break

            if verbose:
                history.append(self.V.copy())
                rewards.append(reward)

        if verbose:
            return history, rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


class DoubleQLearning:

    def __init__(self, environment, policy, alpha=0.1, gamma=0.1, episodes=100, patience=200):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.Q1 = np.zeros((environment.size * environment.size, environment.nb_action))
        self.Q2 = np.zeros((environment.size * environment.size, environment.nb_action))
        self.Q = self.Q1 + self.Q2  # it is a non-necessary field, but it's used for convenience with policy class
        self.policy = policy
        self.policy.set_agent(self)
        self.patience = patience
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        if verbose:
            history = []
            rewards = []

        for _ in range(self.episodes):

            S = 0  # initial state
            if verbose:
                reward = 0

            for _ in range(self.patience):

                # this is not optimised because Q1 + Q2 is done over all the states, but it's convenient
                self.Q = self.Q1 + self.Q2
                A = self.policy.chose_action(S)

                S_prime = self.environment.get_next_state(S, Action(A))
                R = self.environment.get_reward(S, Action(A), S_prime)

                if np.random.binomial(1, 0.5):
                    self.Q1[S][A] += self.a * (
                                R + self.gamma * self.Q2[S_prime][np.argmax(self.Q1[S_prime])] - self.Q1[S][A])
                else:
                    self.Q2[S][A] += self.a * (
                                R + self.gamma * self.Q1[S_prime][np.argmax(self.Q2[S_prime])] - self.Q2[S][A])

                # evaluation of V
                self.V[S] += self.a * (R + self.gamma * self.V[S_prime] - self.V[S])
                S = S_prime

                if verbose:
                    reward += R

                if self.environment.states[S].value['is_terminal']:
                    break

            if verbose:
                history.append(self.V.copy())
                rewards.append(reward)

        if verbose:
            return history, rewards

    def get_policy(self):
        return np.argmax(self.Q1, axis=1)


class ExpectedSarsa:

    def __init__(self, environment, policy, alpha=0.1, gamma=0.1, episodes=100, patience=200):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.policy = policy
        self.policy.set_agent(self)
        self.patience = patience
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        if verbose:
            history = []
            rewards = []

        for _ in range(self.episodes):

            S = 0  # initial state
            if verbose:
                reward = 0

            for _ in range(self.patience):

                A = self.policy.chose_action(S)
                S_prime = self.environment.get_next_state(S, Action(A))
                R = self.environment.get_reward(S, Action(A), S_prime)

                self.Q[S][A] += self.a * (R +
                                          self.gamma *
                                          np.sum(self.policy.probability(S_prime) * self.Q[S_prime]) -
                                          self.Q[S][A])
                # evaluation of V
                self.V[S] += self.a * (R + self.gamma * self.V[S_prime] - self.V[S])
                S = S_prime

                if verbose:
                    reward += R

                if self.environment.states[S].value['is_terminal']:
                    break

            if verbose:
                history.append(self.V.copy())
                rewards.append(reward)

        if verbose:
            return history, rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

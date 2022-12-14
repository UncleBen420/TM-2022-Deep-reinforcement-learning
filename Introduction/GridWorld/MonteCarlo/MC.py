"""
This file contains the classes implementing Monte Carlo method
"""
import random
import numpy as np
from Environment import Agent
from Environment.GridWorld import Action


class MCES:
    """
    Implementation of a Monte Carlo exploratory search.
    """
    def __init__(self, environment, episodes=1000, gamma=0.1, patience=100):
        self.environment = environment
        self.episodes = episodes
        self.gamma = gamma
        self.patience = patience
        self.policy = Agent.init_policy(environment)
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.nb_step = np.zeros((environment.size * environment.size, environment.nb_action))
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))
        self.nb_step_v = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        """
        This method fit the agent over n episode on the environment. It has a patience parameter
        to ensure the algorithme is not stuck in infinite loop.
        :param verbose: if true the function return the evolution
        of the V function over the iteration.
        :return: if true return the v function of multiple iteration, if false return nothing
        """
        if verbose:
            history = []
            rewards = []

        for _ in range(self.episodes):

            if verbose:
                reward = 0

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

                if verbose:
                    reward += R[-1]

                if self.environment.states[new_state].value['is_terminal']:
                    break

            G = 0  # accumulated reward
            already_visited = []
            for t in reversed(range(len(S) - 1)):
                G = self.gamma * G + R[t + 1]

                if (S[t], A[t]) not in already_visited:
                    # version with incremental mean to reduce the memory cost
                    self.nb_step, self.Q = Agent.incremental_mean(G, S[t], A[t], self.nb_step, self.Q)

                    self.nb_step_v, self.V = Agent.incremental_mean_V(G, S[t], self.nb_step_v, self.V)
                    self.policy[S[t]] = np.argmax(self.Q[S[t]])
                    already_visited.append((S[t], A[t]))

            if verbose:
                history.append(self.V.copy())
                rewards.append(reward)

        if verbose:
            return history, rewards

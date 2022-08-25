"""
This file contain the implementation for 2 classes of agents.
"""
import random
import numpy as np


class RandomAgent:
    """
    This class provide a dummy agent that make choice in
    pure random manners.
    """

    def __init__(self, k):
        self.k = k

    def choose(self):
        """
        this function is called to choose an action
        in this case the action is the choice of a bandit.
        :return: the number of the chosen bandit.
        """
        return random.randrange(self.k)

    def update(self, reward, action):
        """ value function update"""
        # random agent do not learn
        pass


class RLAgent:
    """
    This class implement a simple agent.
    It disposes of two action selection function possible.
    """

    def __init__(self, k, action_selection_function="egreedy", e=0.1, c=2):
        self.e = e
        self.k = k
        self.c = c
        self.nb_total_step = 0
        self.nb_step = np.zeros(k)
        self.expected_values = np.zeros(k)
        if action_selection_function == "ucb":
            self.action_selection_function = self.ucb
        else:
            self.action_selection_function = self.e_greedy

    def e_greedy(self):
        """
        this function is used to select an action based on the
        e-greedy function.
        :return: the chosen action
        """
        if np.random.binomial(1, self.e):
            return random.randrange(self.k)

        return np.argmax(self.expected_values)

    def ucb(self):
        """
        this function is used to select an action based on the
        Upper-Confidence-Bound Action Selection function.
        :return: the chosen action
        """

        if np.count_nonzero(self.expected_values == 0):
            return np.where(self.expected_values == 0)[0][0]

        return np.argmax(self.expected_values + self.c *
                         np.sqrt(np.log(self.nb_total_step) / self.nb_step))

    def incremental_mean(self, reward, action):
        """
        do the incremental mean between a reward and the espected value
        :param action: the number of the action taken
        :param reward: is the reward given by the environment
        :return: the mean value
        """
        return self.expected_values[action] + (reward -
                                               self.expected_values[action]) / self.nb_step[action]

    def choose(self):
        """
        this function call the e-greedy or the ucb function based on the user choice
        :return: the chosen action
        """
        return self.action_selection_function()

    def update(self, reward, action):
        """
        update the expected return of the chosen action
        :param action: the number of the action taken
        :param reward: is the reward given by the environment
        """
        self.nb_step[action] += 1
        self.nb_total_step += 1
        self.expected_values[action] = self.incremental_mean(reward, action)

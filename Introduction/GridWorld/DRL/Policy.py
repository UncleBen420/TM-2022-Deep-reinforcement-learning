import random

import numpy as np


class E_Greedy_DRL:
    """
    Implementation of an e-greedy policy
    """

    def __init__(self, epsilon):
        super().__init__()
        self.e = epsilon

    def set_agent(self, agent):
        """
        Must be call by the agent to link it to the policy object
        otherwise methods cannot work.
        :param agent: the agent the object is attached
        """
        self.agent = agent

    def chose_action(self, Q):
        """
        return the chosen action according the e-greedy policy
        :param Q:
        :param state: state in which the agent is.
        :return: the chosen action
        """
        if np.random.binomial(1, self.e):
            return random.randrange(self.agent.nb_action)
        return np.argmax(Q)

    def probability(self, Q):
        """
        Return the probability of each action for this state according
        to the e-greedy policy.
        :param Q:
        :param state: state in which the agent is.
        :return: the probability for each action
        """
        greedy_actions = Q == np.max(Q)  # all actions with the maximal value
        nb_greedy = np.count_nonzero(greedy_actions)  # number of max actions
        non_greedy_probability = self.e / len(Q)
        return greedy_actions * ((1 - self.e) / nb_greedy) + non_greedy_probability

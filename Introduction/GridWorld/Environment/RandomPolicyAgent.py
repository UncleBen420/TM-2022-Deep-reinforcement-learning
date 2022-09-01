import random

import numpy as np

from Environment import Agent
from Environment.GridWorld import Action


class RandomPolicyAgent:

    def __init__(self, environment, threshold=0.001, gamma=0.1):
        self.environment = environment
        self.threshold = threshold
        self.gamma = gamma

    def random_policy(self):
        policy = []
        for s in self.environment.states:

            if s.value['is_terminal']:
                policy.append(-1)
            else:
                policy.append(random.randint(0, self.environment.nb_action - 1))

        return policy

    def generate_t_policy_validations(self, times):

        V = []

        for t in range(times):
            policy = self.random_policy()
            V.append(Agent.evaluate_policy(self.environment, policy, self.threshold, self.gamma))

        return V


"""
This file contain an implementation of an dynamic programming agent capable of resolving
a grid world problem
"""
import numpy as np
from Environment import Agent
from Environment.GridWorld import Action


class DP:
    """
    This class implement a dynamic programming algorithme based on evaluation and updating a policy
    """
    def __init__(self, environment, threshold=0.001, gamma=0.1):
        self.policy = np.zeros((environment.size * environment.size))
        self.V = np.zeros((environment.size * environment.size))
        self.policy = Agent.init_policy(environment)

        self.threshold = threshold
        self.environment = environment
        self.gamma = gamma

    def evaluate_policy(self):
        '''
        This method evaluate the policy by updating the function V
        :return: return the updated function V
        '''

        while True:
            delta = 0
            for s in range(self.environment.states.size):
                # Analysing terminal state is not necessary
                if not self.environment.states[s].value['is_terminal']:
                    v = self.V[s]
                    a = self.policy[s]

                    ns = self.environment.get_next_state(s, Action(a))
                    r = self.environment.get_reward(s, Action(a), ns)
                    p = self.environment.get_probability(s, Action(a), ns, r)
                    self.V[s] = p * (r + self.gamma * self.V[ns])
                    delta = max(delta, abs(v - self.V[s]))

            if delta < self.threshold:
                break
        return self.V

    def update_policy(self):
        """
        This method update the policy.
        :return: return if the policy is stable after updating.
        """

        policy_stable = True
        for s in range(self.environment.states.size):
            # Analysing terminal state is not necessary
            if not self.environment.states[s].value['is_terminal']:

                old_action = self.policy[s]
                possible_actions = np.zeros(self.environment.nb_action)

                for a in range(self.environment.nb_action):
                    ns = self.environment.get_next_state(s, Action(a))
                    r = self.environment.get_reward(s, Action(a), ns)

                    p = self.environment.get_probability(s, Action(a), ns, r)

                    possible_actions[a] = p * (r + self.gamma * self.V[ns])

                self.policy[s] = np.argmax(possible_actions)
                if old_action != self.policy[s]:
                    policy_stable = False

        return policy_stable

    def fit(self, verbose=False):
        """
        This method implement the policy iteration method by calling successively evaluation and update.
        :param verbose: if true the function return the evolution of the V function over the iteration.
        :return: if true return the v function of multiple iteration, if false return nothing
        """
        if verbose:
            v = []
            while True:
                v.append(self.evaluate_policy().copy())
                if self.update_policy():
                    break

            return v

        while True:
            self.evaluate_policy().copy()
            if self.update_policy():
                break

"""
This file implement helper function for the different algorithms implemented
"""
import random
import numpy as np
from Environment.GridWorld import Action
from abc import ABC, abstractmethod


class Policy(ABC):

    def __init__(self):
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    @abstractmethod
    def chose_action(self, state):
        pass

    @abstractmethod
    def probability(self, state):
        pass


class E_Greedy(Policy):

    def __init__(self, epsilon):
        self.e = epsilon

    def chose_action(self, state):
        if np.random.binomial(1, self.e):
            return random.randrange(len(self.agent.Q[state]))
        return np.argmax(self.agent.Q[state])

    def probability(self, state):
        greedy_actions = self.agent.Q[state] == np.max(self.agent.Q[state])  # all actions with the maximal value
        nb_greedy = np.count_nonzero(greedy_actions)  # number of max actions
        non_greedy_probability = self.e / len(self.agent.Q[state])
        return greedy_actions * ((1 - self.e) / nb_greedy) + non_greedy_probability


class Greedy(Policy):

    def chose_action(self, state):
        return np.argmax(self.agent.Q[state])

    def probability(self, state):
        greedy_actions = self.agent.Q[state] == np.max(self.agent.Q[state])
        nb_greedy = np.count_nonzero(greedy_actions)
        return greedy_actions * (1 / nb_greedy)


class UCB(Policy):

    def __init__(self, c):
        self.nb_step = None
        self.c = c
        self.nb_total_step = 0

    def set_agent(self, agent):
        self.agent = agent
        self.nb_step = np.zeros_like(agent.Q)

    def calculation(self, state):
        return self.agent.Q[state] + self.c * np.sqrt(np.log(self.nb_total_step) / self.nb_step[state])

    def chose_action(self, state):
        self.nb_total_step += 1
        self.nb_step[state] += 1

        # if an action has never been taken its considered has a maximizing action
        if np.count_nonzero(self.agent.Q[state] == 0):
            return np.where(self.agent.Q[state] == 0)[0][0]

        return np.argmax(self.calculation(state))

    def probability(self, state):
        chance_of_been_chosen = self.calculation(state)
        chance_of_been_chosen = ((chance_of_been_chosen - np.min(chance_of_been_chosen)) /
                                 (np.max(chance_of_been_chosen) - np.min(chance_of_been_chosen) + 0.000001))
        chance_of_been_chosen = np.where(self.agent.Q[state] == 0, 1., chance_of_been_chosen) + 0.01
        sum_of_probability = np.sum(chance_of_been_chosen)

        return chance_of_been_chosen / sum_of_probability


def incremental_mean(reward, state, action, nb_step, Q):
    """
    do the incremental mean between a reward and the expected value
    :param Q: the Q function represented by a lookup table
    :param nb_step: number of time a incremental mean as been called
    :param state: the current state
    :param action: the action taken
    :param reward: is the reward given by the environment
    :return: the mean value
    """
    nb_step[state][action] += 1

    Q[state][action] += (reward - Q[state][action]) / nb_step[state][action]
    return nb_step, Q


def incremental_mean_V(reward, state, nb_step, V):
    """
    do the incremental mean between a reward and the expected
    value for a V function
    :param V: the V function for every State
    :param nb_step: number of time a incremental mean as been called
    :param state: the current state
    :param reward: is the reward given by the environment
    :return: the mean value
    """
    nb_step[state] += 1
    V[state] += (reward - V[state]) / nb_step[state]
    return nb_step, V


def init_policy(environment):
    """
    This function is used in some algorithme to initialise a policy
    to a path in the grid world that will lead to the terminal state no
    matter were the starting point is.
    :param environment: the grid world environment object
    :return: return a policy with the same shape as the environment
    """
    policy = np.full((environment.size, environment.size), 2)
    for i in range(environment.size):
        if i % 2:
            policy[i, :] = 0
            policy[i, 0] = 3
        else:
            policy[i, -1] = 3

    policy[-1] = 2
    policy[-1, -1] = -1

    return policy.reshape((environment.size * environment.size))


def evaluate_policy(environment, policy, threshold=0.001, gamma=0.1):
    """
    this function can evaluate a policy with the dynamic programming
    evaluation method.
    :param environment: the grid world environment object
    :param policy: the current policy of the agent (it doesn't matter the algorithm)
    :param threshold: the minimum threshold between V and V + 1.
    :param gamma: the gamma parameter for the V update
    :return: a V function indicating the expected return following this policy
    """
    V = np.zeros((environment.size * environment.size))

    while True:
        delta = 0
        for s in range(environment.states.size):
            # Analysing terminal state is not necessary
            if not environment.states[s].value['is_terminal']:
                v = V[s]
                a = policy[s]

                ns = environment.get_next_state(s, Action(a))
                r = environment.get_reward(s, Action(a), ns)
                p = environment.get_probability(s, Action(a), ns, r)
                V[s] = p * (r + gamma * V[ns])
                delta = max(delta, abs(v - V[s]))

        if delta < threshold:
            break
    return V


def render_policy(environment, policy):
    """
    this function allow a user to render the policy to see which action
    a greedy agent would follow.
    :param environment: the grid world environment object
    :param policy: the current policy of the agent (it doesn't matter the algorithm)
    :return: a string representing
    """
    visual = ""
    for i in range(environment.size):
        visual += '|'
        for j in range(environment.size):

            action = policy[i * environment.size + j]

            actions_possible = ''
            # LEFT
            if action == Action.LEFT.value:
                actions_possible = '<'

            # UP
            elif action == Action.UP.value:
                actions_possible = '^'

            # RIGHT
            elif action == Action.RIGHT.value:
                actions_possible = '>'

            # DOWN
            elif action == Action.DOWN.value:
                actions_possible = 'v'

            # TERMINAL
            elif action == Action.TERMINAL.value:
                actions_possible = '*'

            else:
                actions_possible = '!'

            visual += actions_possible + '|'
        visual += '\n'

    return visual

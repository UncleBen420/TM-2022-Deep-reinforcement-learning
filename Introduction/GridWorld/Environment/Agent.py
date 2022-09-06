"""
This file implement helper function for the different algorithms implemented
"""
import random
import numpy as np
from Environment.GridWorld import Action


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


def e_greedy(A, e):
    """
    this function is used to select an action based on the
    e-greedy function.
    :return: the chosen action
    """
    if np.random.binomial(1, e):
        return random.randrange(len(A))
    return np.argmax(A)


def get_e_greedy_prob(A, e):
    """
    Return the probability for a e-greedy policy
    :param A: the set of actions
    :param e: the epsilon parameter
    :return: the probability of the set of action
    """

    greedy_actions = A == np.max(A) # all actions with the maximal value
    nb_greedy = np.count_nonzero(greedy_actions) # number of max actions
    non_greedy_probability = e / len(A)
    return greedy_actions * ((1 - e) / nb_greedy) + non_greedy_probability


def get_greedy_prb(A):
    """
    Return the probability for a greedy policy a.k.a. a policy that
    priorities maximal expected value.
    :param A: the set of actions
    :return: the probability of the set of action
    """
    greedy_actions = A == np.max(A)
    nb_greedy = np.count_nonzero(greedy_actions)
    return greedy_actions * (1 / nb_greedy)


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

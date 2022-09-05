"""
This file implement helper function for the different algorithms implemented
"""
import random
import numpy as np
from Environment.GridWorld import Action


def incremental_mean(reward, state, action, nb_step, Q):
    """
    do the incremental mean between a reward and the expected value
    :param nb_step:
    :param state:
    :param action: the number of the action taken
    :param reward: is the reward given by the environment
    :return: the mean value
    """
    nb_step[state][action] += 1

    Q[state][action] += (reward - Q[state][action]) / nb_step[state][action]
    return nb_step, Q


def incremental_mean_V(reward, state, nb_step, V):
    """
    do the incremental mean between a reward and the expected value
    :param state:
    :param action: the number of the action taken
    :param reward: is the reward given by the environment
    :return: the mean value
    """
    nb_step[state] += 1

    V[state] += (reward - V[state]) / nb_step[state]
    return nb_step, V


def fit_and_evaluate(model, threshold=0.001, gamma=0.1):
    """

    :param model:
    :param threshold:
    :param gamma:
    :return:
    """
    V = []
    for episode in model.fit(verbose=True):
        V.append(evaluate_policy(model.environment, episode, threshold, gamma))

    return V


def e_greedy(A, e):
    """
    this function is used to select an action based on the
    e-greedy function.
    :return: the chosen action
    """
    if np.random.binomial(1, e):
        return random.randrange(len(A))
    return np.argmax(A)


def get_greedy_prob(A, e, nb_actions):
    """

    :param A:
    :param e:
    :param nb_actions:
    :return:
    """
    if np.random.binomial(1, e):
        return e / nb_actions
    return 1 - e + e / np.count_nonzero(A == np.max(A))


def init_policy(environment):
    """

    :param environment:
    :return:
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

    :param environment:
    :param policy:
    :param threshold:
    :param gamma:
    :return:
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

    :param environment:
    :param policy:
    :return:
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

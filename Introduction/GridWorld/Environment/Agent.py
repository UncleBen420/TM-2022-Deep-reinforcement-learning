import random
from abc import abstractmethod

import numpy as np

from Environment.GridWorld import Action


def incremental_mean(reward, state, action, nb_step, Q):
    """
    do the incremental mean between a reward and the espected value
    :param state:
    :param action: the number of the action taken
    :param reward: is the reward given by the environment
    :return: the mean value
    """
    nb_step[state][action] += 1

    Q[state][action] += (reward - Q[state][action]) / nb_step[state][action]
    return nb_step, Q


def fit_and_evaluate(model, threshold=0.001, gamma=0.1):
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


def init_policy(environment):
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

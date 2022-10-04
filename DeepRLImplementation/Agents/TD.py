"""
This file implement 3 learning agent algorithme: Q-learning, N-step Sarsa and Monte Carlo on policy.
"""
import random
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

class QLearning:
    """
    Implementation of the off-policy algorithme QLearning
    """

    def __init__(self, environment, Q,policy, alpha=0.1, gamma=0.1, episodes=100):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.Q = Q
        self.policy = policy
        self.policy.set_agent(self)
        # for evaluation
        self.V = np.zeros(environment.get_nb_state())
        # for visualisation
        self.policy_history = []

    def fit(self):
        """
        fit is called to train the agent on the environment
        :return: return the history of V and accumulated reward and the percent of boats left over the episodes
        """
        mean_v = []
        rewards = []
        boats_left = []

        with tqdm(range(self.episodes), unit="episode") as episode:
            for _ in episode:

                S = self.environment.reload_env()

                reward = 0

                while True:
                    # for visualisation
                    Q_pred = self.Q.predict(S)
                    A = self.policy.chose_action(Q_pred)

                    S_prime, R, is_terminal = self.environment.take_action(A)
                    Q_pred_prime = self.Q.predict(S_prime)
                    Q_pred += self.a * (R + self.gamma * np.max(self.Q[S_prime]) - Q_pred_prime)
                    # evaluation of V
                    self.V[S] += self.a * (R + self.gamma * self.V[S_prime] - self.V[S])
                    S = S_prime

                    reward += R

                    if is_terminal:
                        break

                mv = np.mean(self.V)
                bl = self.environment.get_marked_percent()
                st = self.environment.nb_actions_taken
                mean_v.append(mv)
                rewards.append(reward)
                boats_left.append(bl)

                episode.set_postfix(mean_v=mv, rewards=reward, boats_left=bl, steps_taken=st)

            return mean_v, rewards, boats_left

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


class Policy(ABC):
    """
    This class is an abstract class representing policy.
    Policy are given by the choice of action and the respected probability
    """

    def set_agent(self, agent):
        """
        Must be call by the agent to link it to the policy object
        otherwise methods cannot work.
        :param agent: the agent the object is attached
        """
        self.agent = agent

    @abstractmethod
    def chose_action(self, state, Q):
        """
        return the chosen action according the implemented policy
        :param state: state in which the agent is.
        :return: the chosen action
        """

    @abstractmethod
    def probability(self, state, Q):
        """
        Return the probability of each action for this state according
        to the implemented policy.
        :param state: state in which the agent is.
        :return: the probability for each action
        """


class E_Greedy(Policy):
    """
    Implementation of an e-greedy policy
    """

    def __init__(self, epsilon):
        super().__init__()
        self.e = epsilon

    def chose_action(self, state, Q):
        """
        return the chosen action according the e-greedy policy
        :param Q:
        :param state: state in which the agent is.
        :return: the chosen action
        """
        if np.random.binomial(1, self.e):
            return random.randrange(self.agent.nb_action)
        return np.argmax(Q)

    def probability(self, state, Q):
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



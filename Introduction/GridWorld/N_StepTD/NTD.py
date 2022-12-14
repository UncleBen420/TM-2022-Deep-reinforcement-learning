"""
This file contains implementation of n-step TD algorithms
"""
import numpy as np
from Environment import Agent
from Environment.GridWorld import Action


class OffPolicyNStepSarsa:
    """
    Implementation of the off policy n-step Sarsa
    Contrary to the RL book it use a queue to store the trajectory
    """

    def __init__(self, environment, policy, alpha=0.1, gamma=0.1,
                 episodes=1000, patience=200, steps=4):
        self.environment = environment
        self.a = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.n = steps
        self.Q = np.zeros((environment.size * environment.size, environment.nb_action))
        self.behaviour = policy
        self.behaviour.set_agent(self)
        self.target = Agent.Greedy()
        self.target.set_agent(self)
        self.patience = patience
        # for evaluation
        self.V = np.zeros((environment.size * environment.size))

    def fit(self, verbose=False):
        """
        fit is called to train the agent on the environment
        :param verbose: True to have a history of the V function and the accumulated reward
        :return: return the history of V and accumulated reward if verbose == True
        """
        if verbose:
            history = []
            rewards = []

        for _ in range(self.episodes):

            queue = []
            if verbose:
                reward = 0

            S = 0  # initial state
            queue.append((S, self.behaviour.chose_action(S), 0))
            start = False
            for _ in range(self.patience):

                S, A, _ = queue[-1]
                if not self.environment.states[S].value['is_terminal']:

                    S_prime = self.environment.get_next_state(S, Action(A))
                    R_prime = self.environment.get_reward(S, Action(A), S_prime)
                    A_prime = self.behaviour.chose_action(S_prime)
                    queue.append((S_prime, A_prime, R_prime))

                    if verbose:
                        reward += R_prime

                if start or len(queue) > self.n:
                    start = True
                    S, A, _ = queue.pop(0)
                    if len(queue) <= 1:
                        break

                    p = 1.
                    G = 0.
                    t = 0
                    for St, At, Rt in queue:

                        if not self.environment.states[St].value['is_terminal']:
                            p *= self.target.probability(St)[At] / self.behaviour.probability(St)[At]
                        G += self.gamma ** t * Rt
                        t += 1

                    Gv = G

                    if not self.environment.states[queue[-1][0]].value['is_terminal']:
                        Gv += self.gamma ** self.n * self.V[queue[-1][0]]
                        G += self.gamma ** self.n * self.Q[queue[-1][0]][queue[-1][1]]

                    self.Q[S][A] += self.a * p * (G - self.Q[S][A])
                    self.V[S] += self.a * p * (Gv - self.V[S])

            if verbose:
                history.append(self.V.copy())
                rewards.append(reward)
        if verbose:
            return history, rewards

    def get_policy(self):
        """
        this method allow to get the greedy policy for the agent
        :return: the best action for every state
        """
        return np.argmax(self.Q, axis=1)

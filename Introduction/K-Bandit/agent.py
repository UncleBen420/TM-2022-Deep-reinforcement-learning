import random
import numpy as np

class RandomAgent:
    def __init__(self, K):
        self.K = K

    def choose(self):
        return random.randrange(self.K)

    def update(self, reward, a):
        ''' value function update'''
        # random agent do not learn
        pass


class RLAgent:
    def __init__(self, K, action_selection_function = "egreedy", e=0.1, c=2):
        self.e = e
        self.K = K
        self.c = c
        self.nb_total_step = 0
        self.nb_step = np.zeros(K)
        self.expected_values = np.zeros(K)
        if action_selection_function == "ucb":
            self.action_selection_function = self.ucb
        else:
            self.action_selection_function = self.e_greedy

    def e_greedy(self):
        if np.random.binomial(1, self.e):
            return random.randrange(self.K)
        else:
            return np.argmax(self.expected_values)

    def ucb(self):
        print(self.expected_values)
        if np.count_nonzero(self.expected_values==0):
            print("yo")
            return np.where(self.expected_values == 0)[0][0]
        else:
            return np.argmax(self.expected_values + self.c * np.sqrt(np.log(self.nb_total_step) / self.nb_step))

    def incremental_mean(self, reward, a):
        return self.expected_values[a] + (reward - self.expected_values[a]) / self.nb_step[a]

    def choose(self):
        return self.action_selection_function()

    def update(self, reward, a):
        ''' value function update'''
        self.nb_step[a] += 1
        self.nb_total_step += 1
        self.expected_values[a] = self.incremental_mean(reward, a)

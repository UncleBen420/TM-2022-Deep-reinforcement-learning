'''
the goal of this file is to provide a k-bandits problem.
inspired by https://www.dominodatalab.com/blog/k-armed-bandit-problem
'''
import numpy as np

class BernoulliBandit:
    '''this class provide a simulation for a bandit with probability p and binomial distribution'''

    def __init__(self, p, verbose=True):
        self.p = p
        if verbose:
            print("Creating BernoulliBandit with p = {:.2f}".format(p))

    def pull(self):
        return np.random.binomial(1, self.p)


class BanditsGame:

    def __init__(self, K, T, agent, verbose=True):

        self.T = T
        self.K = K
        self.agent = agent
        self.bandits = [BernoulliBandit(np.random.uniform(), verbose) for i in range(K)]
        self.verbose = verbose

    def run(self):

        results = np.zeros((self.T))

        for t in range(self.T):
            k = self.agent.choose()
            results[t] = self.bandits[k].pull()
            if self.verbose:
                print("T={} \t Playing bandit {} \t Reward is {:.2f}".format(t, k, results[t]))

        return results
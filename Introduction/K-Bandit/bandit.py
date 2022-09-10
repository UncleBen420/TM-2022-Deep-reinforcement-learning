'''
the goal of this file is to provide a k-bandits problem.
inspired by https://www.dominodatalab.com/blog/k-armed-bandit-problem
'''
import numpy as np


class BanditsGame:
    ''' this class provide an environment for an agent to be exposed to the k-bandit problem'''
    def __init__(self, k, timestep, verbose=True):

        self.timestep = timestep
        self.agent = None
        self.bandits = np.random.uniform(0, 1, k)
        self.verbose = verbose

    def pull(self, bandit):
        ''' give a 1 or 0 according to the probability of one of the bandit'''
        return np.random.binomial(1, self.bandits[bandit])

    def run(self):
        ''' run the full simulation'''

        results = np.zeros(self.timestep)

        for i in range(self.timestep):
            chosen_bandit = self.agent.choose()
            results[i] = self.pull(chosen_bandit)
            self.agent.update(results[i], chosen_bandit)

            if self.verbose:
                print("T={} \t Playing bandit {} \t Reward is {:.2f}"
                      .format(i, chosen_bandit, results[i]))

        return results

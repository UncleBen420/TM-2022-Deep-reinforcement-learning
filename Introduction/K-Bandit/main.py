'''
main
'''
import matplotlib.pyplot as plt
import numpy as np
import agent
from bandit import BanditsGame


def incremental_mean(timeseries):
    '''incremental_mean'''
    means = np.zeros((timeseries.size))
    means[0] = timeseries[0]
    for i in range(1, timeseries.size):
        means[i] = means[i-1] + (timeseries[i] - means[i-1])/(i+1)
    return means



def run_simulation(agt, k=5, timestep=1000, verbose=True):
    '''run_simulation'''
    bandit_game = BanditsGame(k, timestep, agt, verbose)
    results = incremental_mean(bandit_game.run())
    plt.plot(results)
    plt.show()


if __name__ == '__main__':

    VERBOSE = False
    K = 5
    TIME_STEP = 1000
    RAND_AGT = agent.RandomAgent(K)
    run_simulation(RAND_AGT, K, TIME_STEP, VERBOSE)

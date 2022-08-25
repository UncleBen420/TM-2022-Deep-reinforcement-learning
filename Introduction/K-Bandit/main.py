'''
main
'''
import matplotlib.pyplot as plt
import numpy as np
import agent
from bandit import BanditsGame


def incremental_mean(timeseries):
    '''incremental_mean'''
    means = np.zeros(timeseries.size)
    means[0] = timeseries[0]
    for i in range(1, timeseries.size):
        means[i] = means[i-1] + (timeseries[i] - means[i-1])/(i+1)
    return means


def run_simulation(agt, label, k=5, timestep=1000, verbose=True):
    '''run_simulation'''
    bandit_game = BanditsGame(k, timestep, agt, verbose)
    results = incremental_mean(bandit_game.run())
    plt.plot(results, label=label)


if __name__ == '__main__':

    VERBOSE = False
    K = 5
    EPSILON = 0.1
    CONFIDENCE = 0.5
    INITIALISATION = 1
    TIME_STEP = 1000
    RAND_AGT = agent.RandomAgent(K)
    EGREEDY_AGT = agent.RLAgent(K, e=EPSILON, action_selection_function="egreedy")
    UCB_AGT = agent.RLAgent(K, c=CONFIDENCE, action_selection_function="ucb")

    run_simulation(RAND_AGT, "random", K, TIME_STEP, VERBOSE)
    run_simulation(EGREEDY_AGT, "e-greedy", K, TIME_STEP, VERBOSE)
    run_simulation(UCB_AGT, "UCB", K, TIME_STEP, VERBOSE)
    plt.legend()
    plt.show()

# This is a sample Python script.
from bandit import BanditsGame
import agent
import matplotlib.pyplot as plt
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def incremental_mean(timeseries):
    means = np.zeros((timeseries.size))
    means[0] = timeseries[0]
    for i in range(1,timeseries.size):
        means[i] = means[i-1] + (timeseries[i] - means[i-1])/(i+1)
    return means



def run_simulation(agt, k=5, timestep=1000, verbose=True):
    bandit_game = BanditsGame(k, timestep, agt, verbose)
    results = incremental_mean(bandit_game.run())
    plt.plot(results)
    plt.show()


if __name__ == '__main__':

    VERBOSE = False
    K = 5
    TIME_STEP = 1000
    randomAgent = agent.RandomAgent(K)
    run_simulation(randomAgent, K, TIME_STEP, VERBOSE)

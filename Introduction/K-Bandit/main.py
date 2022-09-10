"""
The goal of this application is to provide a comparison
study over the problem of the k-armed bandit.
3 differents agent are compared: Random, e-greedy and UCB.
The metric chose to compare them is the reward mean over
the time step.
"""
import matplotlib.pyplot as plt
import numpy as np
import agent
from bandit import BanditsGame


def incremental_mean(timeseries):

    """this function take a sequence of number and produce
    the mean of each number, and it's predecessor.
    :param timeseries: the reward over the different iteration
    """
    means = np.zeros_like(timeseries)
    means[0] = timeseries[0]
    for i in range(1, timeseries.size):
        means[i] = means[i-1] + (timeseries[i] - means[i-1])/(i+1)
    return means


def run_simulation(environment, agt, history):
    """
    run the bandit problem with the given agent
    :param environment: the bandit problem class
    :param agt: the agent learning on the problem
    :param history: an array containing the history of olders runs
    """
    environment.agent = agt
    history.append(incremental_mean(environment.run()))


if __name__ == '__main__':

    VERBOSE = False
    K = 5
    EPSILON = 0.1
    CONFIDENCE = 0.6
    INITIALISATION = 1
    TIME_STEP = 1000
    FOLD = 5

    RANDOM_HISTORY = []
    EGREEDY_HISTORY = []
    UCB_HISTORY = []

    for _ in range(FOLD):
        RAND_AGT = agent.RandomAgent(K)
        EGREEDY_AGT = agent.RLAgent(K, e=EPSILON, action_selection_function="egreedy")
        UCB_AGT = agent.RLAgent(K, c=CONFIDENCE, action_selection_function="ucb")
        BANDIT = BanditsGame(K, TIME_STEP, VERBOSE)
        run_simulation(BANDIT, RAND_AGT, RANDOM_HISTORY)
        run_simulation(BANDIT, EGREEDY_AGT, EGREEDY_HISTORY)
        run_simulation(BANDIT, UCB_AGT, UCB_HISTORY)

    fig, axs = plt.subplots(nrows=3, ncols=1)
    fig.suptitle("Runs by agent")
    fig.tight_layout(h_pad=3, w_pad=3)

    axs[0].set_title('Random agent')
    axs[0].set_xlabel('nb iteration')
    axs[0].set_ylabel('mean of the reward')

    for i in range(FOLD):
        label = "run " + str(i)
        axs[0].plot(RANDOM_HISTORY[i], label=label)

    axs[1].set_title('e-greedy agent')
    axs[1].set_xlabel('nb iteration')
    axs[1].set_ylabel('mean of the reward')

    for i in range(FOLD):
        label = "run " + str(i)
        axs[1].plot(EGREEDY_HISTORY[i], label=label)

    axs[2].set_title('agent UCB')
    axs[2].set_xlabel('nb iteration')
    axs[2].set_ylabel('mean of the reward')

    for i in range(FOLD):
        label = "run " + str(i)
        axs[2].plot(UCB_HISTORY[i], label=label)

    plt.legend()
    plt.show()

    plt.plot(np.mean(RANDOM_HISTORY, axis=0), label="random agent")
    plt.plot(np.mean(EGREEDY_HISTORY, axis=0), label="e-greedy agent")
    plt.plot(np.mean(UCB_HISTORY, axis=0), label="ucb agent")
    plt.title("mean of the runs")
    plt.xlabel('nb iteration')
    plt.ylabel('mean of the reward')
    plt.legend()
    plt.show()



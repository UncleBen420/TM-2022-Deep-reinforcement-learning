"""
This file provide a comparison of a few RL agents over a grid world.
"""
import matplotlib.pyplot as plt
import numpy as np
from DynamicProgramming.DP import DP
from Environment.GridWorld import Board
from Environment.RandomPolicyAgent import RandomPolicyAgent
from MonteCarlo.MC import MCES
from TemporalDifference.TD import Sarsa, QLearning, DoubleQLearning, ExpectedSarsa


def plot_result(environment, agent, name):
    '''
    Fit the agent over the environment and plot the result
    :param environment: the environment in which the agent can evolve
    :param agent: agent object with function fit.
    :param name: the name of the algorithme that will be implemented
    :return: the name and the mean v over n episode.
    '''
    expected_rewards = agent.fit(True)
    random_agent = RandomPolicyAgent(environment)
    random_v = random_agent.generate_t_policy_validations(len(expected_rewards))

    fig, axs = plt.subplots(nrows=3, ncols=1)
    fig.suptitle(name)
    fig.tight_layout(h_pad=3, w_pad=3)

    mean_v = np.mean(expected_rewards, axis=1)
    random_mean_v = np.mean(random_v, axis=1)

    axs[0].plot(expected_rewards)
    axs[0].set_title('Agent V function')
    axs[0].set_xlabel('nb iteration')
    axs[0].set_ylabel('V for every state')

    axs[1].plot(random_v)
    axs[1].set_title('Random V function')
    axs[1].set_xlabel('nb iteration')
    axs[1].set_ylabel('V for every state')

    axs[2].plot(mean_v, label=name)
    axs[2].plot(random_mean_v, label="Random")
    axs[2].set_title('Mean V of the 2 agents')
    axs[2].set_xlabel('nb iteration')
    axs[2].set_ylabel('mean of V')

    plt.legend()
    plt.show()

    print("Mean of V:{0} for {1} iterations.".format(mean_v[-1], len(mean_v)))
    return {'mean': mean_v, 'name': name}


if __name__ == '__main__':
    BOARD = Board(nb_trap=30, size=10)
    print(BOARD.render_board())

    AGENT_DP = DP(BOARD, 0.001, 0.1)
    AGENT_MC = MCES(BOARD, 1000, 0.1, 120)
    AGENT_SA = Sarsa(BOARD, 0.1, 0.1, 0.1, 100)
    AGENT_QL = QLearning(BOARD, 0.1, 0.1, 0.1, 100)
    AGENT_DQ = DoubleQLearning(BOARD, 0.1, 0.1, 0.1, 100)
    AGENT_ES = ExpectedSarsa(BOARD, 0.1, 0.1, 0.1, 100)

    SUMMARY = [plot_result(BOARD, AGENT_DP, "Dynamic Programming"),
               plot_result(BOARD, AGENT_MC, "Monte Carlo ES"),
               plot_result(BOARD, AGENT_SA, "Sarsa"),
               plot_result(BOARD, AGENT_QL, "Q-learning"),
               plot_result(BOARD, AGENT_DQ, "Double Q-learning"),
               plot_result(BOARD, AGENT_ES, "Expected Sarsa")]

    for algo in SUMMARY:
        plt.plot(algo['mean'], label=algo['name'])

    plt.legend()
    plt.show()

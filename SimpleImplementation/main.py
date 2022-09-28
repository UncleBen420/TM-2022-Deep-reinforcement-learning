from time import sleep

import cv2
import re

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from DummyExperiment import EnvironmentDummySoft
from DummyExperiment import EnvironmentDummyHard

from DummyExperiment.Agents import DummyAgent, QLearning, E_Greedy, MonteCarloOnPolicy, NStepSarsa, ExploitAgent, UCB


def plot_result(agent, name):
    """
    Fit the agent over the environment and plot the result
    :param environment: the environment in which the agent can evolve
    :param agent: agent object with function fit.
    :param name: the name of the algorithme that will be implemented
    :return: the name and the mean v over n episode.
    """

    mean_v, sum_of_reward = agent.fit()

    fig, axs = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(name)
    fig.tight_layout(h_pad=3, w_pad=3)

    axs[0].plot(mean_v, label=name)
    axs[0].set_title('Mean V of the agent')
    axs[0].set_xlabel('nb iteration')
    axs[0].set_ylabel('mean of V')
    axs[0].legend()

    axs[1].plot(sum_of_reward)
    axs[1].set_title('Agent accumulated rewards')
    axs[1].set_xlabel('nb iteration')
    axs[1].set_ylabel('sum of reward')

    plt.show()

    print("Mean of V:{0} for {1} iterations.".format(mean_v[-1], len(mean_v)))
    return {'mean': mean_v, 'reward': sum_of_reward, 'name': name}


if __name__ == '__main__':

    de = EnvironmentDummySoft.DummyEnv(nb_max_actions=1000)
    #de = EnvironmentDummyHard.DummyEnv(nb_max_actions=30000)

    de.init_env()
    print(de.render_grid(de.grid))

    dm = DummyAgent(de)
    #plot_result(dm, "Random")

    #plt.imshow(de.render_board_img(de.marked_map, [1, 0, 0]))
    #de.get_gif_trajectory("dummy_agent.gif")
    #plt.show()

    ql = QLearning(de, E_Greedy(0.05), episodes=1000)
    plot_result(ql, "Q-Learning")
    plt.imshow(de.render_board_img(de.marked_map, [1, 0, 0]))
    de.get_gif_trajectory("dummy_agent_ql.gif")
    plt.show()

    #mc = MonteCarloOnPolicy(de, E_Greedy(0.05), episodes=1000)
    #plot_result(mc, "MC")
    #plt.imshow(de.render_board_img(de.marked_map, [1, 0, 0]))
    #de.get_gif_trajectory("dummy_agent_mc.gif")
    #plt.show()

    #ns = NStepSarsa(de, E_Greedy(0.1), episodes=1000, steps=4)
    #plot_result(ns, "ns")
    #plt.imshow(de.render_board_img(de.marked_map, [1, 0, 0]))
    #de.get_gif_trajectory("dummy_agent_ns.gif")
    #plt.show()

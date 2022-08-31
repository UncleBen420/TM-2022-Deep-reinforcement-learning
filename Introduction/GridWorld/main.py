import numpy as np

from DynamicProgramming.DP import DP
from Environment.GridWorld import Board
import matplotlib.pyplot as plt
from Environment.RandomPolicyAgent import RandomPolicyAgent
from MonteCarlo.MonteCarlo import MCES, OnPolicyMC


def plot_result(v, random_v):
    fig, axs = plt.subplots(nrows=3, ncols=1)

    mean_v = np.mean(v, axis=1)
    random_mean_v = np.mean(random_v, axis=1)

    axs[0].plot(v)
    axs[1].plot(random_v)
    axs[2].plot(mean_v, label="DP")
    axs[2].plot(random_mean_v, label="Random")
    plt.legend()
    plt.show()

    print("Mean of V:{0} for {1} iterations.".format(mean_v[-1], len(mean_v)))


def evaluate_DP(grid_world, gamma, threshold):
    dp = DP(grid_world, threshold=threshold, gamma=gamma)
    history, v = dp.policy_iteration()

    random_agent = RandomPolicyAgent(grid_world)
    random_v = random_agent.generate_t_policy_validations(np.array(v).shape[0])

    return plot_result(v, random_v)


def evaluate_MC(grid_world, gamma, threshold, patience, episodes):
    mces = MCES(grid_world, episodes=episodes, gamma=gamma, patience=patience)
    v = mces.fit_and_evaluate(threshold)

    random_agent = RandomPolicyAgent(grid_world)
    random_v = random_agent.generate_t_policy_validations(np.array(v).shape[0])

    plot_result(v, random_v)


if __name__ == '__main__':
    board = Board(nb_trap=30, size=10)
    print(board.render_board())

    #evaluate_DP(board, threshold=0.001, gamma=0.1)
    #evaluate_MC(board, threshold=0.001, gamma=0.1, patience=120, episodes=1000)
    opmc = OnPolicyMC(board)
    opmc.fit()
    print(board.render_policy(opmc.get_policy()))

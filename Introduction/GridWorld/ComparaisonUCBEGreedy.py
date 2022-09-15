import numpy as np
from matplotlib import pyplot as plt

from Environment import Agent
from Environment.GridWorld import Board
from TemporalDifference.TD import QLearning

if __name__ == '__main__':
    BOARD = Board(nb_trap=30, size=10)
    print(BOARD.render_board())

    EPSILON = [0.05, 0.1, 0.2]
    C = [0.1, 0.2, 0.8]
    FOLD = 5

    fig, axs = plt.subplots(nrows=2, ncols=1)
    fig.suptitle("Epsilon influence")
    fig.tight_layout(h_pad=3, w_pad=3)

    for i in range(len(EPSILON)):
        V_UCB = []
        REWARDS_UCB = []
        V_EG = []
        REWARDS_EG = []
        for f in range(FOLD):
            AGENT_UCB = QLearning(BOARD, Agent.UCB(C[i]), 0.1, 0.1, 100, 10000)
            AGENT_EG = QLearning(BOARD, Agent.E_Greedy(EPSILON[i]), 0.1, 0.1, 100, 10000)

            expected_rewards, sum_of_reward = AGENT_UCB.fit(True)
            V_UCB.append(np.mean(expected_rewards, axis=1))
            REWARDS_UCB.append(sum_of_reward)

            expected_rewards, sum_of_reward = AGENT_EG.fit(True)
            V_EG.append(np.mean(expected_rewards, axis=1))
            REWARDS_EG.append(sum_of_reward)

        label = "epsilon: " + str(EPSILON[i])

        axs[0].plot(np.mean(V_EG, axis=0), label=label)
        axs[0].set_title('Agent V function')
        axs[0].set_xlabel('nb iteration')
        axs[0].set_ylabel('V mean of V')

        axs[1].plot(np.mean(REWARDS_EG, axis=0), label=label)
        axs[1].set_title('Agent accumulated rewards')
        axs[1].set_xlabel('nb iteration')
        axs[1].set_ylabel('accumulated rewards')

        label = "c: " + str(C[i])

        axs[0].plot(np.mean(V_UCB, axis=0), label=label)
        axs[0].set_title('Agent V function')
        axs[0].set_xlabel('nb iteration')
        axs[0].set_ylabel('V mean of V')

        axs[1].plot(np.mean(REWARDS_UCB, axis=0), label=label)
        axs[1].set_title('Agent accumulated rewards')
        axs[1].set_xlabel('nb iteration')
        axs[1].set_ylabel('accumulated rewards')

    plt.legend()
    plt.show()

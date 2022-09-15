import numpy as np
from matplotlib import pyplot as plt

from Environment import Agent
from Environment.GridWorld import Board
from N_StepTD.NTD import OffPolicyNStepSarsa
from TemporalDifference.TD import QLearning

if __name__ == '__main__':
    BOARD = Board(nb_trap=30, size=10)
    print(BOARD.render_board())

    N = [1, 2, 4, 8]
    FOLD = 5

    fig, axs = plt.subplots(nrows=2, ncols=1)
    fig.suptitle("Epsilon influence")
    fig.tight_layout(h_pad=3, w_pad=3)

    for n in N:
        V = []
        REWARDS = []
        for f in range(FOLD):
            AGENT_NS = OffPolicyNStepSarsa(BOARD, Agent.E_Greedy(0.05), 0.1, 0.1, 100, 10000, steps=n)
            expected_rewards, sum_of_reward = AGENT_NS.fit(True)
            V.append(np.mean(expected_rewards, axis=1))
            REWARDS.append(sum_of_reward)

        axs[0].plot(np.mean(V, axis=0), label=n)
        axs[0].set_title('Agent V function')
        axs[0].set_xlabel('nb iteration')
        axs[0].set_ylabel('V mean of V')

        axs[1].plot(np.mean(REWARDS, axis=0), label=n)
        axs[1].set_title('Agent accumulated rewards')
        axs[1].set_xlabel('nb iteration')
        axs[1].set_ylabel('accumulated rewards')

    plt.legend()
    plt.show()

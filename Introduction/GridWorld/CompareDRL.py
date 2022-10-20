"""
The goal of this program is to compare deep Rl algorithme with RL algorithms.
"""


import numpy as np
from matplotlib import pyplot as plt

from DRL.DeepAgent import DQLearning
from DRL.Reinforce import Reinforce
from Environment import Agent
from Environment.GridWorld import Board
from TemporalDifference.TD import QLearning

if __name__ == '__main__':
    BOARD = Board(nb_trap=30, size=10)
    print(BOARD.render_board())

    FIGURE, AXIS = plt.subplots(nrows=3, ncols=1)
    FIGURE.suptitle("Deep algorithms comparison")
    FIGURE.tight_layout(h_pad=3, w_pad=3)

    DQL = DQLearning(BOARD, episodes=1000)
    REINFORCE = Reinforce(BOARD, BOARD.size * BOARD.size, episodes=1000)
    AGENT_QLE = QLearning(BOARD, Agent.E_Greedy(0.05), 0.1, 0.1, 1000, 1000)

    V, REWARDS = AGENT_QLE.fit(True)
    MEAN_V = np.mean(V, axis=1)
    AXIS[0].plot(MEAN_V, label="Q-Learning")
    AXIS[1].plot(REWARDS, label="Q-Learning")

    REWARDS, LOSS = REINFORCE.fit()
    AXIS[1].plot(REWARDS, label="reinforce")
    AXIS[2].plot(LOSS, label="reinforce")

    V, REWARDS, LOSS = DQL.fit(True)

    AXIS[0].plot(V, label="Deep Q-Learning")
    AXIS[0].set_title('Agent V function')
    AXIS[0].set_xlabel('nb iteration')
    AXIS[0].set_ylabel('V mean of V')
    AXIS[0].legend()

    AXIS[1].plot(REWARDS, label="Deep Q-Learning")
    AXIS[1].set_title('Agent accumulated rewards')
    AXIS[1].set_xlabel('nb iteration')
    AXIS[1].set_ylabel('accumulated rewards')
    AXIS[1].legend()

    AXIS[2].plot(LOSS, label="Deep Q-Learning")
    AXIS[2].set_title('Agent loss')
    AXIS[2].set_xlabel('nb iteration')
    AXIS[2].set_ylabel('loss')
    AXIS[2].legend()

    plt.show()

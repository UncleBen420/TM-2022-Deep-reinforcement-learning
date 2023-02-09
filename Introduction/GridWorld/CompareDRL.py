"""
The goal of this program is to compare deep Rl algorithme with RL algorithms.
"""
import imageio
import numpy as np
from matplotlib import pyplot as plt

from DRL.DeepAgent import DQLearning
from DRL.Reinforce import Reinforce
from Environment import Agent
from Environment.GridWorld import Board
from TemporalDifference.TD import QLearning

def make_gif_trajectory(environment, trajectory, name):
    """
    This function allow the user to create a gif of all the moves the
    agent has made along the episodes
    :param environment: the environment on which the agent evolve
    :param trajectory: the trajectory that the agent has take
    :param name: the name of the gif file
    """
    frames = []
    for t in trajectory:
        frames.append(environment.render_board_img(t))
    imageio.mimsave(name, frames, duration=0.05)


if __name__ == '__main__':
    BOARD = Board(nb_trap=30, size=10)
    print(BOARD.render_board())

    FIGURE, AXIS = plt.subplots(nrows=2, ncols=1)
    FIGURE.suptitle("Deep algorithms comparison")
    FIGURE.tight_layout(h_pad=3, w_pad=3)

    DQL = DQLearning(BOARD, alpha=0.0001, gamma=0.3, episodes=2000, patience=100)
    REINFORCE = Reinforce(BOARD, BOARD.size * BOARD.size, learning_rate=0.0001, episodes=1000, patience=100, gamma=0.3)
    AGENT_QLE = QLearning(BOARD, Agent.E_Greedy(0.3), 0.3, 0.3, 1000, 100)

    _, REWARDS = AGENT_QLE.fit(True, trajectory=True)
    AXIS[0].plot(REWARDS, label="Q-Learning")

    make_gif_trajectory(BOARD, AGENT_QLE.trajectory[-100:], "Q-learning.gif")

    REWARDS, LOSS = REINFORCE.fit(trajectory=True)
    AXIS[0].plot(REWARDS, label="reinforce")
    AXIS[1].plot(LOSS, label="reinforce")

    make_gif_trajectory(BOARD, REINFORCE.trajectory[-100:], "reinforce.gif")

    AXIS[0].set_title('Agent accumulated rewards')
    AXIS[0].set_xlabel('nb iteration')
    AXIS[0].set_ylabel('accumulated rewards')
    AXIS[0].legend()
    AXIS[1].set_title('Agent loss')
    AXIS[1].set_xlabel('nb iteration')
    AXIS[1].set_ylabel('loss')
    AXIS[1].legend()

    plt.show()

    FIGURE, AXIS = plt.subplots(nrows=2, ncols=1)
    FIGURE.suptitle("Deep algorithms comparison")
    FIGURE.tight_layout(h_pad=3, w_pad=3)


    _, REWARDS = AGENT_QLE.fit(True, trajectory=True)

    AXIS[0].plot(REWARDS, label="Q-Learning")

    REWARDS, LOSS = DQL.fit(True, trajectory=True)
    AXIS[0].plot(REWARDS, label="Deep Q-Learning")
    AXIS[1].plot(LOSS, label="Deep Q-Learning")

    make_gif_trajectory(BOARD, DQL.trajectory[-100:], "Deep Q-learning.gif")

    AXIS[0].set_title('Agent accumulated rewards')
    AXIS[0].set_xlabel('nb iteration')
    AXIS[0].set_ylabel('accumulated rewards')
    AXIS[0].legend()
    AXIS[1].set_title('Agent loss')
    AXIS[1].set_xlabel('nb iteration')
    AXIS[1].set_ylabel('loss')
    AXIS[1].legend()

    plt.show()

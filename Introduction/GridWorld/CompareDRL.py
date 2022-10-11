import numpy as np
from matplotlib import pyplot as plt

from DRL.DeepAgent import DQLearning
from DRL.Reinforce import Reinforce
from Environment import Agent
from Environment.GridWorld import Board
from N_StepTD.NTD import OffPolicyNStepSarsa
from TemporalDifference.TD import QLearning

if __name__ == '__main__':
    BOARD = Board(nb_trap=30, size=10)
    print(BOARD.render_board())

    print(BOARD.get_env_vision(4))

    fig, axs = plt.subplots(nrows=3, ncols=1)
    fig.suptitle("Epsilon influence")
    fig.tight_layout(h_pad=3, w_pad=3)

    DQL = DQLearning(BOARD, episodes=1000)
    Rein = Reinforce(BOARD, BOARD.size * BOARD.size, episodes=1000)
    AGENT_QLE = QLearning(BOARD, Agent.E_Greedy(0.05), 0.1, 0.1, 1000, 1000)

    V, sum_of_reward = AGENT_QLE.fit(True)
    mean_v = np.mean(V, axis=1)
    axs[0].plot(mean_v, label="Q-Learning")
    axs[1].plot(sum_of_reward, label="Q-Learning")

    sum_of_reward, loss = Rein.fit()
    axs[1].plot(sum_of_reward, label="reinforce")
    axs[2].plot(loss, label="reinforce")

    V, sum_of_reward, loss = DQL.fit(True)

    axs[0].plot(V, label="Deep Q-Learning")
    axs[0].set_title('Agent V function')
    axs[0].set_xlabel('nb iteration')
    axs[0].set_ylabel('V mean of V')
    axs[0].legend()

    axs[1].plot(sum_of_reward, label="Deep Q-Learning")
    axs[1].set_title('Agent accumulated rewards')
    axs[1].set_xlabel('nb iteration')
    axs[1].set_ylabel('accumulated rewards')
    axs[1].legend()

    axs[2].plot(loss, label="Deep Q-Learning")
    axs[2].set_title('Agent loss')
    axs[2].set_xlabel('nb iteration')
    axs[2].set_ylabel('loss')
    axs[2].legend()

    plt.show()

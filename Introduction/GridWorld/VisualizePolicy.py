"""
This file provide a comparison between the policy of DP and policy of QL over
different episode
"""
import imageio
import matplotlib.pyplot as plt
from DynamicProgramming.DP import DP
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


def make_gif_policy_evolution(environment, policies, name):
    """
    This function allow the user to create a gif of all the policy the
    agent has made along the episodes
    :param environment: the environment on which the agent evolve
    :param policies: policy at the end of each episode
    :param name: the name of the gif file
    :return:
    """
    frames = []
    for p in policies:
        frames.append(Agent.render_policy_img(environment, p))
    imageio.mimsave(name, frames, duration=0.05)


if __name__ == '__main__':
    BOARD = Board(nb_trap=30, size=10)
    fig, axs = plt.subplots(nrows=3, ncols=2)
    fig.suptitle("Policy Visualisation")
    fig.tight_layout(h_pad=3, w_pad=3)

    axs[0][0].imshow(BOARD.render_board_img(0), cmap='gray')
    axs[0][0].set_title('Grid World')

    AGENT_DP = DP(BOARD, 0.001, 0.1)
    AGENT_DP.fit()
    axs[0][1].imshow(Agent.render_policy_img(BOARD, AGENT_DP.policy), cmap='gray')
    axs[0][1].set_title('DP Policy')
    # experience nb1
    AGENT_QL = QLearning(BOARD, Agent.E_Greedy(0.1), 0.1, 0.1, 100, 10000)
    AGENT_QL.fit()
    axs[1][0].imshow(Agent.render_policy_img(BOARD, AGENT_QL.get_policy()), cmap='gray')
    axs[1][0].set_title('QL policy after 100 episode')
    # experience nb2
    AGENT_QL = QLearning(BOARD, Agent.E_Greedy(0.1), 0.1, 0.1, 200, 10000)
    AGENT_QL.fit()
    axs[1][1].imshow(Agent.render_policy_img(BOARD, AGENT_QL.get_policy()), cmap='gray')
    axs[1][1].set_title('QL policy after 200 episode')
    # experience nb3
    AGENT_QL = QLearning(BOARD, Agent.E_Greedy(0.1), 0.1, 0.1, 400, 10000)
    AGENT_QL.fit()
    axs[2][0].imshow(Agent.render_policy_img(BOARD, AGENT_QL.get_policy()), cmap='gray')
    axs[2][0].set_title('QL policy after 400 episode')
    # experience nb4
    AGENT_QL = QLearning(BOARD, Agent.E_Greedy(0.1), 0.1, 0.1, 800, 10000)
    AGENT_QL.fit(trajectory=True)
    axs[2][1].imshow(Agent.render_policy_img(BOARD, AGENT_QL.get_policy()), cmap='gray')
    axs[2][1].set_title('QL policy after 800 episode')

    # make_gif_trajectory(BOARD, AGENT_QL.trajectory[:1000], "early_steps.gif")
    # make_gif_trajectory(BOARD, AGENT_QL.trajectory[-100:], "end_steps.gif")
    # make_gif_policy_evolution(BOARD, AGENT_QL.policy_history, "policy_evolution.gif")

    plt.show()

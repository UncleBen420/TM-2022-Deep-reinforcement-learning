"""
The goal of this program is to allow user to evaluate 3 different RL algorithm on the dummy environment.
"""
import numpy as np
from matplotlib import pyplot as plt

import environment
from reinforce import Reinforce

class Evaluator:

    def init_plot(self):
        self.fig, self.axs = plt.subplots(nrows=5, ncols=1, layout="constrained")

    def fit(self, agent, name):
        '''
        This method allow user to evaluate a type of agent with a policy with the hyerparameter specified with __init__
        :param agent: the class of agent that must be trained
        :param policy: the class of policy that will be used by the agent
        :param name: the name of the agent (is used for the visualisation)
        '''
        print("starting fitting of {0}".format(name))

        losses, rewards, nb_mark, nb_action, successful_marks = agent.fit()

        self.fig.suptitle("hyper parameters selections for {0}".format(name))

        self.axs[0].plot(rewards, label=name)
        self.axs[0].set_title('Sum of rewards')
        self.axs[0].set_xlabel('nb iteration')
        self.axs[0].set_ylabel('rewards')

        self.axs[1].plot(nb_action, label=name)
        self.axs[1].set_title('Number of actions took during the episode')
        self.axs[1].set_xlabel('nb iteration')
        self.axs[1].set_ylabel('nb step')

        self.axs[2].plot(nb_mark, label=name)
        self.axs[2].set_title('number of mark action during the episode')
        self.axs[2].set_xlabel('nb iteration')
        self.axs[2].set_ylabel('nb marks')

        self.axs[3].plot(successful_marks, label=name)
        self.axs[3].set_title('if the last action was a successful mark')
        self.axs[3].set_xlabel('nb iteration')
        self.axs[3].set_ylabel('1: yes, 0: no')

        self.axs[4].plot(losses, label=name)
        self.axs[4].set_title('Losses')
        self.axs[4].set_xlabel('nb iteration')
        self.axs[4].set_ylabel('loss')

    def evaluate(self, agent, name):
        rewards, nb_mark, nb_action, successful_marks = agent.exploit()

        self.fig.suptitle("hyper parameters selections for {0}".format(name))

        self.axs[0].plot(rewards, label=name)
        self.axs[0].set_title('Sum of rewards')
        self.axs[0].set_xlabel('nb iteration')
        self.axs[0].set_ylabel('rewards')

        self.axs[1].plot(nb_action, label=name)
        self.axs[1].set_title('Number of actions took during the episode')
        self.axs[1].set_xlabel('nb iteration')
        self.axs[1].set_ylabel('nb step')

        self.axs[2].plot(nb_mark, label=name)
        self.axs[2].set_title('number of mark action during the episode')
        self.axs[2].set_xlabel('nb iteration')
        self.axs[2].set_ylabel('nb marks')

        self.axs[3].plot(successful_marks, label=name)
        self.axs[3].set_title('if the last action was a successful mark')
        self.axs[3].set_xlabel('nb iteration')
        self.axs[3].set_ylabel('1: yes, 0: no')

    def show(self):
        self.axs[0].legend(bbox_to_anchor=(1.3, 0.6))
        plt.show()


if __name__ == '__main__':

    ENVIRONMENT = environment.Environment("../../Dataset_waldo", nb_max_actions=5000, difficulty=0)
    ENVIRONMENT.init_env()

    EVALUATOR = Evaluator()
    REIN = Reinforce(ENVIRONMENT, episodes=10, guided_episodes=50)

    EVALUATOR.init_plot()
    EVALUATOR.fit(REIN, "Reinforce")
    EVALUATOR.show()

    ENVIRONMENT.evaluate()
    EVALUATOR.init_plot()
    EVALUATOR.evaluate(REIN, "Reinforce")
    EVALUATOR.show()

    plt.imshow(ENVIRONMENT.heat_map)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    policy = np.array(ENVIRONMENT.policy_hist)
    ax.scatter3D(policy[:, 0], policy[:, 1], policy[:, 2], c=policy[:, 3], cmap='cividis')
    plt.show()


    ENVIRONMENT.get_gif_trajectory("haha.gif")

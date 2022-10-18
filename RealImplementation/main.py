"""
The goal of this program is to allow user to evaluate 3 different RL algorithm on the dummy environment.
"""

from matplotlib import pyplot as plt

import environment
from reinforce import Reinforce

class Evaluator:

    def __init__(self):
        self.fig, self.axs = plt.subplots(nrows=5, ncols=1, layout="constrained")

    def evaluate(self, agent, name):
        '''
        This method allow user to evaluate a type of agent with a policy with the hyerparameter specified with __init__
        :param agent: the class of agent that must be trained
        :param policy: the class of policy that will be used by the agent
        :param name: the name of the agent (is used for the visualisation)
        '''
        print("starting evaluation of {0}".format(name))

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

        #print(len(np.count_nonzero(successful_marks[np.array(nb_action) < 1000])))

    def show(self):
        self.axs[0].legend(bbox_to_anchor=(1.3, 0.6))
        plt.show()


if __name__ == '__main__':

    de = environment.DummyEnv(nb_max_actions=100, replace_charlie=True)
    de.load_env("/home/remy/Documents/P9467.png", "/home/remy/Documents/mask1.png", "/home/remy/Documents/waldo.png")
    de.init_env()

    evaluator = Evaluator()

    rein = Reinforce(de, episodes=300, guided_episodes=150)
    evaluator.evaluate(rein, "Reinforce")
    evaluator.show()
    #plt.imshow(de.heat_map)
    #plt.show()
    de.get_gif_trajectory("haha.gif")

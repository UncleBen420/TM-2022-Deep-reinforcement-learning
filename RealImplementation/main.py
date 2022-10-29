"""
The goal of this program is to allow user to evaluate 3 different RL algorithm on the dummy environment.
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
import environment
from dummy_agent import DummyAgent
from reinforce import Reinforce

class Evaluator:

    def init_plot(self):
        self.fig, self.axs = plt.subplots(nrows=3, ncols=2, layout="constrained")
        self.box_plot_data_mark = []
        self.box_plot_data_gb = []
        self.names_mark = []
        self.names_gb = []

    def fit(self, agent, name):
        '''
        This method allow user to evaluate a type of agent with a policy with the hyerparameter specified with __init__
        :param agent: the class of agent that must be trained
        :param policy: the class of policy that will be used by the agent
        :param name: the name of the agent (is used for the visualisation)
        '''
        print("starting fitting of {0}".format(name))

        losses, rewards, nb_mark, nb_action, successful_marks, good, bad = agent.fit()

        self.fig.suptitle("hyper parameters selections for {0}".format(name))

        self.axs[0][0].plot(rewards, label=name)
        self.axs[0][0].set_title('Sum of rewards')
        self.axs[0][0].set_xlabel('nb iteration')
        self.axs[0][0].set_ylabel('rewards')

        self.axs[1][0].plot(nb_action, label=name)
        self.axs[1][0].set_title('Number of actions took during the episode')
        self.axs[1][0].set_xlabel('nb iteration')
        self.axs[1][0].set_ylabel('nb step')

        self.axs[2][0].plot(nb_mark, label=name)
        self.axs[2][0].set_title('number of mark action during the episode')
        self.axs[2][0].set_xlabel('nb iteration')
        self.axs[2][0].set_ylabel('nb marks')

        self.axs[0][1].plot(successful_marks, label=name)
        self.axs[0][1].set_title('number of successful mark')
        self.axs[0][1].set_xlabel('nb iteration')
        self.axs[0][1].set_ylabel('nb')

        base = np.linspace(0.,1, len(good))
        self.axs[1][1].plot(base, good, label=name, color='g')
        self.axs[1][1].plot(base, bad, label=name, color='r')
        self.axs[1][1].fill_between(base, bad, good, color='g', alpha=.5)
        self.axs[1][1].fill_between(base, bad, 0, color='r', alpha=.5)
        self.axs[1][1].set_title('good action choice over bad')
        self.axs[1][1].set_xlabel('episode')
        self.axs[1][1].set_ylabel('good/bad')

        self.axs[2][1].plot(losses, label=name)
        self.axs[2][1].set_title('Losses')
        self.axs[2][1].set_xlabel('nb iteration')
        self.axs[2][1].set_ylabel('loss')

    def evaluate(self, agent, name):
        rewards, nb_mark, nb_action, successful_marks, good, bad = agent.exploit()

        self.fig.suptitle("hyper parameters selections for {0}".format(name))

        self.axs[0][0].plot(rewards, label=name)
        self.axs[0][0].set_title('Sum of rewards')
        self.axs[0][0].set_xlabel('nb iteration')
        self.axs[0][0].set_ylabel('rewards')

        self.axs[1][0].plot(nb_action, label=name)
        self.axs[1][0].set_title('Number of actions took during the episode')
        self.axs[1][0].set_xlabel('nb iteration')
        self.axs[1][0].set_ylabel('nb step')

        self.axs[2][0].plot(nb_mark, label=name)
        self.axs[2][0].set_title('number of mark action during the episode')
        self.axs[2][0].set_xlabel('nb iteration')
        self.axs[2][0].set_ylabel('nb marks')

        self.axs[0][1].plot(successful_marks, label=name)
        self.axs[0][1].set_title('number of successful mark')
        self.axs[0][1].set_xlabel('nb iteration')
        self.axs[0][1].set_ylabel('nb')

        self.box_plot_data_gb.append(good)
        self.names_gb.append(name + "good")
        self.box_plot_data_gb.append(bad)
        self.names_gb.append(name + "bad")

        index = np.where(np.array(nb_mark) > 0)
        self.box_plot_data_mark.append(np.divide(np.array(successful_marks)[index], np.array(nb_mark)[index]))
        self.names_mark.append(name)

    def show_eval(self):
        self.axs[1][1].boxplot(self.box_plot_data_gb, labels=self.names_gb)
        self.axs[1][1].set_title('good/action repartition')
        self.axs[1][1].legend()

        self.axs[2][1].boxplot(self.box_plot_data_mark, labels=self.names_mark)
        self.axs[2][1].set_title('mark precision')
        self.axs[2][1].legend()
        self.axs[0][1].legend(bbox_to_anchor=(1.3, 0.6))
        plt.show()

    def show(self):
        self.axs[0][1].legend(bbox_to_anchor=(1.3, 0.6))
        plt.show()


if __name__ == '__main__':

    ENVIRONMENT = environment.Environment("../../Dataset_waldo", difficulty=1, depth=True)
    ENVIRONMENT.init_env()

    EVALUATOR = Evaluator()
    REIN = Reinforce(ENVIRONMENT, episodes=2000, val_episode=100)
    DUMMY = DummyAgent(ENVIRONMENT, val_episode=100)

    EVALUATOR.init_plot()
    EVALUATOR.fit(REIN, "Reinforce")
    EVALUATOR.show()

    EVALUATOR.init_plot()
    EVALUATOR.evaluate(DUMMY, "Dummy agent")
    ENVIRONMENT.evaluation_mode = True
    EVALUATOR.evaluate(REIN, "Reinforce")
    EVALUATOR.show_eval()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    policy = ENVIRONMENT.heat_map
    x, y = np.meshgrid(np.linspace(0, 1, policy.shape[1]), np.linspace(0, 1, policy.shape[2]))
    vmin = np.max(policy)
    vmax = np.max(policy)

    for i in range(policy.shape[0]):
        cset = ax.contourf(x, y, policy[i], 100, zdir='z', offset=i * 50, alpha=0.4)
    cset = ax.contourf(x, y, cv2.cvtColor(ENVIRONMENT.hist_img, cv2.COLOR_BGR2GRAY), 100, zdir='z', cmap='Greys_r',offset=0)
    plt.show()


    ENVIRONMENT.get_gif_trajectory("haha.gif")

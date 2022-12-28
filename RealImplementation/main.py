"""
The goal of this program is to allow user to evaluate 3 different RL algorithm on the dummy environment.
"""
import numpy as np
from matplotlib import pyplot as plt
import environment
from dummy_agent import DummyAgent
from policygradient import PolicyGradient
import cv2

class Evaluator:

    def init_plot(self):
        self.fig, self.axs = plt.subplots(nrows=2, ncols=2, layout="constrained")
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

        losses, rewards, nb_action, good, bad, effective_action = agent.fit()

        self.fig.suptitle("hyper parameters selections for {0}".format(name))

        self.axs[0][0].plot(rewards, label=name)
        self.axs[0][0].set_title('Sum of rewards')
        self.axs[0][0].set_xlabel('nb iteration')
        self.axs[0][0].set_ylabel('rewards')

        self.axs[1][0].plot(nb_action, label="total")
        self.axs[1][0].plot(effective_action, label="effective")
        self.axs[1][0].set_title('Number of actions took during the episode')
        self.axs[1][0].set_xlabel('nb iteration')
        self.axs[1][0].set_ylabel('nb step')

        base = np.linspace(0.,1, len(good))
        self.axs[1][1].plot(base, good, label=name, color='g')
        self.axs[1][1].plot(base, bad, label=name, color='r')
        self.axs[1][1].fill_between(base, bad, good, color='g', alpha=.5)
        self.axs[1][1].fill_between(base, bad, 0, color='r', alpha=.5)
        self.axs[1][1].set_title('good action choice over bad')
        self.axs[1][1].set_xlabel('episode')
        self.axs[1][1].set_ylabel('good/bad')

        self.axs[0][1].plot(losses, label=name)
        self.axs[0][1].set_title('Losses')
        self.axs[0][1].set_xlabel('nb iteration')
        self.axs[0][1].set_ylabel('loss')

    def evaluate(self, agent, name):
        rewards, nb_action, good, bad, conventional, time, effective_action = agent.exploit()
        if len(conventional) > 0:
            self.axs[1][0].plot(conventional, label="conventional policy")
            time_conventional = np.array(conventional) * np.mean(np.array(time) / np.array(nb_action))
            self.axs[0][1].plot(time_conventional, label="conventional policy")

        self.fig.suptitle("hyper parameters selections for {0}".format(name))

        self.axs[0][0].plot(rewards, label=name)
        self.axs[0][0].set_title('Sum of rewards')
        self.axs[0][0].set_xlabel('nb iteration')
        self.axs[0][0].set_ylabel('rewards')

        self.axs[1][0].plot(effective_action, label=name)
        self.axs[1][0].set_title('Number of actions took during the episode')
        self.axs[1][0].set_xlabel('nb iteration')
        self.axs[1][0].set_ylabel('nb step')

        self.axs[0][1].plot(time, label=name)
        self.axs[0][1].set_title('Time take for one episode')
        self.axs[0][1].set_xlabel('nb iteration')
        self.axs[0][1].set_ylabel('time')


        self.box_plot_data_gb.append(good)
        self.names_gb.append(name + " good")
        self.box_plot_data_gb.append(bad)
        self.names_gb.append(name + " bad")

    def show_eval(self):
        self.axs[1][1].boxplot(self.box_plot_data_gb, labels=self.names_gb)
        self.axs[1][1].set_title('good/action repartition')
        self.axs[1][1].legend()
        self.axs[0][0].legend()
        self.axs[1][0].legend()
        self.axs[0][1].legend()
        plt.show()

    def show(self):
        self.axs[0][1].legend(bbox_to_anchor=(1.3, 0.6))
        plt.show()


if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # Initialise the environment
    # ------------------------------------------------------------------------------------------------------------------

    ENVIRONMENT = environment.Environment("../../Dataset_waldo", difficulty=2)
    ENVIRONMENT.init_env()
    EVALUATOR = Evaluator()
    PG = PolicyGradient(ENVIRONMENT, episodes=200, val_episode=50)
    DUMMY = DummyAgent(ENVIRONMENT, val_episode=50)

    # ------------------------------------------------------------------------------------------------------------------
    # Train the agent
    # ------------------------------------------------------------------------------------------------------------------

    EVALUATOR.init_plot()
    EVALUATOR.fit(PG, "Reinforce")
    EVALUATOR.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate the agent
    # ------------------------------------------------------------------------------------------------------------------

    EVALUATOR.init_plot()
    EVALUATOR.evaluate(DUMMY, "Dummy agent")
    ENVIRONMENT.evaluation_mode = True
    EVALUATOR.evaluate(PG, "Reinforce")
    EVALUATOR.show_eval()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    policy = ENVIRONMENT.V_map

    x, y = np.meshgrid(np.linspace(0, 1, policy.shape[1]), np.linspace(0, 1, policy.shape[2]))

    for i in range(policy.shape[0]):
        cb = ax.contourf(x, y, policy[i], 100, zdir='z', offset=i * 50, cmap="plasma", alpha=0.4)
    plt.colorbar(cb)

    _ = ax.contourf(x,
                    y,
                    cv2.cvtColor(ENVIRONMENT.hist_img, cv2.COLOR_BGR2GRAY),
                    100,
                    zdir='z',
                    cmap='Greys_r',
                    offset=-1)
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Plot Agent sub image visited
    # ------------------------------------------------------------------------------------------------------------------

    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"

        mycmap = cmap
        mycmap._init()
        mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
        return mycmap

    mycmap = transparent_cmap(plt.cm.Reds)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    policy = ENVIRONMENT.heat_map

    x, y = np.meshgrid(np.linspace(0, 1, policy.shape[1]), np.linspace(0, 1, policy.shape[2]))

    for i in range(policy.shape[0]):
        _ = ax.contourf(x, y, policy[i], 100, zdir='z', offset=i * 50, cmap=mycmap)

    _ = ax.contourf(x,
                    y,
                    cv2.cvtColor(ENVIRONMENT.hist_img, cv2.COLOR_BGR2GRAY),
                    100,
                    zdir='z',
                    cmap='Greys_r',
                    offset=-1)
    plt.show()

    ENVIRONMENT.get_gif_trajectory("real_implementation_trajectory.gif")

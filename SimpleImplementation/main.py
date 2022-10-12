"""
The goal of this program is to allow user to evaluate 3 different RL algorithm on the dummy environment.
"""

from matplotlib import pyplot as plt
import EnvironmentDummySoft
from Agents import QLearning, E_Greedy, MonteCarloOnPolicy, NStepSarsa
from DeepAgent.reinforce import Reinforce

class Evaluator:

    def __init__(self):
        self.fig, self.axs = plt.subplots(nrows=4, ncols=1, layout="constrained")

    def evaluate(self, agent, name):
        '''
        This method allow user to evaluate a type of agent with a policy with the hyerparameter specified with __init__
        :param agent: the class of agent that must be trained
        :param policy: the class of policy that will be used by the agent
        :param name: the name of the agent (is used for the visualisation)
        '''
        print("starting evaluation of {0}".format(name))

        _, rewards, nb_mark, nb_action, successful_marks = agent.fit()

        self.fig.suptitle("hyper parameters selections for {0}".format(name))

        self.axs[0].plot(rewards, label=name)
        self.axs[0].set_title('Sum of rewards')
        self.axs[0].set_xlabel('nb iteration')
        self.axs[0].set_ylabel('rewards')

        self.axs[1].plot(nb_action, label=name)
        self.axs[1].set_title('Sum of rewards')
        self.axs[1].set_xlabel('nb iteration')
        self.axs[1].set_ylabel('rewards')

        self.axs[2].plot(nb_mark, label=name)
        self.axs[2].set_title('Percent of number of boat left')
        self.axs[2].set_xlabel('nb iteration')
        self.axs[2].set_ylabel('percent nb boats')

        self.axs[3].plot(successful_marks, label=name)
        self.axs[3].set_title('Percent of number of boat left')
        self.axs[3].set_xlabel('nb iteration')
        self.axs[3].set_ylabel('percent nb boats')

        #print(len(np.count_nonzero(successful_marks[np.array(nb_action) < 1000])))

    def show(self):
        self.axs[0].legend(bbox_to_anchor=(1.3, 0.6))
        plt.show()


if __name__ == '__main__':

    de = EnvironmentDummySoft.DummyEnv(nb_max_actions=1000, replace_charlie=False, deep=True)

    de.init_env()
    plt.imshow(de.render_board_img())
    plt.show()

    evaluator = Evaluator()

    rein = Reinforce(de, episodes=10, n_inputs=5)
    evaluator.evaluate(rein, "Reinforce")
    de.deep = False
    evaluator.evaluate(QLearning(de, E_Greedy(0.05), episodes=10), "Q-Learning")
    evaluator.evaluate(MonteCarloOnPolicy(de, E_Greedy(0.05), episodes=10), "Monte Carlo")
    evaluator.evaluate(NStepSarsa(de, E_Greedy(0.05), episodes=10), "N-Step Sarsa")
    evaluator.show()

    de.get_gif_trajectory("haha.gif")

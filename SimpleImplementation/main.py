"""
The goal of this environment is to find waldo in a simulated high resolution image.
4 algorithms are compared: Q-learning, N-step Sarsa, Monte Carlo and Reinforce.
"""
from matplotlib import pyplot as plt
import EnvironmentDummySoft
from Agents import QLearning, E_Greedy, MonteCarloOnPolicy, NStepSarsa
from DeepAgent.reinforce import Reinforce


class Evaluator:
    """
    This class is used to evaluate and plot results of experiment
    """

    def __init__(self):
        self.fig, self.axs = plt.subplots(nrows=4, ncols=1, layout="constrained")
        self.fig.suptitle("algorithms comparison on a simple dummy env")

    def evaluate(self, agent, name):
        '''
        This method allow user to evaluate a type of agent
        :param agent: the class of agent that must be trained
        :param name: the name of the agent (is used for the visualisation)
        '''
        print("starting evaluation of {0}".format(name))

        _, rewards, nb_mark, nb_action, successful_marks = agent.fit()

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
        """
        plot the results.
        """
        self.axs[0].legend(bbox_to_anchor=(1.3, 0.6))
        plt.show()


if __name__ == '__main__':

    DUMMY_ENV = EnvironmentDummySoft.DummyEnv(nb_max_actions=1000, replace_charlie=False, deep=True)
    DUMMY_ENV.init_env()

    plt.imshow(DUMMY_ENV.render_board_img())
    plt.show()

    EVALUATOR = Evaluator()
    EVALUATOR.evaluate(Reinforce(DUMMY_ENV, episodes=1000, n_inputs=5), "Reinforce")
    DUMMY_ENV.get_gif_trajectory("simple_implementation.gif")
    DUMMY_ENV.deep = False
    EVALUATOR.evaluate(QLearning(DUMMY_ENV, E_Greedy(0.05), episodes=1000), "Q-Learning")
    EVALUATOR.evaluate(MonteCarloOnPolicy(DUMMY_ENV, E_Greedy(0.05), episodes=1000), "Monte Carlo")
    EVALUATOR.evaluate(NStepSarsa(DUMMY_ENV, E_Greedy(0.05), episodes=1000), "N-Step Sarsa")
    EVALUATOR.show()

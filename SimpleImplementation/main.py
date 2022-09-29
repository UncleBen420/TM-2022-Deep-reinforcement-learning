from time import sleep

import cv2
import re

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from DummyExperiment import EnvironmentDummySoft
from DummyExperiment import EnvironmentDummyHard

from DummyExperiment.Agents import DummyAgent, QLearning, E_Greedy, MonteCarloOnPolicy, NStepSarsa, ExploitAgent, UCB


class Evaluator:

    def __init__(self, environment, nb_folds, episodes, alphas, gammas, epsilons):
        self.environment = environment
        self.episodes = episodes
        self.nb_folds = nb_folds
        self.epsilons = epsilons
        self.hyper_parameters = np.array(np.meshgrid(gammas, alphas, epsilons)).T.reshape(-1, 3)

    def evaluate(self, agent, policy, name):

        print("starting evaluation of {0}".format(name))

        results = []

        fig, axs = plt.subplots(nrows=3, ncols=1, layout="constrained")
        fig.suptitle("hyper parameters selections for {0}".format(name))

        for i, hyper_parameter in enumerate(self.hyper_parameters):
            v_means = []
            sum_of_rewards = []
            boats_left = []

            gamma, alpha, epsilon = hyper_parameter
            label = "g: " + str(gamma) + ",a: " + str(alpha) + ",e: " + str(epsilon)

            for fold in range(self.nb_folds):
                print("{0}/{1}: training with gamma at {2}, alpha at {3} and epsilon at {4}"
                      .format(fold+1, self.nb_folds, gamma, alpha, epsilon))
                self.environment.init_env()
                agt = agent(de, policy(epsilon), episodes=self.episodes)
                metrics = agt.fit()

                v_means.append(metrics[0])
                sum_of_rewards.append(metrics[1])
                boats_left.append(metrics[2])

            axs[0].plot(np.mean(v_means, axis=0), label=label)
            axs[0].set_title('Mean V of the agent')
            axs[0].set_xlabel('nb iteration')
            axs[0].set_ylabel('mean of V')
            axs[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left")

            axs[1].plot(np.mean(sum_of_rewards, axis=0), label=label)
            axs[1].set_title('Sum of rewards')
            axs[1].set_xlabel('nb iteration')
            axs[1].set_ylabel('rewards')

            axs[2].plot(np.mean(boats_left, axis=0), label=label)
            axs[2].set_title('Percent of number of boat left')
            axs[2].set_xlabel('nb iteration')
            axs[2].set_ylabel('percent nb boats')

            fig.legend(bbox_to_anchor=(1.3, 0.6))

        plt.show()


if __name__ == '__main__':

    de = EnvironmentDummySoft.DummyEnv(nb_max_actions=1000)

    de.init_env()
    print(de.render_grid(de.grid))

    dm = DummyAgent(de)
    evaluator = Evaluator(de, 3, 1000, [0.1, 0.2], [0.5, 0.6], [0.01, 0.05])
    evaluator.evaluate(QLearning, E_Greedy, "Q-Learning")
    evaluator.evaluate(MonteCarloOnPolicy, E_Greedy, "Monte Carlo")
    evaluator.evaluate(NStepSarsa, E_Greedy, "N-Step Sarsa")

import gc
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from components.agent import PolicyGradient
from components.environment import Environment


def describe(arr):
    print("Measures of Central Tendency")
    print("Mean =", np.mean(arr))
    print("Median =", np.median(arr))
    print("Measures of Dispersion")
    print("Minimum =", np.min(arr))
    print("Maximum =", np.max(arr))
    print("Variance =", np.var(arr))
    print("Standard Deviation =", np.std(arr))


class Trainer:
    def __init__(self):
        self.label_path = None
        self.img_path = None
        self.label_list = None
        self.img_list = None
        self.env = Environment()
        self.agent = PolicyGradient(self.env)

    def train(self, nb_episodes, train_path):

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING PREPARATION
        # --------------------------------------------------------------------------------------------------------------

        self.agent.model_summary()

        self.img_path = os.path.join(train_path, "img")
        self.label_path = os.path.join(train_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        losses = []
        rewards = []
        vs = []
        td_errors = []
        nb_action = []
        nb_conv_action = []
        nb_min_zoom_action = []

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING STEPS
        # --------------------------------------------------------------------------------------------------------------
        with tqdm(range(nb_episodes), unit="episode") as episode:
            for i in episode:
                # random image selection in the training set
                while True:
                    index = random.randint(0, len(self.img_list) - 1)
                    img = os.path.join(self.img_path, self.img_list[index])
                    bb = os.path.join(self.label_path, self.img_list[index].split('.')[0] + '.txt')
                    if os.path.exists(bb):
                        break

                first_state = self.env.reload_env(img, bb)
                loss, sum_reward, sum_v, mean_tde = self.agent.fit_one_episode(first_state)

                rewards.append(sum_reward)
                losses.append(loss)
                st = self.env.nb_actions_taken
                vs.append(sum_v / st)
                td_errors.append(mean_tde)
                nb_action.append(st)

                episode.set_postfix(rewards=sum_reward, loss=loss, nb_action=st, V=sum_v / st, tde=mean_tde)

                if i > 450:
                    plt.imshow(self.env.get_heat_map())
                    plt.show()

        # --------------------------------------------------------------------------------------------------------------
        # PLOT AND WEIGHTS SAVING
        # --------------------------------------------------------------------------------------------------------------

        plt.plot(rewards)
        plt.show()
        plt.plot(losses)
        plt.show()
        plt.plot(td_errors)
        plt.show()

    def evaluate(self, eval_path, result_path='.', plot_metric=False):

        self.img_path = os.path.join(eval_path, "img")
        self.label_path = os.path.join(eval_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        rewards = []
        vs = []
        nb_action = []
        nb_conv_action = []
        precision = []
        pertinence = []

        # --------------------------------------------------------------------------------------------------------------
        # EVALUATION STEPS
        # --------------------------------------------------------------------------------------------------------------
        with tqdm(range(len(self.img_list)), unit="episode") as episode:
        #with tqdm(range(5), unit="episode") as episode:
            for i in episode:
                collected = gc.collect()
                img_filename = self.img_list[i]
                img = os.path.join(self.img_path, img_filename)
                bb = os.path.join(self.label_path, img_filename.split('.')[0] + '.txt')
                if not os.path.exists(bb):
                    continue

                first_state = self.env.reload_env(img, bb)
                sum_reward, sum_v = self.agent.exploit_one_episode(first_state)
                st = self.env.nb_actions_taken
                rewards.append(sum_reward)
                vs.append(sum_v / st)
                nb_action.append(st)
                nb_conv_action.append(self.env.conventional_policy_nb_step)

                pr = self.env.min_zoom_action / self.env.nb_max_conv_action
                prc = self.env.conventional_policy_nb_step / self.env.nb_max_conv_action
                precision.append(1 - pr)
                pertinence.append(prc - pr)

                episode.set_postfix(rewards=sum_reward, nb_action=st, V=sum_v / st)



        # --------------------------------------------------------------------------------------------------------------
        # PLOT
        # --------------------------------------------------------------------------------------------------------------
        print("")
        print("Overall precision on evaluation")
        print("-------------------------------")
        describe(precision)
        print("-------------------------------")

        print("")
        print("Overall pertinence on evaluation")
        print("-------------------------------")
        describe(pertinence)
        print("-------------------------------")


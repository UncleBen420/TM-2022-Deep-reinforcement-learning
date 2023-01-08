import gc
import os
import random

import imageio
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from components.dot import DOT
from components.environment import Environment
from components.tod import TOD


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
        self.agent = DOT(self.env)
        self.agent_tod = TOD(self.env)

    def train_tod(self):
        rewards = []
        losses = []
        nb_action = []
        self.env.reduce()

        with tqdm(range(len(self.env.bboxes)), unit="episode") as episode:
            for i in episode:

                first_state = self.env.reload_env_tod(i)
                loss, sum_reward, = self.agent_tod.fit_one_episode(first_state)

                st = self.env.nb_actions_taken_tod
                rewards.append(sum_reward / st)
                losses.append(loss)
                nb_action.append(st)

                episode.set_postfix(rewards=sum_reward, loss=loss, nb_action=st)
        self.agent_tod.train_classification()
        self.agent_tod.trim_ds()

    def eval_tod(self):
        rewards = []
        losses = []
        nb_action = []
        self.env.reduce()
        frames = []
        with tqdm(range(len(self.env.bboxes)), unit="episode") as episode:
            for i in episode:
                first_state = self.env.reload_env_tod(i)
                sum_reward = self.agent_tod.exploit_one_episode(first_state)

                st = self.env.nb_actions_taken_tod
                rewards.append(sum_reward / st)
                nb_action.append(st)

                episode.set_postfix(rewards=sum_reward,  nb_action=st)
                frames.extend(self.env.steps_recorded)

            plt.imshow(self.env.TOD_history())
            plt.show()

            return frames

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
        class_losses = []

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

                train_tod = False
                if i > 10 and i % 5 == 0:
                    train_tod = True

                self.env.train_tod = train_tod
                first_state = self.env.reload_env(img, bb)
                loss, sum_reward, sum_v, mean_tde = self.agent.fit_one_episode(first_state)

                st = self.env.nb_actions_taken
                rewards.append(sum_reward / st)
                losses.append(loss)
                vs.append(sum_v / st)
                td_errors.append(mean_tde)
                nb_action.append(st)

                episode.set_postfix(rewards=sum_reward / st, loss=loss, nb_action=st, V=sum_v, tde=mean_tde)

                if i > 10 and i % 5 == 0:
                    self.env.add_bbox_for_class()
                    self.train_tod()
                    #plt.imshow(self.env.TOD_history())
                    #plt.show()

        # --------------------------------------------------------------------------------------------------------------
        # PLOT AND WEIGHTS SAVING
        # --------------------------------------------------------------------------------------------------------------

        plt.plot(rewards)
        plt.show()
        plt.plot(losses)
        plt.show()
        plt.plot(class_losses)
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
        self.env.record = True
        with tqdm(range(len(self.img_list)), unit="episode") as episode:
            for i in episode:
                img_filename = self.img_list[i]
                img = os.path.join(self.img_path, img_filename)
                bb = os.path.join(self.label_path, img_filename.split('.')[0] + '.txt')
                if not os.path.exists(bb):
                    continue

                self.env.train_tod = True
                first_state = self.env.reload_env(img, bb)
                sum_reward, sum_v = self.agent.exploit_one_episode(first_state)
                st = self.env.nb_actions_taken
                rewards.append(sum_reward)
                nb_action.append(st)
                plt.imshow(self.env.DOT_history())
                plt.show()

                episode.set_postfix(rewards=sum_reward, nb_action=st, V=sum_v / st)
                frames = self.env.steps_recorded
                frames.extend(self.eval_tod())
                frames.extend([self.env.TOD_history()])


                imageio.mimsave(img_filename + ".gif", frames, duration=0.01)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT
        # --------------------------------------------------------------------------------------------------------------

        plt.plot(rewards)
        plt.show()
import gc
import os
import random

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from components.dot import DOT
from components.environment import Environment
from components.tod import TOD

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def describe(arr):
    print("Measures of Central Tendency")
    print("Mean =", np.mean(arr))
    print("Median =", np.median(arr))
    print("Measures of Dispersion")
    print("Minimum =", np.min(arr))
    print("Maximum =", np.max(arr))
    print("Variance =", np.var(arr))
    print("Standard Deviation =", np.std(arr))

def create_video(frames, filename):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(filename, fourcc, float(30), (264, 264))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()


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

                rewards.append(sum_reward)
                losses.append(loss)

                episode.set_postfix(rewards=sum_reward, loss=loss)
        loss_class = self.agent_tod.train_classification()
        self.agent_tod.trim_ds()

        return np.mean(losses), np.mean(rewards), loss_class

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

                rewards.append(sum_reward)

                episode.set_postfix(rewards=sum_reward)
                frames.extend(self.env.steps_recorded)

            plt.imshow(self.env.TOD_history())
            plt.show()

            return frames

    def train(self, nb_episodes, train_path):

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING PREPARATION
        # --------------------------------------------------------------------------------------------------------------

        self.agent.model_summary()
        self.agent_tod.model_summary()

        self.img_path = os.path.join(train_path, "img")
        self.label_path = os.path.join(train_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        losses = []
        rewards = []
        vs = []
        losses_tod = []
        rewards_tod = []
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
                loss, sum_reward, _, _ = self.agent.fit_one_episode(first_state)

                rewards.append(sum_reward)
                losses.append(loss)
                episode.set_postfix(rewards=sum_reward, loss=loss)

                if i > 10 and i % 5 == 0:
                    self.env.add_bbox_for_class()
                    loss_tod, reward_tod, loss_class = self.train_tod()
                    losses_tod.append(loss_tod)
                    rewards_tod.append(reward_tod)
                    class_losses.append(loss_class)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT AND WEIGHTS SAVING
        # --------------------------------------------------------------------------------------------------------------

        plt.plot(rewards)
        plt.show()
        plt.plot(losses)
        plt.show()
        plt.plot(class_losses)
        plt.show()
        plt.plot(rewards_tod)
        plt.show()
        plt.plot(losses_tod)
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
        iou_error = 0.
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

                iou_error += self.env.get_iou_error()

                #create_video(frames, img_filename + ".avi")
                #imageio.mimsave(img_filename + ".gif", frames, duration=0.01)
        iou_error /= len(self.img_list)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT
        # --------------------------------------------------------------------------------------------------------------

        plt.scatter(self.env.iou_base, self.env.iou_final)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

        cm = confusion_matrix(self.env.truth_values, self.env.predictions)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.show()

        print(iou_error)
import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from components.dot import DOT
from components.environment import Environment
from components.tod import TOD
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from datetime import date

MODEL_RES = 200

def create_video(frames, filename):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(filename, fourcc, float(10), (MODEL_RES, MODEL_RES))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()


class Trainer:
    def __init__(self, learning_rate=0.0005, gamma=0.1, lr_gamma=0.8):
        self.label_path = None
        self.img_path = None
        self.label_list = None
        self.img_list = None
        self.env = Environment()
        self.agent = DOT(self.env, learning_rate, gamma, lr_gamma)
        self.agent_tod = TOD(self.env, learning_rate, gamma, lr_gamma)

    def train(self, nb_episodes, train_path, plot_metric=False):

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

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING STEPS
        # --------------------------------------------------------------------------------------------------------------
        with tqdm(range(nb_episodes), unit="episode") as episode:
            for i in episode:

                if i > 10 and i % 10 == 0:
                    self.env.train_tod = True

                # random image selection in the training set
                while True:
                    index = random.randint(0, len(self.img_list) - 1)
                    img = os.path.join(self.img_path, self.img_list[index])
                    bb = os.path.join(self.label_path, self.img_list[index].split('.')[0] + '.txt')
                    if os.path.exists(bb):
                        break

                if i > 900 and i % 10 == 0:
                    self.env.train_tod = True
                    plt.imshow(self.env.TOD_history())
                    plt.show()

                first_state = self.env.reload_env(img, bb)
                loss, sum_reward, _, _ = self.agent.fit_one_episode(first_state)

                rewards.append(sum_reward)
                losses.append(loss)
                episode.set_postfix(rewards=sum_reward, loss=loss)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT AND WEIGHTS SAVING
        # --------------------------------------------------------------------------------------------------------------
        today = date.today()
        self.agent.save(str(today) +"-DOT-weights.pt")
        self.agent_tod.save(str(today) + "-TOD-weights.pt")

        if plot_metric:
            plt.plot(rewards)
            plt.show()
            plt.plot(losses)
            plt.show()
            plt.plot(self.env.tod_rewards)
            plt.show()
            plt.plot(self.env.tod_policy_loss)
            plt.show()
            plt.plot(self.env.tod_class_loss)
            plt.show()
            plt.plot(self.env.tod_conf_loss)
            plt.show()

    def evaluate(self, eval_path, plot_metric=False):
        self.env.eval_tod = True
        self.img_path = os.path.join(eval_path, "img")
        self.label_path = os.path.join(eval_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        rewards = []

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

                first_state = self.env.reload_env(img, bb)
                sum_reward, sum_v = self.agent.exploit_one_episode(first_state)
                st = self.env.nb_actions_taken
                rewards.append(sum_reward)

                iou_error += self.env.get_iou_error()

                episode.set_postfix(rewards=sum_reward, nb_action=st, V=sum_v / st)

                if plot_metric:
                    frames = self.env.steps_recorded
                    frames.extend(self.env.TOD_history())
                    create_video(frames, img_filename + ".avi")

        iou_error /= len(self.img_list)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT
        # --------------------------------------------------------------------------------------------------------------
        if plot_metric:
            plt.scatter(self.env.iou_base, self.env.iou_final)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
            final_iou = np.array(self.env.iou_final)
            base_iou = np.array(self.env.iou_base)
            index = np.argsort(base_iou)
            final_iou = final_iou[index]
            base_iou = base_iou[index]
            down = np.where(base_iou >= final_iou)[0]
            up = np.where(base_iou < final_iou)[0]
            plt.bar(up * 3, final_iou[up] - base_iou[up], 3, bottom=base_iou[up], color="green")
            plt.bar(down * 3, base_iou[down] - final_iou[down], 3, final_iou[down], color="red")
            plt.show()

            cm = confusion_matrix(self.env.truth_values, self.env.predictions)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            plt.show()
            class_accuracy = cm.diagonal() / cm.sum(axis=1)
            print("Accuracy by class: {0}".format(class_accuracy))
            print("IOU error: {0}".format(iou_error))
            print("Total accuracy {0}".format(accuracy_score(self.env.truth_values, self.env.predictions)))
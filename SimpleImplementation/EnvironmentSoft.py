"""
This file implement a dummy environment to train the agents on and compare them. The term "Soft" mean that the
states of the environment are not linked to it's size (contrary to a grid world for exemple).
"""

import math
import random
import cv2
import re
import numpy as np


def check_cuda():
    cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n')
               if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
    return len(cv_info) > 0


class SoftEnv:
    """
    this class implement the grid world problem as a frozen lake problem.
    """
    def __init__(self, model_resolution, item_model, surface_model):
        self.bb_map = None
        self.marked_image = None
        self.max_zoom = None
        self.nb_actions_taken = None
        self.channels = None
        self.W = None
        self.H = None
        self.marked_map = None
        self.marked = None
        self.history = None
        self.full_img = None
        self.sub_img = None
        self.bboxes = None
        self.x = 0
        self.y = 0
        self.z = 1

        self.min_resolution = 32
        self.model_resolution = model_resolution

        self.cv = cv2.cuda if check_cuda() else cv2

    def reload_env(self):
        del self.history
        del self.marked
        del self.marked_map

        self.history = []
        self.marked = []
        self.marked_map = np.zeros((self.W, self.H), dtype=bool)
        self.nb_actions_taken = 0
        self.z = random.randint(1, self.max_zoom - 1)
        self.x = random.randint(0, self.W / (self.model_resolution ** self.z) - 1)
        self.y = random.randint(0, self.H / (self.model_resolution ** self.z) - 1)
        self.compute_sub_img()
        self.get_prediction()
        self.get_vision()

        return self.get_current_state()

    def init_env(self, image, labels):
        self.full_img = cv2.imread(image)
        self.H, self.W, self.channels = self.full_img.shape
        self.max_zoom = int(math.log(np.min([self.W, self.H]), self.min_resolution))
        self.bb_map = np.zeros((self.H, self.W), dtype=np.uint8)
        print(self.max_zoom)
        with open(labels, "r") as file:
            lines = file.read()
        for i, line in enumerate(lines.split('\n')):
            if len(line.split(' ')) > 1:
                _, x_min, x_max, y_min, y_max = line.split(' ')
                x_min = int(float(x_min))
                x_max = int(float(x_max))
                y_min = int(float(y_min))
                y_max = int(float(y_max))

                self.bb_map[y_min:y_max, x_min:x_max] = i
        self.marked_image = np.zeros((self.H, self.W, self.channels), dtype=np.uint8)
        self.marked_image[self.bb_map > 0] = [255, 0, 0]
        self.marked_image = self.cv.addWeighted(self.marked_image, 0.3, self.full_img, 0.7, 0)

    def compute_sub_img(self):
        self.z = 2
        window = self.min_resolution ** self.z
        self.sub_img = self.full_img[window * self.y:window + window * self.y, window * self.x:window + window * self.x]
        self.sub_img = self.cv.resize(self.sub_img, (self.model_resolution, self.model_resolution))

    def get_prediction(self):
        pass

    def get_reward(self):
        pass

    def go_to_next_state(self):
        pass

    def zoom(self):
        self.zoom_step += 1

    def move(self):
        pass


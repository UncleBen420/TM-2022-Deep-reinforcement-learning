"""
This file contain the implementation of the real environment.
"""
import math
import os
import random
import re
from enum import Enum
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


def check_cuda():
    """
    check if opencv can use cuda
    :return: return True if opencv can detect cuda. False otherwise.
    """
    cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n')
               if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
    return len(cv_info) > 0

MODEL_RES = 32
HIST_RES = 100


class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, dataset_path, nb_max_actions=100, difficulty=0, depth=False):
        self.sub_images_queue = None
        self.base_img = None
        self.gpu_full_img = None
        self.charlie_y = 0
        self.charlie_x = 0
        self.history = None
        self.policy_hist = {}
        self.heat_map = np.zeros((6, HIST_RES, HIST_RES))
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        self.search_style = -1 if depth else 0
        self.nb_action = 16

        self.zoom_padding = 2
        self.z = 1
        self.x = 0
        self.y = 0
        # (x, y)
        self.sub_grid = None
        # State of the environment
        self.dummy_boat_model = None
        self.dummy_surface_model = None
        self.vision = np.zeros(7, dtype=int)
        self.guided = True
        self.cv_cuda = check_cuda()
        self.difficulty = difficulty
        self.charlie = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, "waldo.png")), cv2.COLOR_BGR2RGB)
        self.image_list = sorted(os.listdir(os.path.join(dataset_path, "images")))
        self.mask_list = sorted(os.listdir(os.path.join(dataset_path, "masks")))
        self.dataset_path = dataset_path
        self.evaluation_mode = False

    def place_charlie(self):
        """
        this method place change the charlie's position on the map.
        """
        while True:
            x = random.randint(0, self.W - 1)
            y = random.randint(0, self.H - 1)
            if self.mask[y][x][0] == 0:
                self.charlie_x = x
                self.charlie_y = y
                self.full_img = self.base_img.copy()
                self.full_img[self.charlie_y:self.charlie_y + self.charlie.shape[0],
                self.charlie_x:self.charlie_x + self.charlie.shape[1]] = self.charlie
                break

    def reload_env(self):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        del self.history
        del self.sub_images_queue
        self.sub_images_queue = []
        self.history = []
        self.nb_actions_taken = 0
        self.nb_bad_choice = 0
        self.nb_good_choice = 0
        self.z = self.max_zoom
        self.x = 0
        self.y = 0
        self.nb_mark = 0

        self.marked_correctly = False
        if self.cv_cuda:
            self.get_sub_images(self.gpu_full_img)
        else:
            self.get_sub_images(self.full_img)
        S = self.get_current_state_deep()

        return S

    def load_env(self):
        """
        This method read a image file and the mask (where waldo can be placed).
        :param img: image representing the environment.
        :param mask: image corresponding to the mask where waldo can be placed.
        """
        index = random.randint(0, len(self.image_list) - 1)
        img = os.path.join(self.dataset_path, "images", self.image_list[index])
        mask = os.path.join(self.dataset_path, "masks", self.mask_list[index])

        self.base_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        self.H, self.W, self.channels = self.base_img.shape
        self.ratio = HIST_RES / self.H
        self.mask = cv2.imread(mask)
        self.max_distance = math.sqrt(self.W ** 2 + self.H ** 2)
        min_dim = np.min([self.W, self.H])
        self.max_zoom = int(math.log(min_dim, 2))
        self.min_zoom = self.max_zoom - 5

    def init_env(self):
        """
        This method is used to load the image representing the environment to the gpu
        It place charlie on the image has well
        """
        if self.base_img is None or self.difficulty == 2:
            self.load_env()

        self.place_charlie()

        if self.cv_cuda:
            self.gpu_full_img = cv2.cuda_GpuMat()
            self.gpu_full_img.upload(self.full_img)
        self.hist_img = cv2.resize(self.full_img, (HIST_RES, HIST_RES))
        self.compute_mask_map()

    def compute_mask_map(self):
        """
        Compute the sub grid at the agent position given the x, y and z axis.
        """
        h = int(self.H / (self.zoom_padding << (self.min_zoom - 2)))
        w = int(self.W / (self.zoom_padding << (self.min_zoom - 2)))

        mask_map = cv2.cvtColor(cv2.resize(self.mask, (h, w)), cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3, 3), 'uint8')
        mask_map = cv2.erode(mask_map, kernel, iterations=1)

        _, mask_map = cv2.threshold(mask_map, 10, 255, cv2.THRESH_BINARY)
        self.ROI = np.array(np.where(mask_map == False))
        self.ROI[0, :] *= w
        self.ROI[1, :] *= h

    def record(self):
        window = self.zoom_padding << (self.z - 1)
        self.heat_map[self.z - self.min_zoom][int(window * self.y * self.ratio):
                      int((window + window * self.y) * self.ratio),
        int(window * self.x * self.ratio):
        int((window + window * self.x) * self.ratio)] += 1

    def get_sub_images(self, img):
        if self.cv_cuda:
            self.sub_vision = cv2.cuda.resize(img, (MODEL_RES, MODEL_RES)).download()
        else:
            self.sub_vision = cv2.resize(img, (MODEL_RES, MODEL_RES))

        sub_z = self.z - 1
        sub_x = self.x << 1
        sub_y = self.y << 1

        h = int(img.shape[0] / 2)
        w = int(img.shape[1] / 2)
        self.sub_images = []
        if self.cv_cuda:
            for i in range(2):
                for j in range(2):
                    x_ = sub_x + i
                    y_ = sub_y + j

                    minY = h * j
                    minX = w * i
                    maxY = h + h * j
                    maxX = w + w * i

                    gpu = cv2.cuda_GpuMat(img, (minY, minX, maxY, maxX))
                    self.sub_images.append(((x_, y_, sub_z), gpu))
        else:
            for i in range(2):
                for j in range(2):
                    x_ = sub_x + i
                    y_ = sub_y + j
                    self.sub_images.append(((x_, y_, sub_z),
                                            (img[h * j:h + h * j, w * i: w + w * i])))

    def sub_img_contain_charlie(self, x, y, z):
        """
        This method allow the user to know if the current subgrid contain charlie or not
        :return: true if the sub grid contains charlie.
        """
        window = self.zoom_padding << (z - 1)
        return ((x * window <= self.charlie_x < x * window + window or
                x * window <= self.charlie_x + self.charlie.shape[1] < x * window + window)
                and
                (y * window <= self.charlie_y < y * window + window or
                y * window <= self.charlie_y + self.charlie.shape[0] < y * window + window))

    def get_current_state_deep(self):
        """
        give to the agent 2 images (the sub image and the hist image). they are squeeze into
        a single array.
        :return: the current state.
        """

        return np.array(self.sub_vision.squeeze() / 255)

    def sub_image_contain_roi(self, x, y, z):
        window = self.zoom_padding << (z - 1)
        for i in range(self.ROI.shape[1]):
            if (x * window <= self.ROI[1][i] < x * window + window and
                    y * window <= self.ROI[0][i] < y * window + window):
                return True

        return False

    def action_selection(self, action, counter=0):
        if action >= 1:
            reward, is_terminal = self.action_selection(action >> 1, counter + 1)
        else:
            reward = 0.
            is_terminal = False

        if action % 2:
            pos, _ = self.sub_images[counter]
            x, y, z = pos

            if self.sub_image_contain_roi(x, y, z):
                self.nb_good_choice += 1
            else:
                self.nb_bad_choice += 1

            if z >= self.min_zoom:
                self.sub_images_queue.append(self.sub_images[counter])
            elif self.sub_img_contain_charlie(x, y, z):
                reward = 101#(4 << (self.min_zoom + 1)) - self.nb_actions_taken
                is_terminal = True
                if self.difficulty > 0:
                    self.place_charlie()

        return reward, is_terminal

    def take_action(self, action):
        """
        This method allow the agent to take an action over the environment.
        :param action: the number of the action that the agent has take.
        :return: the next state, the reward, if the state is terminal and a tips of which action the agent should have
        choose.
        """

        self.history.append((self.x, self.y, self.z, action))
        if self.evaluation_mode:
            self.record()

        # check if we are at the maximum zoom possible
        reward, is_terminal = self.action_selection(action)

        if len(self.sub_images_queue) <= 0:
            # for the case if charlie is detected at the last step
            reward = -100 if not is_terminal else reward
            is_terminal = True
        else:
            position, img = self.sub_images_queue.pop(self.search_style)
            self.x, self.y, self.z = position
            self.get_sub_images(img)
            self.nb_actions_taken += 1

        return self.get_current_state_deep(), reward, is_terminal

    def get_gif_trajectory(self, name):
        """
        This function allow the user to create a gif of all the moves the
        agent has made along the episodes
        :param name: the name of the gif file
        """
        frames = []
        for i in range(self.nb_actions_taken):
            x, y, z, a = self.history[i]
            mm = self.hist_img.copy()

            color = [0, 255, 0]

            window = (self.zoom_padding ** z)
            mm[int(window * y * self.ratio):
               int((window + window * y) * self.ratio),
            int(window * x * self.ratio):
            int((window + window * x) * self.ratio)] = color

            frames.append(mm)

        imageio.mimsave(name, frames, duration=0.5)

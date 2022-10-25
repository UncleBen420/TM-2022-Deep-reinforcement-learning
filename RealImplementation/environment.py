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


class Event(Enum):
    """
    this enum class simplify the different state of the grid
    """
    UNKNOWN = 0
    VISITED = 1
    BLOCKED = 2


class Action(Enum):
    """
    this enum class represent all the action that the agent is allowed to do.
    """
    ZOOM1 = 0
    ZOOM2 = 1
    ZOOM3 = 2
    ZOOM4 = 3
    DEZOOM = 4
    LEFT = 5
    UP = 6
    RIGHT = 7
    DOWN = 8

MODEL_RES = 32
HIST_RES = 32

class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, dataset_path, nb_max_actions=100, difficulty=0, only_zoom=False):
        self.base_img = None
        self.gpu_full_img = None
        self.charlie_y = 0
        self.charlie_x = 0
        self.nb_actions_taken = 0
        self.history = None
        self.policy_hist = {}
        self.heat_map = np.zeros((HIST_RES, HIST_RES))
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        if only_zoom:
            self.nb_action = 5
        else:
            self.nb_action = 9

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

        self.history = np.zeros((self.nb_max_actions, 4), dtype=int)
        self.nb_actions_taken = 0
        self.z = self.max_zoom
        self.x = 0
        self.y = 0
        self.nb_mark = 0

        self.marked_correctly = False

        self.compute_sub_grid()
        self.compute_hist()
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
        self.min_zoom = self.max_zoom - 4

    def init_env(self):
        """
        This method is used to load the image representing the environment to the gpu
        It place charlie on the image has well
        """
        if self.base_img is None or self.difficulty == 2:
            self.load_env()
            print("new env")
        print("new charlie pos")

        self.place_charlie()

        if self.cv_cuda:
            self.gpu_full_img = cv2.cuda_GpuMat()
            self.gpu_full_img.upload(self.full_img)
        self.hist_img = cv2.resize(self.full_img, (HIST_RES, HIST_RES), interpolation=cv2.INTER_NEAREST)

    def compute_sub_grid(self):
        """
        Compute the sub grid at the agent position given the x, y and z axis.
        """
        window = self.zoom_padding << (self.z - 1)
        if self.cv_cuda:
            minX = window * self.x
            maxY = window + window * self.y
            maxX = window + window * self.x
            minY = window * self.y

            self.gpu_sub_vision = cv2.cuda_GpuMat(self.gpu_full_img, (minY, minX, maxY, maxX))
            self.sub_vision = cv2.cuda.resize(self.sub_vision, (MODEL_RES, MODEL_RES)).download()
        else:
            self.sub_vision = self.full_img[window * self.y:window + window * self.y,
                              window * self.x:window + window * self.x]
            self.sub_vision = cv2.resize(self.sub_vision, (MODEL_RES, MODEL_RES))

    def compute_hist(self):
        """
        compute an image indicating the agent position on the full image
        """
        window = self.zoom_padding << (self.z - 1)
        self.hist = self.hist_img.copy()
        self.hist[int(window * self.y * self.ratio):
                  int((window + window * self.y) * self.ratio),
                  int(window * self.x * self.ratio):
                  int((window + window * self.x) * self.ratio)] = [255., 0., 0.]

    def record(self, h, a):
        if h not in self.policy_hist.keys():
            self.policy_hist[h] = []
        self.policy_hist[h].append(a)

        window = self.zoom_padding << (self.z - 1)
        self.heat_map[int(window * self.y * self.ratio):
                  int((window + window * self.y) * self.ratio),
                  int(window * self.x * self.ratio):
                  int((window + window * self.x) * self.ratio)] += 1

    def get_distance_reward(self):
        """
        this method return the distance between the agent position and the charlie's position.
        :return: the euclidian distance.
        """
        pad = self.zoom_padding << (self.z - 1)
        return math.sqrt((self.x * pad - self.charlie_x) ** 2 + (self.y * pad - self.charlie_y) ** 2)

    def sub_grid_contain_charlie(self):
        """
        This method allow the user to know if the current subgrid contain charlie or not
        :return: true if the sub grid contains charlie.
        """
        window = self.zoom_padding << (self.z - 1)
        return (self.x * window <= self.charlie_x < self.x * window + window and
                self.y * window <= self.charlie_y < self.y * window + window)

    def get_current_state_deep(self):
        """
        give to the agent 2 images (the sub image and the hist image). they are squeeze into
        a single array.
        :return: the current state.
        """
        return np.append(self.sub_vision.squeeze(), self.hist.squeeze()) / 255

    def take_action(self, action):
        """
        This method allow the agent to take an action over the environment.
        :param action: the number of the action that the agent has take.
        :return: the next state, the reward, if the state is terminal and a tips of which action the agent should have
        choose.
        """
        action = Action(action)

        self.history[self.nb_actions_taken] = (self.x, self.y, self.z, action.value)
        if self.evaluation_mode:
            self.record((self.x, self.y, self.z), action.value)

        old_pos = (self.x, self.y, self.z)
        if action == Action.LEFT:
            self.x -= 0 if self.x <= 0 else 1
        elif action == Action.UP:
            self.y -= 0 if self.y <= 0 else 1
        elif action == Action.RIGHT:
            self.x += 0 if (self.x + 1) >= self.W / (self.zoom_padding << (self.z - 1)) else 1
        elif action == Action.DOWN:
            self.y += 0 if (self.y + 1) >= self.H / (self.zoom_padding << (self.z - 1)) else 1
        elif action == Action.ZOOM1:
            if not self.z - 1 < self.min_zoom:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1

        elif action == Action.ZOOM2:
            if not self.z - 1 < self.min_zoom:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1

                self.x += 1

        elif action == Action.ZOOM3:
            if not self.z - 1 < self.min_zoom:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1

                self.y += 1

        elif action == Action.ZOOM4:
            if not self.z - 1 < self.min_zoom:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1

                self.x += 1
                self.y += 1

        elif action == Action.DEZOOM:
            if not self.z + 1 >= self.max_zoom:
                self.x = self.x >> 1
                self.y = self.y >> 1
                self.z += 1

        self.compute_sub_grid()
        self.compute_hist()
        self.nb_actions_taken += 1

        # before the move we must check if the agent should mark
        should_have_mark = self.sub_grid_contain_charlie() and self.z < self.max_zoom - 2


        reward = - (self.get_distance_reward() / self.max_distance)
        #reward = -1

        is_terminal = self.nb_max_actions <= self.nb_actions_taken

        if should_have_mark:
            is_terminal = True
            reward = 100
            if self.difficulty:
                self.init_env()

        return self.get_current_state_deep(), reward, is_terminal, action.value

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

            if a == Action.MARK:
                color = [255, 0, 0]
            else:
                color = [0, 255, 0]

            window = (self.zoom_padding ** z)
            mm[int(window * y * self.ratio):
                      int((window + window * y) * self.ratio),
                      int(window * x * self.ratio):
                      int((window + window * x) * self.ratio)] = color

            frames.append(mm)

        imageio.mimsave(name, frames, duration=0.5)

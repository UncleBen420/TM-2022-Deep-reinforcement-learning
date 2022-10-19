"""
This file implement a dummy environment to train the agents on and compare them. The term "Soft" mean that the
states of the environment are not linked to it's size (contrary to a grid world for exemple).
"""
import math
import random
import re
from enum import Enum
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

def check_cuda():
    """
    check if opencv can detect cuda
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
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    ZOOM1 = 4
    ZOOM2 = 5
    ZOOM3 = 6
    ZOOM4 = 7
    DEZOOM = 8
    MARK = 9

MODEL_RES = 64
HIST_RES = 32

class DummyEnv:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, nb_max_actions=100, replace_charlie=False):
        self.gpu_full_img = None
        self.action_dones = None
        self.charlie_y = 0
        self.charlie_x = 0
        self.reward_grid = None
        self.nb_actions_taken = 0
        self.grid = None
        self.history = []
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        self.nb_action = 7
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
        self.heat_map = np.zeros((HIST_RES, HIST_RES))
        self.replace_charlie = replace_charlie

    def place_charlie(self):

        while True:
            x = random.randint(0, self.W - 1)
            y = random.randint(0, self.H - 1)
            if self.mask[x][y][0] == 0:
                self.charlie_x = x
                self.charlie_y = y
                self.full_img = self.base_img.copy()
                self.full_img[self.charlie_x:self.charlie_x + self.charlie.shape[0],
                self.charlie_y:self.charlie_y + self.charlie.shape[1]] = self.charlie
                break
                

    def reload_env(self):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        del self.history

        self.history = np.zeros((self.nb_max_actions, 4), dtype=int)
        self.action_dones = []
        self.marked = []
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

    def load_env(self, img, mask, charlie):
        self.base_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        self.H, self.W, self.channels = self.base_img.shape
        self.ratio = HIST_RES / self.H
        self.charlie = cv2.cvtColor(cv2.imread(charlie), cv2.COLOR_BGR2RGB)
        self.mask = cv2.imread(mask)

    def init_env(self):
        self.place_charlie()
        if self.cv_cuda:
            self.gpu_full_img = cv2.cuda_GpuMat()
            self.gpu_full_img.upload(self.full_img)

        min_dim = np.min([self.W, self.H])
        self.hist_img = cv2.resize(self.full_img, (HIST_RES, HIST_RES), interpolation=cv2.INTER_NEAREST)

        self.max_zoom = int(math.log(min_dim, 2))
        self.min_zoom = self.max_zoom - 4

    def compute_sub_grid(self):
        window = self.zoom_padding << (self.z - 1)
        if self.cv_cuda:
            minX = window * self.x
            maxY = window + window * self.y
            maxX = window + window * self.x
            minY = window * self.y

            self.gpu_sub_vision = cv2.cuda_GpuMat(self.gpu_full_img, (minY, minX, maxY, maxX))
            self.sub_vision = cv2.cuda.resize(self.sub_vision, (MODEL_RES, MODEL_RES)).download()
        else:
            self.sub_vision = self.full_img[window * self.x:window + window * self.x, window * self.y:window + window * self.y]
            self.sub_vision = cv2.resize(self.sub_vision, (MODEL_RES, MODEL_RES))

    def compute_hist(self):
        window = self.zoom_padding << (self.z - 1)
        self.hist = self.hist_img.copy()
        self.hist[int(window * self.x * self.ratio):
                  int((window + window * self.x) * self.ratio),
                  int(window * self.y * self.ratio):
                  int((window + window * self.y) * self.ratio)] = [255., 0., 0.]
        self.heat_map[int(window * self.x * self.ratio):
                  int((window + window * self.x) * self.ratio),
                  int(window * self.y * self.ratio):
                  int((window + window * self.y) * self.ratio)] += 1

    def get_distance_reward(self):
        dist = math.sqrt(((self.x << (self.max_zoom - self.z)) - (self.charlie_x >> self.min_zoom - 1) ) ** 2 +
                         ((self.y << (self.max_zoom - self.z)) - (self.charlie_y >> self.min_zoom - 1)) ** 2)
        return dist

    def sub_grid_contain_charlie(self):
        window = self.zoom_padding << (self.z - 1)
        return (self.x * window <= self.charlie_x < self.x * window + window and
                self.y * window <= self.charlie_y < self.y * window + window)

    def get_current_state_deep(self):
        return np.append(self.sub_vision.squeeze(), self.hist.squeeze()) / 255


    def take_action(self, action):
        action = Action(action)

        # before the move we must check if the agent should mark
        should_have_mark = self.sub_grid_contain_charlie() and self.z < self.max_zoom - 3

        self.history[self.nb_actions_taken] = (self.x, self.y, self.z, action.value)

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

        if self.guided and not action == Action.MARK and should_have_mark:
            action = Action.MARK
        elif action == Action.MARK and should_have_mark:
            self.marked_correctly = True

        #reward = - self.get_distance_reward()
        reward = -1

        is_terminal = self.nb_max_actions <= self.nb_actions_taken

        if action == Action.MARK:
            self.nb_mark += 1
            if should_have_mark:
                is_terminal = True
                reward = 10
                if self.replace_charlie:
                    self.init_env()
            else:
                reward -= 10

        return self.get_current_state_deep(), reward, is_terminal, action.value

    def get_gif_trajectory(self, name):
        """
        This function allow the user to create a gif of all the moves the
        agent has made along the episodes
        :param environment: the environment on which the agent evolve
        :param trajectory: the trajectory that the agent has take
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
            mm[int(window * x * self.ratio):
                      int((window + window * x) * self.ratio),
                      int(window * y * self.ratio):
                      int((window + window * y) * self.ratio)] = color

            frames.append(mm)

        imageio.mimsave(name, frames, duration=0.5)

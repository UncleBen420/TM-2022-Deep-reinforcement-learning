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

MODEL_RES = 224

class DummyEnv:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, nb_max_actions=100):
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
        self.cv = cv2.cuda if check_cuda() else cv2

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

    def init_env(self, img, label):
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

        self.H, self.W, self.channels = self.full_img.shape
        self.ratio = MODEL_RES / self.H
        print(self.ratio)

        min_dim = np.min([self.W, self.H])
        self.hist_img = self.cv.resize(self.full_img, (MODEL_RES, MODEL_RES), interpolation=cv2.INTER_NEAREST)

        self.max_zoom = int(math.log(min_dim, 2))
        #self.heat_map = np.zeros((self.W, self.H))

        self.bb_map = np.zeros((self.H, self.W), dtype=np.uint8)
        with open(label, "r") as file:
            lines = file.read()
        for i, line in enumerate(lines.split('\n')):
            if len(line.split(' ')) > 1:
                y, x = line.split(' ')
                self.charlie_x = int(float(x))
                self.charlie_y = int(float(y))

        print(self.charlie_x)
        print(self.charlie_y)

    def compute_sub_grid(self):
        window = self.zoom_padding << (self.z - 1)
        self.sub_vision = self.full_img[window * self.x:window + window * self.x, window * self.y:window + window * self.y]
        self.sub_vision = self.cv.resize(self.sub_vision, (MODEL_RES, MODEL_RES))


    def compute_hist(self):
        window = self.zoom_padding << (self.z - 1)
        self.hist = self.hist_img.copy()
        self.hist[int(window * self.x * self.ratio):
                  int((window + window * self.x) * self.ratio),
                  int(window * self.y * self.ratio):
                  int((window + window * self.y) * self.ratio)] = [255., 0., 0.]

    def get_distance_reward(self):
        dist = math.sqrt(((self.x << (self.max_zoom - self.z)) - self.charlie_x) ** 2 +
                         ((self.y << (self.max_zoom - self.z)) - self.charlie_y) ** 2)
        return dist

    def sub_grid_contain_charlie(self):
        window = self.zoom_padding << (self.z - 1)
        return (self.x * window <= self.charlie_x <= self.x * window + window and
                self.y * window <= self.charlie_y <= self.y * window + window)

    def get_current_state_deep(self):
        return np.append(self.sub_vision.squeeze(), self.hist.squeeze()) / 127.5 - 1


    def take_action(self, action):
        action = Action(action)

        # before the move we must check if the agent should mark
        should_have_mark = self.sub_grid_contain_charlie() and self.z < self.max_zoom - 2
        if should_have_mark:
            plt.imshow(self.sub_vision)
            plt.show()

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
            if not self.z - 1 <= 0:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1

        elif action == Action.ZOOM2:
            if not self.z - 1 <= 0:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1

                self.x += 1

        elif action == Action.ZOOM3:
            if not self.z - 1 <= 0:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1

                self.y += 1

        elif action == Action.ZOOM4:
            if not self.z - 1 <= 0:
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

        reward = - self.get_distance_reward()

        is_terminal = self.nb_max_actions <= self.nb_actions_taken

        if action == Action.MARK:
            self.nb_mark += 1
            if should_have_mark:
                is_terminal = True
                reward += 1000
            else:
                reward -= 1000

        elif old_pos == (self.x, self.y, self.z):
            reward -= 2

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
        mm = self.render_board_img()
        for i in range(self.nb_actions_taken):
            x, y, z, a = self.history[i]

            if a == Action.MARK:
                color = [0, 0, 1]
            else:
                color = [0, 1, 0]

            window = (self.zoom_padding ** z) * 10
            mm[window * x:window + window * x
              ,window * y:window + window * y] = mm[window * x:window + window * x
                                                         ,window * y:window + window * y] >> color

            frames.append(mm.copy())

        imageio.mimsave(name, frames, duration=0.5)


class PieceRender(Enum):
    """
    this enum class represent the visualisation of the board.
    """
    CHARLIE = [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    HOUSE = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
             [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
             [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    WATER = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]]

    GROUND = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
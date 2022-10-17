"""
This file implement a dummy environment to train the agents on and compare them. The term "Soft" mean that the
states of the environment are not linked to it's size (contrary to a grid world for exemple).
"""
import math
import random
from enum import Enum
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


class Piece(Enum):
    """
    this enum class simplify the representation of the different
    pieces on the board.
    """
    CHARLIE = 1.
    WATER = 0.5
    GROUND =  0.


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


class DummyEnv:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, size=64, model_resolution=2, max_zoom=4, nb_max_actions=100,
                 replace_charlie=True, replace_env=False):
        self.action_dones = None
        self.charlie_y = 0
        self.charlie_x = 0
        self.reward_grid = None
        self.nb_actions_taken = 0
        self.grid = None
        self.history = []
        self.heat_map = np.zeros((size, size))
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        self.nb_action = 7

        self.size = size
        self.model_resolution = model_resolution
        self.max_zoom = int(math.log(size, self.model_resolution))
        self.max_move = int(self.size / self.model_resolution)
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
        self.replace_charlie = replace_charlie
        self.replace_env = replace_env

    def place_charlie(self):
        while True:
            self.grid[self.charlie_x][self.charlie_y] = 0.5 + (random.random() / 10.)
            x = random.randint(0, self.size - 1)
            y = random.randint(1, self.size - 1)
            if self.grid[x][y] >= 0.5 and (0 <= self.grid[x][y - 1] < 0.5):
                self.grid[x][y] = 1.
                self.charlie_x = x
                self.charlie_y = y
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

    def init_env(self):

        def dilate():
            temp = np.zeros((self.size, self.size), dtype=float)
            for i in range(self.size):
                for j in range(self.size):
                    temp[i][j] = self.grid[i][j]
                    current = 0

                    current += 1 if i < self.size - 1 and self.grid[i + 1][j] >= 0.5 else 0
                    current += 1 if j < self.size - 1 and self.grid[i][j + 1] >= 0.5 else 0
                    current += 1 if i > 0 and self.grid[i - 1][j] >= 0.5 else 0
                    current += 1 if j > 0 and self.grid[i][j - 1] >= 0.5 else 0
                    if current:
                        temp[i][j] = 0.5
            self.grid = temp

        self.grid = np.zeros((self.size * self.size), dtype=float)

        for i in range(5):
            self.grid[i] = 0.5

        np.random.shuffle(self.grid)

        self.grid = self.grid.reshape((self.size, self.size)) + (np.random.rand(self.size, self.size) / 10)
        for _ in range(10):
            dilate()

        self.place_charlie()

    def compute_sub_grid(self):
        window = self.model_resolution << (self.z - 1)
        self.sub_grid = self.grid[window * self.x:window + window * self.x, window * self.y:window + window * self.y]

        self.sub_vision = np.zeros((self.sub_grid.shape[0], self.sub_grid.shape[1], 3), dtype=float)
        self.sub_vision += self.sub_grid[:,:,None]
        #img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_NEAREST)
        self.sub_vision = cv2.resize(self.sub_vision, (10, 10))


    def compute_hist(self):
        window = self.model_resolution << (self.z - 1)

        self.hist = np.zeros((self.size, self.size, 3), dtype=float) + self.grid[:, :, None]
        self.hist[window * self.x:window + window * self.x, window * self.y:window + window * self.y] = [1., 0., 0.]
        self.heat_map[window * self.x:window + window * self.x, window * self.y:window + window * self.y] += 1.

        #img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_NEAREST)
        self.hist = cv2.resize(self.hist, (10, 10), interpolation=cv2.INTER_NEAREST)

    def get_distance_reward(self):
        return math.sqrt(((self.x << (self.max_zoom - self.z)) - self.charlie_x) ** 2 +
                         ((self.y << (self.max_zoom - self.z)) - self.charlie_y) ** 2)

    def get_current_state_deep(self):
        return np.append(self.sub_vision.squeeze(), self.hist.squeeze())

    def take_action(self, action):
        action = Action(action)

        # before the move we must check if the agent should mark
        should_have_mark = np.count_nonzero(self.sub_vision >= 0.8)

        self.history[self.nb_actions_taken] = (self.x, self.y, self.z, action.value)

        old_pos = (self.x, self.y, self.z)
        if action == Action.LEFT:
            self.x -= 0 if self.x <= 0 else 1
        elif action == Action.UP:
            self.y -= 0 if self.y <= 0 else 1
        elif action == Action.RIGHT:
            self.x += 0 if (self.x + 1) >= self.size / (self.model_resolution << (self.z - 1)) else 1
        elif action == Action.DOWN:
            self.y += 0 if (self.y + 1) >= self.size / (self.model_resolution << (self.z - 1)) else 1
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

                if self.replace_env:
                    self.init_env()
                elif self.replace_charlie:
                    self.place_charlie()



            else:
                reward -= 1000

        elif old_pos == (self.x, self.y, self.z):
            reward -= 2

        return self.get_current_state_deep(), reward, is_terminal, action.value

    def render_board_img(self):
        """
        this method allow to get a image representation of the board in
        the form of a numpy array.
        :param agent_state: the current position of the agent
        :return: a numpy array representing the board
        """
        visual = np.ones((self.size * 10, self.size * 10, 3), dtype=np.uint8)

        for i in range(self.size):
            for j in range(self.size):

                render = np.ones((10, 10, 3), dtype=np.uint8)

                piece = math.floor(self.grid[i][j] * 10) / 10

                piece = Piece(piece)

                if piece == Piece.CHARLIE:
                    icon = PieceRender.CHARLIE.value
                    color = [1, 0, 0]
                elif piece == Piece.WATER:
                    icon = PieceRender.WATER.value
                    color = [0, 0, 1]
                elif piece == Piece.GROUND:
                    icon = PieceRender.GROUND.value
                    color = [1, 1, 1]

                icon = np.array(icon, dtype=np.uint8)
                render *= icon[:, :, None]

                visual[(i * 10):((i + 1) * 10), (j * 10):((j + 1) * 10)][:] *= render * np.array(color, dtype=np.uint8)

        return visual * 255

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

            window = (self.model_resolution ** z) * 10
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
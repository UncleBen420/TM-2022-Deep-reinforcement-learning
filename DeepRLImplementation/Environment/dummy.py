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
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class TransformDummy:
    """
    Class use to transform the input image in a normalised one.
    """

    def __init__(self):
        self.transforms = A.Compose([
            # A.Resize(resize_to, resize_to),
            ToTensorV2()
        ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']


class Piece(Enum):
    """
    this enum class simplify the representation of the different
    pieces on the board.
    """
    BOAT = {"code": 'B', 'label': 0}
    HOUSE = {"code": 'H', 'label': 1}
    WATER = {"code": '~', 'label': 2}
    GROUND = {"code": '^', 'label': 3}


class Event(Enum):
    """
    this enum class simplify the different state of the grid
    """
    UNKNOWN = 0
    VISITED = 1
    MARKED = 2
    BLOCKED = 3


class Action(Enum):
    """
    this enum class represent all the action that the agent is allowed to do.
    """
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    ZOOM = 4
    DEZOOM = 5
    MARK = 6


class DummyEnv:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, size=128, model_resolution=16, max_zoom=4, nb_max_actions=1000):
        self.grid = None
        self.history = []
        self.marked = []
        self.marked_map = np.zeros((size, size), dtype=bool)
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        self.nb_action = 7
        self.size = size
        self.model_resolution = model_resolution
        self.pad = 2
        self.min_zoom = 4
        self.max_zoom = int(math.log(size, self.pad)) - 2
        self.z = max_zoom - 1
        self.x = 0
        self.y = 0
        # (x, y)
        self.sub_grid = None
        # State of the environment
        self.dummy_boat_model = None
        self.dummy_surface_model = None
        self.vision = np.zeros(7, dtype=int)
        self.transform = TransformDummy()
        self.evolution = []

        def preprocess(i):
            """
            This function is vectorize over all the pieces on the subgrid. It gives the probability of having
            a boat or a house.
            :param i: the Piece that is analysed.
            :return: the changed value of i
            """
            return i.value["label"]

        self.preprocess = np.vectorize(preprocess)

    def reload_env(self):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        del self.history
        del self.marked
        del self.marked_map

        self.history = []
        self.marked = []
        self.marked_map = np.zeros((self.size, self.size), dtype=bool)
        self.nb_actions_taken = 0
        self.z = random.randint(1, self.max_zoom - 1)
        self.x = random.randint(0, self.size / (self.pad ** self.z) - 1)
        self.y = random.randint(0, self.size / (self.pad ** self.z) - 1)
        self.compute_sub_grid()
        self.get_vision()

        return self.get_current_state()

    def init_env(self):

        def dilate():
            temp = np.full((self.size, self.size), Piece.GROUND, dtype=Piece)
            for i in range(self.size):
                for j in range(self.size):
                    temp[i][j] = self.grid[i][j]
                    current = 0
                    current += 1 if i < self.size - 1 and self.grid[i + 1][j] == Piece.WATER else 0
                    current += 1 if j < self.size - 1 and self.grid[i][j + 1] == Piece.WATER else 0
                    current += 1 if i > 0 and self.grid[i - 1][j] == Piece.WATER else 0
                    current += 1 if j > 0 and self.grid[i][j - 1] == Piece.WATER else 0
                    if current:
                        temp[i][j] = Piece.WATER
            self.grid = temp

        def place_boat(i):
            if i is Piece.WATER:
                p = random.normalvariate(0.3, 0.3)
                if p > 0.8:
                    return Piece.BOAT
            return i

        def place_car(i):
            if i is Piece.GROUND:
                p = random.normalvariate(0.3, 0.3)
                if p > 0.8:
                    return Piece.HOUSE
            return i

        self.grid = np.full((self.size * self.size), Piece.GROUND, dtype=Piece)

        for i in range(10):
            self.grid[i] = Piece.WATER

        np.random.shuffle(self.grid)

        self.grid = self.grid.reshape((self.size, self.size))
        for _ in range(20):
            dilate()

        place_all_boat = np.vectorize(place_boat)
        self.grid = place_all_boat(self.grid)

        place_all_car = np.vectorize(place_car)
        self.grid = place_all_car(self.grid)

    def transform(self, img):
        """
        Normalize and turn the images into a tensor.
        :param img:
        :return:
        """
        img_pil = Image.fromarray(img)
        return self.transform(img_pil)

    def compute_sub_grid(self):
        window = self.pad ** self.z
        self.sub_grid = self.grid[window * self.x:window + window * self.x, window * self.y:window + window * self.y]
        # normalization
        self.sub_img = (self.preprocess(self.sub_grid) / 3 * 255).astype(np.uint8)
        self.sub_img = cv2.resize(self.sub_img, (self.model_resolution,
                                                 self.model_resolution), cv2.INTER_NEAREST).astype(float)

        self.sub_img += np.random.rand(self.model_resolution, self.model_resolution) / 500.

    def get_vision(self):
        move_set = [(self.x - 1, self.y, self.z),
                    (self.x + 1, self.y, self.z),
                    (self.x, self.y - 1, self.z),
                    (self.x, self.y + 1, self.z),
                    (self.x, self.y, self.z - 1),
                    (self.x, self.y, self.z + 1),
                    (self.x, self.y, self.z)]
        # check if a place had already been visited or marked
        for i in range(7):
            if move_set[i] in self.history:
                self.vision[i] = Event.VISITED.value
            elif move_set[i] in self.marked:
                self.vision[i] = Event.MARKED.value
            else:
                self.vision[i] = Event.UNKNOWN.value

        self.vision[0] = Event.BLOCKED.value if self.x <= 0 else self.vision[0]
        self.vision[1] = Event.BLOCKED.value if (self.x + 1) >= self.size / (
                self.pad ** self.z) else self.vision[1]

        self.vision[2] = Event.BLOCKED.value if self.y <= 0 else self.vision[2]
        self.vision[3] = Event.BLOCKED.value if (self.y + 1) >= self.size / (
                self.pad ** self.z) else self.vision[3]

        self.vision[4] = Event.BLOCKED.value if self.z - 1 < self.min_zoom else self.vision[4]
        self.vision[5] = Event.BLOCKED.value if self.z + 1 >= self.max_zoom else self.vision[5]

    def get_current_state(self):
        img = self.transform(self.sub_img).float()
        #vision = self.vision / 3 + np.random.rand(1, len(self.vision)) / 100.0
        #vision = torch.FloatTensor(vision)
        vision = np.zeros_like(self.marked_map, dtype=np.double)
        vision[self.marked_map] = 0.5
        window = self.pad ** self.z
        vision[window * self.x:window + window * self.x, window * self.y:window + window * self.y] = 1.
        vision = torch.FloatTensor(vision.reshape(-1))
        return img.unsqueeze(0), vision.unsqueeze(0)

    def mark(self):
        self.marked.append((self.x, self.y, self.z))

    def mark_map(self):
        window = self.pad ** self.z
        self.marked_map[window * self.x:window + window * self.x, window * self.y:window + window * self.y] = True

    def get_marked_map(self):
        window = self.pad ** self.z
        return self.marked_map[window * self.x:window + window * self.x, window * self.y:window + window * self.y]

    def get_marked_percent(self):
        total_piece = np.count_nonzero(self.grid == Piece.BOAT)
        marked_piece = np.count_nonzero(self.grid[self.marked_map] == Piece.BOAT)
        return (total_piece - marked_piece) / total_piece * 100

    def get_reward(self, action):
        reward = -1

        if action == Action.MARK and not self.marked[-1] in self.marked[:-1]:
            reward += np.count_nonzero(self.sub_grid == Piece.BOAT) * 10
            reward -= np.count_nonzero(self.sub_grid == Piece.HOUSE) * 10
        elif action == Action.MARK:
            reward -= 100

        if self.history[-1] in self.history[:-1]:
            reward -= 10

        return reward


    def get_reward2(self, action):
        reward = -1 * self.z
        nb_boat = np.count_nonzero(self.sub_grid == Piece.BOAT)
        nb_house = np.count_nonzero(self.sub_grid == Piece.HOUSE)
        factor = nb_boat / self.sub_grid.size * 100

        if action == Action.MARK and not self.marked[-1] in self.marked[:-1]:
            reward += nb_boat * factor

        elif action == Action.MARK:
            reward -= 100

        if self.history[-1] in self.history[:-1]:
            reward -= 10
        elif nb_boat > nb_house:
            reward += factor

        return reward

    def take_action(self, action):
        action = Action(action)

        self.history.append((self.x, self.y, self.z))

        if action == Action.LEFT:
            self.x -= 1 if self.vision[0] != Event.BLOCKED.value else 0
        elif action == Action.UP:
            self.y -= 1 if self.vision[2] != Event.BLOCKED.value else 0
        elif action == Action.RIGHT:
            self.x += 1 if self.vision[1] != Event.BLOCKED.value else 0
        elif action == Action.DOWN:
            self.y += 1 if self.vision[3] != Event.BLOCKED.value else 0
        elif action == Action.ZOOM:
            self.z -= 1 if self.vision[4] != Event.BLOCKED.value else 0
        elif action == Action.DEZOOM:
            if self.vision[5] != Event.BLOCKED.value:
                self.x = int(self.x / self.pad)
                self.y = int(self.y / self.pad)
                self.z += 1
        elif action == Action.MARK:
            self.mark()
            self.mark_map()

        self.compute_sub_grid()
        self.get_vision()
        self.nb_actions_taken += 1
        is_terminal = self.nb_max_actions <= self.nb_actions_taken or self.get_marked_percent() <= 5.
        if is_terminal:
            self.evolution.append(self.render_board_img(self.grid, self.marked_map, [1, 0, 0]))
        return self.get_current_state(), self.get_reward(action), is_terminal

    def render_grid(self, grid):
        """
        this method allow to get a string representation of the board.
        for command line usage.
        :return: the string representation of the board
        """
        visual = ""
        for i in range(grid.shape[0]):
            visual += '_'
        visual += '\n'
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                visual += grid[i, j].value["code"]

            visual += '\n'
        for i in range(grid.shape[0]):
            visual += '_'
        return visual

    def render_board_img(self, grid, map, color):
        """
        this method allow to get a image representation of the board in
        the form of a numpy array.
        :param agent_state: the current position of the agent
        :return: a numpy array representing the board
        """
        visual = np.ones((map.shape[0] * 10, map.shape[1] * 10, 3), dtype=np.uint8)

        for i in range(map.shape[0]):
            for j in range(map.shape[1]):

                render = np.ones((10, 10, 3), dtype=np.uint8)

                if grid[i][j] == Piece.BOAT:
                    icon = PieceRender.BOAT.value
                elif grid[i][j] == Piece.HOUSE:
                    icon = PieceRender.HOUSE.value
                elif grid[i][j] == Piece.WATER:
                    icon = PieceRender.WATER.value
                elif grid[i][j] == Piece.GROUND:
                    icon = PieceRender.GROUND.value

                icon = np.array(icon, dtype=np.uint8)
                render *= icon[:, :, None]
                if map[i][j]:
                    render[:, :] *= np.uint8(color)

                visual[(i * 10):((i + 1) * 10), (j * 10):((j + 1) * 10)][:] *= render

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
        mm = self.render_board_img(self.marked_map, [1, 0, 0])
        for iteration in self.history:
            temp_map = np.zeros((self.size, self.size), dtype=bool)

            window = self.pad ** iteration[2]
            temp_map[window * iteration[0]:window + window * iteration[0]
            , window * iteration[1]:window + window * iteration[1]] = True

            frames.append(cv2.bitwise_and(self.render_board_img(temp_map, [0, 0, 1]), mm))

        imageio.mimsave(name, frames, duration=0.05)


class PieceRender(Enum):
    """
    this enum class represent the visualisation of the board.
    """
    BOAT = [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]]

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

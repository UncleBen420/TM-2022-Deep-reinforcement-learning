import math
import random
from enum import Enum

import cv2
import imageio
import numpy as np


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
    this enum class simplify the representation of the different
    pieces on the board.
    """
    UNKNOWN = 0
    VISITED = 1
    MARKED = 2
    BLOCKED = 3


class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    ZOOM = 4
    DEZOOM = 5
    MARK = 6


class DummyEnv:
    """
    this class implement the grid world problem as a frozen lake problem.
    """

    def __init__(self, size=64, model_resolution=2, max_zoom=4, nb_max_actions=100):
        self.nb_actions_taken = 0
        self.grid = None
        self.history = []
        self.marked = []
        self.marked_map = np.zeros((size, size), dtype=bool)
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        self.nb_action = 7
        self.size = size
        self.model_resolution = model_resolution
        self.max_zoom = max_zoom
        self.zoom_factor = max_zoom - 1
        # (x, y)
        self.move_factor = [0, 0]
        self.sub_grid = None
        # State of the environment
        self.dummy_boat_model = None
        self.dummy_surface_model = None
        self.vision = np.zeros(7, dtype=int)
        self.states = np.arange(2 * 2 * (4 ** 6) * 3).reshape((2, 2, 4, 4, 4, 4, 4, 4, 3))

        def model_probalities(i):
            if i is Piece.BOAT:
                # simulate a neural network, the more the agent zoom the more the probability of
                # seeing the boat increase
                if not np.random.binomial(1, 9. * (10 ** (-1 * self.zoom_factor))):
                    i = Piece.WATER
            elif i is Piece.HOUSE:
                # simulate a case where the model mistake a house for a boat
                if not np.random.binomial(1, 9. * (10 ** (-1 * self.zoom_factor))):
                    i = Piece.GROUND
                #elif np.random.binomial(1, 0.05):
                #    i = Piece.BOAT
            return i

        self.get_probabilities = np.vectorize(model_probalities)

    def reload_env(self):
        del self.history
        del self.marked
        del self.marked_map

        self.history = []
        self.marked = []
        self.marked_map = np.zeros((self.size, self.size), dtype=bool)
        self.nb_actions_taken = 0
        self.move_factor = [0, 0]
        self.zoom_factor = self.max_zoom - 1
        self.compute_sub_grid()
        self.fit_dummy_model()
        self.get_vision()

        return self.get_current_state()

    def reload_env_random_state(self):
        del self.history
        del self.marked
        del self.marked_map

        self.history = []
        self.marked = []
        self.marked_map = np.zeros((self.size, self.size), dtype=bool)
        self.nb_actions_taken = 0
        self.zoom_factor = random.randint(1, self.max_zoom - 1)
        self.move_factor = [random.randint(0, self.size - 1) / self.zoom_factor,
                            random.randint(0, self.size - 1) / self.zoom_factor]

        self.compute_sub_grid()
        self.fit_dummy_model()
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
                if p > 0.7:
                    return Piece.HOUSE
            return i

        self.grid = np.full((self.size * self.size), Piece.GROUND, dtype=Piece)

        for i in range(10):
            self.grid[i] = Piece.WATER

        np.random.shuffle(self.grid)

        self.grid = self.grid.reshape((self.size, self.size))
        for _ in range(10):
            dilate()

        place_all_boat = np.vectorize(place_boat)
        self.grid = place_all_boat(self.grid)

        place_all_car = np.vectorize(place_car)
        self.grid = place_all_car(self.grid)

    def compute_sub_grid(self):

        #pad = self.model_resolution ** (self.zoom_factor - 1)
        window = self.model_resolution ** self.zoom_factor
        self.sub_grid = self.grid[window * self.move_factor[0]:window + window * self.move_factor[0]
        , window * self.move_factor[1]:window + window * self.move_factor[1]]

    def get_vision(self):
        move_set = [(self.move_factor[0] - 1, self.move_factor[1], self.zoom_factor),
                    (self.move_factor[0] + 1, self.move_factor[1], self.zoom_factor),
                    (self.move_factor[0], self.move_factor[1] - 1, self.zoom_factor),
                    (self.move_factor[0], self.move_factor[1] + 1, self.zoom_factor),
                    (self.move_factor[0], self.move_factor[1], self.zoom_factor - 1),
                    (self.move_factor[0], self.move_factor[1], self.zoom_factor + 1),
                    (self.move_factor[0], self.move_factor[1], self.zoom_factor)]
        # check if a place had already been visited or marked
        for i in range(7):
            self.vision[i] = (Event.VISITED.value if move_set[i] in self.history
                              else Event.UNKNOWN.value)
            self.vision[i] = (Event.MARKED.value if move_set[i] in self.marked
                              else self.vision[i])

        self.vision[0] = Event.BLOCKED.value if self.move_factor[0] <= 0 else self.vision[0]
        self.vision[1] = Event.BLOCKED.value if (self.move_factor[0] + 1) >= self.size / (
                self.model_resolution ** self.zoom_factor) else self.vision[1]

        self.vision[2] = Event.BLOCKED.value if self.move_factor[1] <= 0 else self.vision[2]
        self.vision[3] = Event.BLOCKED.value if (self.move_factor[1] + 1) >= self.size / (
                self.model_resolution ** self.zoom_factor) else self.vision[3]

        self.vision[4] = Event.BLOCKED.value if self.zoom_factor - 1 <= 0 else self.vision[4]
        self.vision[5] = Event.BLOCKED.value if self.zoom_factor + 1 >= self.max_zoom else self.vision[5]

    def fit_dummy_model(self):
        proba = self.get_probabilities(self.sub_grid)
        self.dummy_boat_model = 1 if np.count_nonzero(proba == Piece.BOAT) else 0
        self.dummy_surface_model = 1 if np.count_nonzero(proba == Piece.WATER) > np.count_nonzero(
            proba == Piece.GROUND) else 0

    def get_current_state(self):
        return \
            self.states[self.dummy_boat_model][self.dummy_surface_model][self.vision[0]][self.vision[1]][
                self.vision[2]][
                self.vision[3]][self.vision[4]][self.vision[5]][self.vision[6]]

    def get_nb_state(self):
        return self.states.size

    def mark(self):
        self.marked.append((self.move_factor[0], self.move_factor[1], self.zoom_factor))

    def get_marked_map(self):
        window = self.model_resolution ** self.zoom_factor
        self.marked_map[window * self.move_factor[0]:window + window * self.move_factor[0]
        , window * self.move_factor[1]:window + window * self.move_factor[1]] = True

    def get_remaining_piece(self, piece):
        total_piece = np.count_nonzero(self.grid == piece)
        marked_piece = np.count_nonzero(self.grid[self.marked_map] == piece)
        return total_piece - marked_piece

    def get_reward(self, action, is_terminal):
        reward = -1
        if action == Action.MARK and not self.marked[-1] in self.marked[:-1]:
            reward += np.count_nonzero(self.sub_grid == Piece.BOAT) * 10
            reward -= np.count_nonzero(self.sub_grid == Piece.HOUSE) * 10
        elif action == Action.MARK:
            reward -100

        if is_terminal:
            print(self.nb_actions_taken)

        return reward

    def take_action(self, action):
        action = Action(action)

        self.history.append((self.move_factor[0], self.move_factor[1], self.zoom_factor))

        if action == Action.LEFT:
            self.move_factor[0] -= 1 if self.vision[0] != Event.BLOCKED.value else 0
        elif action == Action.UP:
            self.move_factor[1] -= 1 if self.vision[2] != Event.BLOCKED.value else 0
        elif action == Action.RIGHT:
            self.move_factor[0] += 1 if self.vision[1] != Event.BLOCKED.value else 0
        elif action == Action.DOWN:
            self.move_factor[1] += 1 if self.vision[3] != Event.BLOCKED.value else 0
        elif action == Action.ZOOM:
            self.zoom_factor -= 1 if self.vision[4] != Event.BLOCKED.value else 0
        elif action == Action.DEZOOM:
            if self.vision[5] != Event.BLOCKED.value:
                self.move_factor[0] = int(self.move_factor[0] / self.model_resolution)
                self.move_factor[1] = int(self.move_factor[1] / self.model_resolution)
                self.zoom_factor += 1
        elif action == Action.MARK:
            self.mark()
            self.get_marked_map()

        self.compute_sub_grid()
        self.get_vision()
        self.fit_dummy_model()

        new_state = self.get_current_state()
        is_terminal = self.nb_max_actions < self.nb_actions_taken or self.get_remaining_piece(Piece.BOAT) < 15

        reward = self.get_reward(action, is_terminal)
        self.nb_actions_taken += 1

        return new_state, reward, is_terminal

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

    def render_board_img(self, map, color):
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

                if self.grid[i][j] == Piece.BOAT:
                    icon = PieceRender.BOAT.value
                elif self.grid[i][j] == Piece.HOUSE:
                    icon = PieceRender.HOUSE.value
                elif self.grid[i][j] == Piece.WATER:
                    icon = PieceRender.WATER.value
                elif self.grid[i][j] == Piece.GROUND:
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

            window = self.model_resolution ** iteration[2]
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

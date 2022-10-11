"""
This file implement a dummy environment to train the agents on and compare them. The term "Soft" mean that the
states of the environment are not linked to it's size (contrary to a grid world for exemple).
"""
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
    CHARLIE = {"code": 'B', 'label': 0}
    WATER = {"code": '~', 'label': 2}
    GROUND = {"code": '^', 'label': 3}


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
    ZOOM = 4
    DEZOOM = 5
    MARK = 6


class DummyEnv:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, size=64, model_resolution=2, max_zoom=4, nb_max_actions=100, replace_charlie=True, full_vision=True, deep=False):
        self.charlie_y = 0
        self.charlie_x = 0
        self.reward_grid = None
        self.nb_actions_taken = 0
        self.grid = None
        self.history = []
        self.marked = []
        self.marked_map = np.zeros((size, size), dtype=bool)
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        self.nb_action = 7
        self.deep_res = 5
        if full_vision:
            self.deep_states_size = self.deep_res + 3
        else:
            self.deep_states_size = self.deep_res + 7
        self.size = size
        self.model_resolution = model_resolution
        self.max_zoom = int(math.log(size, self.model_resolution)) - 1
        self.max_move = int(self.size / self.model_resolution)
        self.z = max_zoom - 1
        self.x = 0
        self.y = 0
        # (x, y)
        self.sub_grid = None
        # State of the environment
        self.dummy_boat_model = None
        self.dummy_surface_model = None
        self.vision = np.zeros(7, dtype=int)

        self.min_reward = 0
        self.max_reward = 1

        self.replace_charlie = replace_charlie
        self.full_vision = full_vision
        self.deep = deep
        if full_vision:
            self.states = np.arange(2 * 3 * self.max_zoom * self.max_move * self.max_move).reshape((2, 3, self.max_zoom, self.max_move, self.max_move))
        else:
            self.states = np.arange(2 * 3 * (3 ** 6) * 2).reshape((2, 3, 3, 3, 3, 3, 3, 3, 2))

        self.deep_state = np.zeros((self.max_move, self.max_move, self.max_zoom))

        def model_probalities(i):
            """
            This function is vectorize over all the pieces on the subgrid. It gives the probability of having
            a boat or a house.
            :param i: the Piece that is analysed.
            :return: the changed value of i
            """
            if i is Piece.CHARLIE:
                # simulate a neural network, the more the agent zoom the more the probability of
                # seeing the boat increase
                if not np.random.binomial(1, .95 / self.z):
                    i = Piece.WATER
            return i

        self.get_probabilities = np.vectorize(model_probalities)

    def place_charlie(self):
        while True:
            self.grid[self.charlie_x][self.charlie_y] = Piece.WATER
            x = random.randint(0, self.size - 1)
            y = random.randint(1, self.size - 1)
            if self.grid[x][y] is Piece.WATER and self.grid[x][y - 1] is Piece.GROUND:
                self.grid[x][y] = Piece.CHARLIE
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
        del self.marked
        del self.marked_map

        self.history = []
        self.marked = []
        self.marked_map = np.zeros((self.size, self.size), dtype=bool)
        self.nb_actions_taken = 0
        self.z = 1
        #self.x = random.randint(0, self.max_move - 1)
        #self.y = random.randint(0, self.max_move - 1)
        self.x = 0
        self.y = 0
        self.nb_mark = 0
        if self.replace_charlie:
            self.place_charlie()

        self.compute_sub_grid()
        self.fit_dummy_model()
        self.get_vision()

        if self.deep:
            S = self.get_current_state_deep()
        else:
            S = self.get_current_state()

        return S

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

        self.grid = np.full((self.size * self.size), Piece.GROUND, dtype=Piece)
        self.reward_grid = np.zeros((self.size, self.size))

        for i in range(5):
            self.grid[i] = Piece.WATER

        np.random.shuffle(self.grid)

        self.grid = self.grid.reshape((self.size, self.size))
        for _ in range(10):
            dilate()

        self.place_charlie()



    def compute_sub_grid(self):
        window = self.model_resolution ** self.z
        self.sub_grid = self.grid[window * self.x:window + window * self.x, window * self.y:window + window * self.y]

    def get_distance_reward(self):
        return - math.sqrt((self.x - self.charlie_x) ** 2 + (self.y - self.charlie_y) ** 2 + (self.z - 1) ** 2)

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
            else:
                self.vision[i] = Event.UNKNOWN.value

        self.vision[0] = Event.BLOCKED.value if self.x <= 0 else self.vision[0]
        self.vision[1] = Event.BLOCKED.value if (self.x + 1) >= self.size / (
                self.model_resolution ** self.z) else self.vision[1]

        self.vision[2] = Event.BLOCKED.value if self.y <= 0 else self.vision[2]
        self.vision[3] = Event.BLOCKED.value if (self.y + 1) >= self.size / (
                self.model_resolution ** self.z) else self.vision[3]

        self.vision[4] = Event.BLOCKED.value if self.z - 1 <= 0 else self.vision[4]
        self.vision[5] = Event.BLOCKED.value if self.z + 1 >= self.max_zoom else self.vision[5]

    def fit_dummy_model(self):
        proba = self.get_probabilities(self.sub_grid)
        self.dummy_boat_model = 1 if np.count_nonzero(proba == Piece.CHARLIE) else 0
        if np.count_nonzero(proba == Piece.GROUND) and np.count_nonzero(proba == Piece.WATER):
            self.dummy_surface_model = 2
        elif np.count_nonzero(proba == Piece.WATER):
            self.dummy_surface_model = 1
        else:
            self.dummy_surface_model = 0

    def get_current_state(self):
        if self.full_vision:
            return self.states[self.dummy_boat_model][self.dummy_surface_model][self.z - 1][self.x][self.y]

        return self.states[self.dummy_boat_model][self.dummy_surface_model][self.vision[0]][self.vision[1]][
            self.vision[2]][self.vision[3]][self.vision[4]][self.vision[5]][self.vision[6]]

    def sub_grid_value(self, i):
        """
        This function is vectorize over all the pieces on the subgrid. It gives the probability of having
        a boat or a house.
        :param i: the Piece that is analysed.
        :return: the changed value of i
        """
        return i.value['label']

    def get_current_state_deep(self):

        deep_vision = []
        deep_vision.append(self.dummy_boat_model)
        deep_vision.append(self.dummy_surface_model)
        deep_vision.append(self.vision[0])
        deep_vision.append(self.vision[1])
        deep_vision.append(self.vision[2])
        deep_vision.append(self.vision[3])
        deep_vision.append(self.vision[4])
        deep_vision.append(self.vision[5])
        deep_vision.append(self.vision[6])

        return np.array(deep_vision, dtype=float)

    def get_nb_state(self):
        return self.states.size

    def take_action(self, action):
        action = Action(action)

        self.history.append((self.x, self.y, self.z))

        self.compute_sub_grid()

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
                self.x = int(self.x / self.model_resolution)
                self.y = int(self.y / self.model_resolution)
                self.z += 1

        self.get_vision()
        self.fit_dummy_model()
        self.nb_actions_taken += 1

        self.get_current_state_deep()

        reward = self.get_distance_reward()
        #if self.history[-1] in self.history[:-1]:
        #    reward = - reward ** 2

        is_terminal = self.nb_max_actions <= self.nb_actions_taken

        should_have_mark = self.z <= 1 and np.count_nonzero(self.sub_grid == Piece.CHARLIE)

        if action == Action.MARK:
            self.nb_mark += 1
            if should_have_mark:
                is_terminal = True
                reward = 10000
            else:
                reward = - reward ** 2
        if should_have_mark:
            reward = 10000
            is_terminal = True
            print("dumb_guck")

        if self.deep:
            S = self.get_current_state_deep()
        else:
            S = self.get_current_state()

        return S, reward, is_terminal, should_have_mark

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

                if self.grid[i][j] == Piece.CHARLIE:
                    icon = PieceRender.CHARLIE.value
                    color = [1, 0, 0]
                elif self.grid[i][j] == Piece.WATER:
                    icon = PieceRender.WATER.value
                    color = [0, 0, 1]
                elif self.grid[i][j] == Piece.GROUND:
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
        for i in self.history:

            window = (self.model_resolution ** i[2]) * 10
            mm[window * i[0]:window + window * i[0]
              ,window * i[1]:window + window * i[1]] = mm[window * i[0]:window + window * i[0]
                                                         ,window * i[1]:window + window * i[1]] >> [0, 1, 0]

            frames.append(mm.copy())

        imageio.mimsave(name, frames, duration=0.05)


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
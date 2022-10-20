"""
This file implement a dummy environment to train the agents on and compare them. The term "Soft" mean that the
states of the environment are not linked to it's size (contrary to a grid world for exemple).
"""
import math
import random
from enum import Enum
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

    def __init__(self, size=64, model_resolution=2, max_zoom=4, nb_max_actions=100, replace_charlie=True, deep=False):
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
        self.dummy_charlie_model = None
        self.dummy_surface_model = None
        self.vision = np.zeros(7, dtype=int)
        self.max_distance = math.sqrt(self.size ** 2 + self.size ** 2)

        self.replace_charlie = replace_charlie
        self.deep = deep  
        self.states = np.arange(2 * 3 * self.max_zoom * self.max_move * self.max_move).reshape((2, 3, self.max_zoom,
                                                                                                self.max_move,
                                                                                                self.max_move))

        def model_probalities(i):
            """
            This function is vectorize over all the pieces on the subgrid. It gives the probability of seen charlie.
            :param i: the Piece that is analysed.
            :return: the changed value of i
            """
            if i is Piece.CHARLIE:
                # simulate a neural network, the more the agent zoom the more the probability of
                # seeing charlie(waldo) increase
                if not np.random.binomial(1, .95 / self.z):
                    i = Piece.WATER
            return i

        self.get_probabilities = np.vectorize(model_probalities)

    def place_charlie(self):
        """
        this method place change the charlie's position on the map.
        """
        while True:
            self.grid[self.charlie_y][self.charlie_x] = Piece.WATER
            x = random.randint(0, self.size - 1)
            y = random.randint(1, self.size - 1)
            if self.grid[y][x] is Piece.WATER and self.grid[y][x - 1] is Piece.GROUND:
                self.grid[y][x] = Piece.CHARLIE
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

        self.history = []
        self.action_dones = []
        self.nb_actions_taken = 0
        self.z = 1
        self.x = 0
        self.y = 0
        self.nb_mark = 0
        self.marked_correctly = False

        if self.replace_charlie:
            self.place_charlie()

        self.compute_sub_grid()
        self.fit_dummy_model()

        if self.deep:
            S = self.get_current_state_deep()
        else:
            S = self.get_current_state()

        return S

    def init_env(self):
        """
        This method is used to generate an environment.
        """

        def dilate():
            """
            This function create lake by dilation.
            """
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
        # Place 5 lake on the map
        for i in range(5):
            self.grid[i] = Piece.WATER

        np.random.shuffle(self.grid)

        self.grid = self.grid.reshape((self.size, self.size))
        # Make the lake grow.
        for _ in range(10):
            dilate()

        self.place_charlie()

    def compute_sub_grid(self):
        """
        Compute the sub grid at the agent position given the x, y and z axis.
        """
        window = self.model_resolution ** self.z
        self.sub_grid = self.grid[window * self.y:window + window * self.y, window * self.x:window + window * self.x]

    def fit_dummy_model(self):
        """
        give the state of the current sub_grid. (if it contains charlie or if the agent is on water or land=
        """
        proba = self.get_probabilities(self.sub_grid)
        self.dummy_charlie_model = 1 if np.count_nonzero(proba == Piece.CHARLIE) else 0
        if np.count_nonzero(proba == Piece.GROUND) and np.count_nonzero(proba == Piece.WATER):
            self.dummy_surface_model = 2
        elif np.count_nonzero(proba == Piece.WATER):
            self.dummy_surface_model = 1
        else:
            self.dummy_surface_model = 0

    def get_current_state(self):
        """
        This method give the current state has a number of all the possible state.
        :return: the current state
        """
        return self.states[self.dummy_charlie_model][self.dummy_surface_model][self.z - 1][self.x][self.y]

    def sub_grid_value(self, i):
        """
        This function is vectorize over all the pieces on the subgrid. It gives the probability of having
        a boat or a house.
        :param i: the Piece that is analysed.
        :return: the changed value of i
        """
        return i.value['label']

    def get_current_state_deep(self):
        """
        give the current state has an array but with the same information has the classic state.
        :return: the current state.
        """
        deep_vision = []
        deep_vision.append(self.dummy_charlie_model)
        deep_vision.append(self.dummy_surface_model)
        deep_vision.append(self.x)
        deep_vision.append(self.y)
        deep_vision.append(self.z)

        return np.array(deep_vision, dtype=float)

    def get_nb_state(self):
        """
        :return: the number of states
        """
        return self.states.size

    def get_distance_reward(self):
        """
        this method return the distance between the agent position and the charlie's position.
        :return: the euclidian distance.
        """
        pad = self.model_resolution << (self.z - 1)
        return math.sqrt((self.x * pad - self.charlie_x) ** 2 + (self.y * pad - self.charlie_y) ** 2)


    def take_action(self, action):
        """
        This method allow the agent to take an action over the environment.
        :param action: the number of the action that the agent has take.
        :return: the next state, the reward, if the state is terminal and a tips of which action the agent should have
        choose.
        """
        action = Action(action)

        # history is used to plot the trajectory of the agent
        self.history.append((self.x, self.y, self.z))
        self.action_dones.append(action)

        # before the move we must check if the agent should mark
        should_have_mark = self.z <= 1 and np.count_nonzero(self.sub_grid == Piece.CHARLIE)

        if action == Action.LEFT:
            self.x -= 0 if self.x <= 0 else 1
        elif action == Action.UP:
            self.y -= 0 if self.y <= 0 else 1
        elif action == Action.RIGHT:
            self.x += 0 if (self.x + 1) >= self.size / (self.model_resolution ** self.z) else 1
        elif action == Action.DOWN:
            self.y += 0 if (self.y + 1) >= self.size / (self.model_resolution ** self.z) else 1
        elif action == Action.ZOOM:
            if not self.z - 1 <= 0:
                self.z -= 1
                self.x = self.x << 1
                self.y = self.y << 1
        elif action == Action.DEZOOM:
            if not self.z + 1 >= self.max_zoom:
                self.x = int(self.x / self.model_resolution)
                self.y = int(self.y / self.model_resolution)
                self.z += 1

        self.compute_sub_grid()

        # if the agent has not mark but should have, the last action is not marked correctly.
        if not action == Action.MARK and should_have_mark:
            action = Action.MARK
        elif action == Action.MARK and should_have_mark:
            self.marked_correctly = True

        self.fit_dummy_model()
        self.nb_actions_taken += 1

        #reward = -1
        reward = - (self.get_distance_reward() / self.max_distance)

        is_terminal = self.nb_max_actions <= self.nb_actions_taken

        if action == Action.MARK:
            self.nb_mark += 1
            if should_have_mark:
                is_terminal = True
                reward = 100
            else:
                reward = -10

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
        for i in range(len(self.history)):
            x, y, z = self.history[i]
            a = self.action_dones[i]

            if a == Action.MARK:
                color = [0, 0, 1]
            else:
                color = [0, 1, 0]

            window = (self.model_resolution ** z) * 10
            mm[window * y:window + window * y
              ,window * x:window + window * x] = mm[window * y:window + window * y
                                                         ,window * x:window + window * x] >> color

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
"""
This file contains the implementation of the grid world environment.
"""
from enum import Enum

import numpy as np


class Piece(Enum):
    """
    this enum class simplify the representation of the different
    pieces on the board.
    """
    GOAL = {"code": 'X', "reward": 10, "is_terminal": True}
    PIT = {"code": 'O', "reward": -10, "is_terminal": False}
    SOIL = {"code": ' ', "reward": 0, "is_terminal": False}


class PieceRender(Enum):
    """
    this enum class represent the visualisation of the board.
    """
    GOAL = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    AGENT = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    PIT = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
           [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    EMPTY = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


class Action(Enum):
    """
    This enum class represent the different action an agent can
    take in the gridworld.
    """
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    TERMINAL = -1


class Board:
    """
    this class implement the grid world problem as a frozen lake problem.
    """
    def __init__(self, size=5, nb_trap=2):

        if nb_trap >= size * size:
            raise Exception("number of trap cannot be greater or equal to the size^2")

        self.states = None
        self.size = size
        self.nb_action = 4
        self.nb_trap = nb_trap
        self.agent = (0, 0)

        self.init_board()

    def init_board(self):
        """
        this method is used to generate a board with the number of pit requested by
        the used. It could be minus one if a pit is generated at the
        first or last position.
        """
        self.states = np.full((self.size * self.size), Piece.SOIL)
        for i in range(self.nb_trap):
            self.states[i] = Piece.PIT

        np.random.shuffle(self.states)

        self.states[0] = Piece.SOIL
        self.states[self.size * self.size - 1] = Piece.GOAL

        if not self.bfs():
            self.init_board()

    def bfs(self):
        """
        this method implement a bfs algorithm to check if there is a
        possible path between the agent initial position and the goal position.
        :return: True if there is a path False otherwise.
        """
        mark_map = np.zeros((self.size, self.size))

        # directions
        Dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # queue
        q = []
        mark_map[self.agent[0]][self.agent[1]] = 1

        # insert the player position
        q.append(self.agent)

        # until queue is empty
        while len(q) > 0:
            p = q[0]
            q.pop(0)

            # destination is reached.
            if self.states[p[0] * self.size + p[1]].value["code"] == 'X':
                return True

            # mark as visited
            mark_map[p[0]][p[1]] = 1

            # check all four directions
            for i in range(4):

                # using the direction array
                a = p[0] + Dir[i][0]
                b = p[1] + Dir[i][1]

                # not blocked and valid
                if 0 <= a < self.size and 0 <= b < self.size and mark_map[a][b] != 1 and \
                        self.states[a * self.size + b].value['code'] != 'O':
                    q.append((a, b))

        return False

    def move_agent(self, action):
        """
        this method allow the agent to move from one state to another.
        It also checks if the agent is in a border in with case it cannot go in this direction.
        :param action: the action the agent has taken.
        :return: the new agent position.
        """
        # LEFT
        if action == Action.LEFT and self.agent[1] > 0:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (0, -1)))
        # UP
        elif action == Action.UP and self.agent[0] > 0:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (-1, 0)))
        # RIGHT
        elif action == Action.RIGHT and self.agent[1] < self.size - 1:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (0, 1)))
        # DOWN
        elif action == Action.DOWN and self.agent[0] < self.size - 1:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (1, 0)))

        return self.agent[0] * self.size + self.agent[1]

    def get_next_state(self, state, action):
        """
        this method call move agent. It is more formal in RL term. it takes also the
        state in which the agent should be.
        :param state: the state in which the agent should be.
        :param action: the action the agent has taken.
        :return: the new agent position a.k.a. the next state.
        """
        # Convert 1D state to 2D
        self.agent = (int(state / self.size), state % self.size)

        return self.move_agent(action)

    def get_reward(self, state, action, next_state):
        """
        this method allow is used to get the current reward of the new state.
        :param state: the old state
        :param action: the action that has been taken in the old state.
        :param next_state: the current new state.
        :return: the reward the agent has reserved.
        """
        # agent have a malus everytime they take a step to force the algorithm to
        # converge to a solution with the minimal number of steps
        malus = -1
        # if the agent take an action go in a wall it take another malus
        if state == next_state:
            malus = -2

        return self.states[next_state].value["reward"] + malus

    def get_probability(self, state, action, new_state, reward):
        """
        This method give the probability of geting a reward r in a state s'
        according a state s and an action a. This method is more formal than
        useful because in this grid world the probability are always 1.
        :param state: old state.
        :param action: action that has been taken.
        :param new_state: new state the agent is in.
        :param reward: the got reward.
        :return: the probability.
        """
        return 1.

    def render_board(self):
        """
        this method allow to get a string representation of the board.
        for command line usage.
        :return: the string representation of the board
        """
        visual = ""
        for i in range(self.size):
            visual += '__'
        visual += '_\n'
        for i in range(self.size):
            visual += '|'
            for j in range(self.size):
                if self.agent == (i, j):
                    visual += 'A|'
                else:
                    visual += self.states[i * self.size + j].value["code"] + '|'

            visual += '\n'
        for i in range(self.size):
            visual += '__'
        visual += '_'
        return visual

    def render_board_img(self, agent_state):
        """
        this method allow to get a image representation of the board in
        the form of a numpy array.
        :param agent_state: the current position of the agent
        :return: a numpy array representing the board
        """
        visual = np.ones((self.size * 10, self.size * 10, 1), dtype=np.uint8)

        for i in range(self.size):
            for j in range(self.size):
                render_case = PieceRender.EMPTY.value
                icon = 0.

                if (i + j) % 2:
                    render_case = np.logical_not(render_case)

                if self.states[i * self.size + j].value["code"] == 'X':
                    icon = PieceRender.GOAL.value
                elif self.states[i * self.size + j].value["code"] == 'O':
                    icon = PieceRender.PIT.value

                if agent_state == (i * self.size + j):
                    icon = PieceRender.AGENT.value

                render_case = np.logical_xor(icon, render_case).reshape((10, 10, 1))

                visual[(i * 10):((i + 1) * 10), (j * 10):((j + 1) * 10)] = render_case

        visual *= 255

        return visual

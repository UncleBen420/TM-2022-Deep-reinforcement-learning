import numpy as np
from enum import Enum


class Piece(Enum):
    GOAL = {"code": 'X', "reward": 10, "is_terminal": 1}
    PIT = {"code": 'O', "reward": -10, "is_terminal": 0}
    SOIL = {"code": ' ', "reward": 0, "is_terminal": 0}

class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    TERMINAL = -1

class Board:
    def __init__(self, size=5, nb_trap=2):
        self.states = None
        self.size = size
        self.nb_action = 4
        self.nb_trap = nb_trap
        self.agent = (0, 0)

        self.init_board()

    def init_board(self):
        self.states = np.full((self.size * self.size), Piece.SOIL)
        for i in range(self.nb_trap):
            self.states[i] = Piece.PIT

        np.random.shuffle(self.states)

        self.states[0] = Piece.SOIL
        self.states[self.size * self.size - 1] = Piece.GOAL

        if not self.bfs():
            self.init_board()

    def bfs(self):
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
        # LEFT
        if action == Action.LEFT and self.agent[1] > 0:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (0, -1)))
        # UP
        elif action == Action.UP and self.agent[0] > 0:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (-1, 1)))
        # RIGHT
        elif action == Action.RIGHT and self.agent[1] < self.size - 1:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (0, 1)))
        # DOWN
        elif action == Action.DOWN and self.agent[0] < self.size - 1:
            self.agent = tuple(map(lambda i, j: i + j, self.agent, (1, 0)))

        return self.agent[0] * self.size + self.agent[1]

    def get_next_state(self, state, action):

        # Convert 1D state to 2D
        self.agent = (int(state / self.size), state % self.size)

        return self.move_agent(action)

    def get_reward(self, state, action, next_state):
        malus = -1
        if state == next_state:
            malus = -2

        return self.states[self.agent[0] * self.size + self.agent[1]].value["reward"] + malus

    def get_probability(self, state, action, new_state, reward):
        return 1.

    def render_board(self):
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

    def render_policy(self, policy):

        visual = ""

        for i in range(self.size):
            visual += '|'
            for j in range(self.size):

                action = policy[i * self.size + j]

                actions_possible = ''
                # LEFT
                if action == Action.LEFT.value:
                    actions_possible = '<'

                # UP
                if action == Action.UP.value:
                    actions_possible = '^'

                # RIGHT
                if action == Action.RIGHT.value:
                    actions_possible = '>'

                # DOWN
                if action == Action.DOWN.value:
                    actions_possible = 'v'

                # TERMINAL
                if action == Action.TERMINAL.value:
                    actions_possible = '*'

                visual += actions_possible + '|'
            visual += '\n'

        return visual
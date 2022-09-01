"""
unit test for the bandit environment file
"""

import pytest

from Environment.GridWorld import Board, Piece, Action


class TestGridWorld:
    """test class"""

    def test_init_board(self):
        with pytest.raises(Exception) as e:
            board = Board(size=2, nb_trap=4)
            assert e.message == "number of trap cannot be greater or equal to the size"

        board = Board()
        assert board.states.size == 25
        assert board.states[0].value == {"code": ' ', "reward": 0, "is_terminal": 0}
        assert board.states[24].value == {"code": 'X', "reward": 10, "is_terminal": 1}
        assert board.agent == (0, 0)

    def test_bfs(self):
        board_test_success = [Piece.SOIL, Piece.SOIL, Piece.SOIL,
                              Piece.PIT, Piece.PIT, Piece.SOIL,
                              Piece.SOIL, Piece.GOAL, Piece.SOIL]

        board_test_fail = [Piece.SOIL, Piece.PIT, Piece.SOIL,
                           Piece.PIT, Piece.PIT, Piece.SOIL,
                           Piece.SOIL, Piece.GOAL, Piece.SOIL]

        board = Board()
        board.size = 3
        board.states = board_test_success
        assert board.bfs()

        board.states = board_test_fail
        assert not board.bfs()

    def test_move_agent(self):
        board = Board()
        old_position = board.agent
        board.move_agent(Action.LEFT)
        assert old_position == board.agent
        board.move_agent(Action.UP)
        assert old_position == board.agent

        board.move_agent(Action.RIGHT)
        assert old_position != board.agent
        board.agent = old_position
        board.move_agent(Action.DOWN)
        assert old_position != board.agent

        board.agent = (4, 4)

        old_position = board.agent
        board.move_agent(Action.LEFT)
        assert old_position != board.agent
        board.move_agent(Action.UP)
        assert old_position != board.agent

        board.agent = (4, 4)

        board.move_agent(Action.RIGHT)
        assert old_position == board.agent
        board.agent = old_position
        board.move_agent(Action.DOWN)
        assert old_position == board.agent

        board.agent = (1, 1)

        board.move_agent(Action.LEFT)
        assert board.agent == (1, 0)
        board.move_agent(Action.UP)
        assert board.agent == (0, 0)

        board.move_agent(Action.RIGHT)
        assert board.agent == (0, 1)
        board.move_agent(Action.DOWN)
        assert board.agent == (1, 1)

    def test_get_next_state(self):
        board = Board(size=3)
        assert board.get_next_state(4, Action.LEFT) == 3
        assert board.get_next_state(4, Action.RIGHT) == 5
        assert board.get_next_state(4, Action.UP) == 1
        assert board.get_next_state(4, Action.DOWN) == 7

    def test_get_reward(self):
        board_grid = [Piece.SOIL, Piece.SOIL, Piece.SOIL,
                      Piece.PIT, Piece.PIT, Piece.SOIL,
                      Piece.SOIL, Piece.GOAL, Piece.SOIL]

        board = Board(size=3)
        board.states = board_grid
        assert board.get_reward(0, Action.RIGHT, 1) == -1
        assert board.get_reward(0, Action.DOWN, 3) == -11
        assert board.get_reward(6, Action.RIGHT, 7) == 9
        assert board.get_reward(6, Action.LEFT, 6) == -2
        assert board.get_reward(3, Action.LEFT, 3) == -12

    def test_get_reward_with_agent(self):
        board_grid = [Piece.SOIL, Piece.SOIL, Piece.SOIL,
                      Piece.PIT, Piece.PIT, Piece.SOIL,
                      Piece.SOIL, Piece.GOAL, Piece.SOIL]

        board = Board(size=3)
        board.states = board_grid
        assert board.get_reward_with_agent(Action.RIGHT) == -1
        assert board.get_reward_with_agent(Action.DOWN) == -11
        assert board.get_reward_with_agent(Action.LEFT) == -11
        assert board.get_reward_with_agent(Action.LEFT) == -12
        assert board.get_reward_with_agent(Action.DOWN) == -1
        assert board.get_reward_with_agent(Action.DOWN) == -2
        assert board.get_reward_with_agent(Action.RIGHT) == 9

    def test_render_board(self):
        board_grid = [Piece.SOIL, Piece.SOIL, Piece.SOIL,
                      Piece.PIT, Piece.PIT, Piece.SOIL,
                      Piece.SOIL, Piece.GOAL, Piece.SOIL]

        render = "_______\n|A| | |\n|O|O| |\n| |X| |\n_______"

        board = Board(size=3)
        board.states = board_grid
        assert board.render_board() == render

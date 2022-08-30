"""
unit test for the bandit environment file
"""
import numpy as np

from Environment.GridWorld import Board, Piece, Action


class TestGridWorld:
    """test class"""

    def test_init_board(self):
        board = Board()

        assert board.states.size == 25
        assert board.states[0].value == {"code": ' ', "reward": 0, "is_terminal": 0}
        assert board.states[24].value == {"code": 'X', "reward": 10, "is_terminal": 1}
        assert board.agent == (0,0)

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

        board.agent = (4,4)

        old_position = board.agent
        board.move_agent(Action.LEFT)
        assert old_position != board.agent
        board.move_agent(Action.UP)
        assert old_position != board.agent

#        board.move_agent(Action.RIGHT)
#        assert old_position == board.agent
#        board.agent = old_position
#        board.move_agent(Action.DOWN)
#        assert old_position == board.agent




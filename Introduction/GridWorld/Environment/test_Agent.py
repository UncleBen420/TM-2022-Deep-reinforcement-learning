"""
unit test for the bandit environment file
"""
from Environment import Agent
from Environment.GridWorld import Board


class TestGridWorld:
    """test class"""

    def test_render_policy(self):
        policy_test = [1, 2, 3, 1, 2, 3, 0, -1, 10]
        render = "|^|>|v|\n|^|>|v|\n|<|*|!|\n"
        board = Board(size=3)
        assert Agent.render_policy(board, policy_test) == render

    def test_init_policy(self):

        policy_test_1 = [2,2,3,
                         3,0,0,
                         2,2,-1]

        policy_test_2 = [2, 2, 2, 3,
                         3, 0, 0, 0,
                         2, 2, 2, 3,
                         2, 2, 2, -1]

        board = Board(size=3)
        assert (Agent.init_policy(board) == policy_test_1).all()

        board = Board(size=4)
        assert (Agent.init_policy(board) == policy_test_2).all()
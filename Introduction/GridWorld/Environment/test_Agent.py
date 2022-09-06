"""
unit test for the bandit environment file
"""
import numpy as np

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
        policy_test_1 = [2, 2, 3,
                         3, 0, 0,
                         2, 2, -1]

        policy_test_2 = [2, 2, 2, 3,
                         3, 0, 0, 0,
                         2, 2, 2, 3,
                         2, 2, 2, -1]

        board = Board(size=3)
        assert (Agent.init_policy(board) == policy_test_1).all()

        board = Board(size=4)
        assert (Agent.init_policy(board) == policy_test_2).all()

    def test_e_greedy(self):
        """test the e-greedy function. for testing purposes, the random
        function are mocked to obtain expected behaviour."""

        random_numbers = [0, 1]
        random_numbers2 = [1]

        temp = [Agent.np.random.binomial, Agent.random.randrange]

        Agent.np.random.binomial = lambda n, m: random_numbers.pop(0)
        Agent.random.randrange = lambda n: random_numbers2.pop(0)
        Action = [0., 0., 1.]

        assert Agent.e_greedy(Action, 0.1) == 2
        assert Agent.e_greedy(Action, 0.1) == 1

        Agent.np.random.binomial = temp[0]
        Agent.random.randrange = temp[1]

    def test_get_e_greedy_prob(self):

        Action = [0., 0., 0., 0.1]
        np.testing.assert_array_almost_equal(Agent.get_e_greedy_prob(Action, 0.1), [0.025, 0.025, 0.025, 0.925])
        np.testing.assert_array_almost_equal(Agent.get_e_greedy_prob(Action, 0.5), [0.125, 0.125, 0.125, 0.625])

        Action = [0., 0., 0.1, 0.1]
        np.testing.assert_array_almost_equal(Agent.get_e_greedy_prob(Action, 0.1), [0.025, 0.025, 0.475, 0.475])
        np.testing.assert_array_almost_equal(Agent.get_e_greedy_prob(Action, 0.5), [0.125, 0.125, 0.375, 0.375])

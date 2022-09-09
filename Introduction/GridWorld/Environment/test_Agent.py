"""
unit test for the agent helper file
"""
import numpy as np

from Environment import Agent
from Environment.Agent import DIRECTION
from Environment.GridWorld import Board


class TestGridWorld:
    """test class"""

    def test_render_policy(self):
        policy_test = [1, 2, 3, 1, 2, 3, 0, -1, 10]
        render = "|^|>|v|\n|^|>|v|\n|<|*|!|\n"
        board = Board(size=3)
        assert Agent.render_policy(board, policy_test) == render

    def test_init_policy(self):
        """
        this method is used to test the init policy function.
        It is done by comparing the initiated policy to a policy
        It should resemble.
        """
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
        """
        test the e-greedy function. for testing purposes, the random
        function are mocked to obtain expected behaviour.
        """

        class DummyAgent:
            pass

        da = DummyAgent()
        da.Q = [[0., 0., 1.], [1., 0., 0.]]

        eg = Agent.E_Greedy(0.1)
        eg.set_agent(da)

        random_numbers = [0, 1]
        random_numbers2 = [1]

        temp = [Agent.np.random.binomial, Agent.random.randrange]

        Agent.np.random.binomial = lambda n, m: random_numbers.pop(0)
        Agent.random.randrange = lambda n: random_numbers2.pop(0)

        assert eg.chose_action(0) == 2
        assert eg.chose_action(0) == 1

        Agent.np.random.binomial = temp[0]
        Agent.random.randrange = temp[1]

    def test_get_e_greedy_prob(self):
        """
        this method is used to test the probability for the e-greedy policy.
        """
        class DummyAgent:
            """
            dummy class mocking an agent
            """
            pass

        da = DummyAgent()
        da.Q = [[0., 0., 1.], [1., 0., 0.]]

        eg = Agent.E_Greedy(0.1)
        eg.set_agent(da)

        da.Q = [[0., 0., 0., 0.1]]
        np.testing.assert_array_almost_equal(eg.probability(0), [0.025, 0.025, 0.025, 0.925])
        eg.e = 0.5
        np.testing.assert_array_almost_equal(eg.probability(0), [0.125, 0.125, 0.125, 0.625])

        da.Q = [[0., 0., 0.1, 0.1]]
        eg.e = 0.1
        np.testing.assert_array_almost_equal(eg.probability(0), [0.025, 0.025, 0.475, 0.475])
        eg.e = 0.5
        np.testing.assert_array_almost_equal(eg.probability(0), [0.125, 0.125, 0.375, 0.375])

    def test_render_policy_img(self):
        """
        test if the function render_policy_img is actually returning the good
        representation of the policy as a numpy array.
        """
        policy = [0, 1, 2,
                  1, 2, 3,
                  2, -1, 4]

        render = np.append(DIRECTION.LEFT.value, np.logical_not(DIRECTION.UP.value), axis=1)
        render = np.append(render, DIRECTION.RIGHT.value, axis=1)
        render2 = np.append(np.logical_not(DIRECTION.UP.value), DIRECTION.RIGHT.value, axis=1)
        render2 = np.append(render2, np.logical_not(DIRECTION.DOWN.value), axis=1)
        render3 = np.append(DIRECTION.RIGHT.value, np.logical_not(DIRECTION.TERMINAL.value), axis=1)
        render3 = np.append(render3, DIRECTION.UNKNOWN.value, axis=1)
        render = np.append(render, render2, axis=0)
        render = np.append(render, render3, axis=0).reshape((30, 30, 1))
        render *= 255

        board = Board(size=3)
        np.testing.assert_array_equal(Agent.render_policy_img(board, policy), render)

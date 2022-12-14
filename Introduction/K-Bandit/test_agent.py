"""
unit test for the file containing the implementation of the agent
"""
import numpy as np
import agent

class TestAgent:
    """test class"""

    def test_incremental_mean(self):
        """test the incremental mean function. It try to see if changing
        the value of an action effect the other."""

        agt = agent.RLAgent(2, e=0.1, action_selection_function="egreedy")
        agt.nb_step[0] += 1
        assert agt.incremental_mean(10, 0) == 10
        agt.nb_step[1] += 1
        assert agt.incremental_mean(10, 1) == 10

        agt.expected_values[0] = 10
        agt.nb_step[0] += 1
        assert agt.incremental_mean(5, 0) == 7.5
        agt.expected_values[1] = 10
        agt.nb_step[1] += 1
        assert agt.incremental_mean(5, 1) == 7.5

    def test_update(self):
        """test the method update in the same manner that the
        test_incremental_mean do"""

        agt = agent.RLAgent(2, e=0.1, action_selection_function="egreedy")

        agt.update(10, 0)
        assert agt.expected_values[0] == 10
        assert agt.expected_values[1] == 0

        agt.update(20, 0)
        assert agt.expected_values[0] == 15
        assert agt.expected_values[1] == 0

        agt.update(10, 1)
        assert agt.expected_values[0] == 15
        assert agt.expected_values[1] == 10

        agt.update(-10, 1)
        assert agt.expected_values[0] == 15
        assert agt.expected_values[1] == 0

    def test_e_greedy(self):
        """test the e-greedy function. for testing purposes, the random
        function are mocked to obtain expected behaviour."""

        agt = agent.RLAgent(2, e=0.1, action_selection_function="egreedy")

        random_numbers = [0, 1]
        random_numbers2 = [1]

        temp = [agent.np.random.binomial, agent.random.randrange]

        agent.np.random.binomial = lambda n, m: random_numbers.pop(0)
        agent.random.randrange = lambda n: random_numbers2.pop(0)
        agt.expected_values[0] = 10
        assert agt.e_greedy() == 0
        assert agt.e_greedy() == 1

        agent.np.random.binomial = temp[0]
        agent.random.randrange = temp[1]

    def test_ucb(self):
        """test the ucb function. it try to see if indeed, ucb see the
        actions which have never been taken has maximizing"""

        agt = agent.RLAgent(2, c=5, action_selection_function="ucd")

        agt.expected_values[0] = 10
        agt.nb_step[0] += 1
        agt.nb_total_step += 1

        assert agt.ucb() == 1
        agt.expected_values[1] = 2
        agt.nb_step[1] += 1
        agt.nb_total_step += 1
        assert agt.ucb() == 0


'''
unit test for main file
'''
import numpy as np
import agent

class TestAgent:
    '''class'''

    def test_incremental_mean(self):
        '''test1'''

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
        '''test1'''

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
        '''test1'''

        agt = agent.RLAgent(2, e=0.1, action_selection_function="egreedy")

        random_numbers = [0, 1]
        random_numbers2 = [1]

        agent.np.random.binomial = lambda n, m: random_numbers.pop(0)
        agent.random.randrange = lambda n: random_numbers2.pop(0)
        agt.expected_values[0] = 10
        assert agt.e_greedy() == 0
        assert agt.e_greedy() == 1

    def test_ucb(self):
        '''test1'''

        agt = agent.RLAgent(2, c=5, action_selection_function="ucd")

        agt.expected_values[0] = 10
        agt.nb_step[0] += 1
        agt.nb_total_step += 1

        assert agt.ucb() == 1
        agt.expected_values[1] = 2
        agt.nb_step[1] += 1
        agt.nb_total_step += 1
        assert agt.ucb() == 0


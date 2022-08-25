'''
unit test for main file
'''
import numpy as np

import agent
from bandit import BanditsGame

class TestBanditGame:
    '''class'''

    def test_pull(self):
        '''test1'''

        agt = agent.RandomAgent(5)
        bandit = BanditsGame(5,100,agt,False)
        bandit.bandits = [1., 0., 0., 0., 0.]
        assert bandit.pull(0) == 1
        assert bandit.pull(1) == 0
        assert bandit.pull(2) == 0
        assert bandit.pull(3) == 0
        assert bandit.pull(4) == 0

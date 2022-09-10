"""
unit test for the bandit environment file
"""
import numpy as np

import agent
from bandit import BanditsGame


class TestBanditGame:
    """test class"""

    def test_pull(self):
        """test the pull method. It is done by setting the probability
        of the bandit to 1 or 0. this way we obtain expected behaviour."""

        agt = agent.RandomAgent(5)
        bandit = BanditsGame(5, 100, False, agt)
        bandit.bandits = [1., 0., 0., 0., 0.]
        assert bandit.pull(0) == 1
        assert bandit.pull(1) == 0
        assert bandit.pull(2) == 0
        assert bandit.pull(3) == 0
        assert bandit.pull(4) == 0

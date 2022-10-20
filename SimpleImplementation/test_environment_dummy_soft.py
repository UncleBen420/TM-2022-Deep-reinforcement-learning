"""
unit test for the file containing the implementation of the agent
"""
import numpy as np
import EnvironmentDummy


class TestEnv:
    """test class"""

    def test_get_state(self):
        de = EnvironmentDummy.DummyEnv()
        de.init_env()

        de.dummy_charlie_model = 0
        de.dummy_surface_model = 0
        de.x = 0
        de.y = 0
        de.z = 1
        assert de.get_current_state() == 0

        de.dummy_charlie_model = 0
        de.dummy_surface_model = 0
        de.x = 1
        de.y = 0
        de.z = 1
        assert de.get_current_state() == 32

        de.dummy_charlie_model = 1
        de.dummy_surface_model = 0
        de.x = 0
        de.y = 0
        de.z = 1
        assert de.get_current_state() == 15360

    def test_fit_dummy_model(self):
        temp = EnvironmentDummy.np.random.binomial

        EnvironmentDummy.np.random.binomial = lambda n, m: 1 if m > 0.5 else 0

        de = EnvironmentDummy.DummyEnv()
        de.init_env()
        de.sub_grid = np.array([[EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.CHARLIE],
                                [EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER]])
        de.fit_dummy_model()
        assert de.dummy_charlie_model == 0
        assert de.dummy_surface_model == 1

        de.z = 1
        de.fit_dummy_model()
        assert de.dummy_charlie_model == 1
        assert de.dummy_surface_model == 1

        de.sub_grid = np.array([[EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.GROUND],
                                [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.WATER]])

        de.fit_dummy_model()
        assert de.dummy_charlie_model == 0
        assert de.dummy_surface_model == 2

        de.sub_grid = np.array([[EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND],
                                [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND]])

        de.fit_dummy_model()
        assert de.dummy_charlie_model == 0
        assert de.dummy_surface_model == 0

        EnvironmentDummy.np.random.binomial = temp

    def test_compute_sub_grid(self):

        de = EnvironmentDummy.DummyEnv()
        de.grid = np.array([[EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER,
                             EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER],
                            [EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER,
                             EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER],
                            [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND,
                             EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND],
                            [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND,
                             EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND]])

        de.x = 0
        de.y = 0
        de.z = 1

        de.compute_sub_grid()
        np.testing.assert_array_equal(de.sub_grid, [[EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER],
                                                    [EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER]])

        de.y = 1
        de.compute_sub_grid()
        np.testing.assert_array_equal(de.sub_grid, [[EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND],
                                                    [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND]])

        de.y = 0
        de.z = 2
        de.compute_sub_grid()
        np.testing.assert_array_equal(de.sub_grid, [[EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER,
                                                     EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER],
                                                    [EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER,
                                                     EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.WATER],
                                                    [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND,
                                                     EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND],
                                                    [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND,
                                                     EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.GROUND]])

    def test_get_distance_reward(self):
        de = EnvironmentDummy.DummyEnv()

        de.charlie_x = 10
        de.charlie_y = 10
        de.x = 0
        de.y = 0
        de.z = 1
        assert round(de.get_distance_reward(), 3) == 14.142

        de.x = 0
        de.y = 0
        de.z = 2
        assert round(de.get_distance_reward(), 3) == 14.142

        de.x = 1
        de.y = 0
        de.z = 1
        assert round(de.get_distance_reward(), 3) == 12.806

        de.x = 2
        de.y = 0
        de.z = 1
        assert round(de.get_distance_reward(), 3) == 11.662


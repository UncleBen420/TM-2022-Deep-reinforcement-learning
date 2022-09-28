"""
unit test for the file containing the implementation of the agent
"""
import numpy as np
import EnvironmentDummySoft


class TestEnv:
    """test class"""

    def test_get_vision(self):
        de = EnvironmentDummySoft.DummyEnv()
        de.init_env()
        de.reload_env()

        de.history.append((0, 0, de.max_zoom - 1))
        de.z =de.max_zoom - 1
        de.x = 0
        de.y = 0
        de.get_vision()
        np.testing.assert_equal([3., 0., 3., 0., 0., 3., 1.], de.vision)
        de.x = 1
        de.y = 1
        de.get_vision()
        np.testing.assert_equal([0., 0., 0., 0., 0., 3., 0.], de.vision)
        de.x = 1
        de.y = 0
        de.get_vision()
        np.testing.assert_equal([1., 0., 3., 0., 0., 3., 0.], de.vision)
        de.x = 0
        de.y = 1
        de.get_vision()
        np.testing.assert_equal([3., 0., 1., 0., 0., 3., 0.], de.vision)
        de.x = 0
        de.y = 0
        de.z -= 1
        de.get_vision()
        np.testing.assert_equal([3., 0., 3., 0., 0., 1., 0.], de.vision)
        de.z = 1
        de.get_vision()
        np.testing.assert_equal([3., 0., 3., 0., 3., 0., 0.], de.vision)
        de.x = de.size / 2 - 1
        de.y = 0
        de.get_vision()
        np.testing.assert_equal([0., 3., 3., 0., 3., 0., 0.], de.vision)
        de.x = de.size / 2 - 1
        de.y = de.size / 2 - 1
        de.get_vision()
        np.testing.assert_equal([0., 3., 0., 3., 3., 0., 0.], de.vision)

    def test_get_state(self):
        de = EnvironmentDummySoft.DummyEnv()
        de.init_env()

        de.dummy_boat_model = 0
        de.dummy_surface_model = 0
        de.vision = [0, 0, 0, 0, 0, 0, 0]
        assert de.get_current_state() == 0

        de.dummy_boat_model = 0
        de.dummy_surface_model = 0
        de.vision = [0, 0, 0, 0, 0, 0, 1]
        assert de.get_current_state() == 1

        de.dummy_boat_model = 0
        de.dummy_surface_model = 1
        de.vision = [0, 0, 0, 0, 0, 0, 0]
        assert de.get_current_state() == 12288

    def test_fit_dummy_model(self):
        temp = EnvironmentDummySoft.np.random.binomial

        EnvironmentDummySoft.np.random.binomial = lambda n, m: 1 if m > 0.5 else 0

        de = EnvironmentDummySoft.DummyEnv()
        de.init_env()
        de.sub_grid = np.array([[EnvironmentDummySoft.Piece.WATER, EnvironmentDummySoft.Piece.BOAT],
                                [EnvironmentDummySoft.Piece.BOAT, EnvironmentDummySoft.Piece.WATER]])
        de.fit_dummy_model()
        assert de.dummy_boat_model == 0
        assert de.dummy_surface_model == 1

        de.z = 1
        de.fit_dummy_model()
        assert de.dummy_boat_model == 1
        assert de.dummy_surface_model == 1

        de.sub_grid = np.array([[EnvironmentDummySoft.Piece.HOUSE, EnvironmentDummySoft.Piece.GROUND],
                                [EnvironmentDummySoft.Piece.GROUND, EnvironmentDummySoft.Piece.HOUSE]])

        de.z = 3
        de.fit_dummy_model()
        assert de.dummy_boat_model == 0
        assert de.dummy_surface_model == 0

        de.z = 1
        de.fit_dummy_model()
        assert de.dummy_boat_model == 0
        assert de.dummy_surface_model == 0

        EnvironmentDummySoft.np.random.binomial = temp

    def test_get_reward(self):
        de = EnvironmentDummySoft.DummyEnv()
        de.init_env()
        de.history.append((0, 1, 2))
        de.sub_grid = np.array([[EnvironmentDummySoft.Piece.WATER, EnvironmentDummySoft.Piece.BOAT],
                                [EnvironmentDummySoft.Piece.BOAT, EnvironmentDummySoft.Piece.WATER]])

        assert de.get_reward(EnvironmentDummySoft.Action.DOWN) == -1

        de.history.append((0, 1, 2))
        assert de.get_reward(EnvironmentDummySoft.Action.DOWN) == -11

        de.history.append((0, 2, 2))
        de.marked.append((0, 0, 1))
        assert de.get_reward(EnvironmentDummySoft.Action.MARK) == 19

        de.sub_grid = np.array([[EnvironmentDummySoft.Piece.WATER, EnvironmentDummySoft.Piece.BOAT],
                                [EnvironmentDummySoft.Piece.BOAT, EnvironmentDummySoft.Piece.HOUSE]])
        assert de.get_reward(EnvironmentDummySoft.Action.MARK) == 9
        de.marked.append((0, 2, 1))
        de.marked.append((0, 0, 1))
        assert de.get_reward(EnvironmentDummySoft.Action.MARK) == -101
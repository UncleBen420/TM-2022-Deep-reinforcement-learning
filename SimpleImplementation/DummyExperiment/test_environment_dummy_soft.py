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

        np.testing.assert_equal([3., 0., 3., 0., 0., 3.], de.vision)
        de.move_factor = (1, 1)
        de.get_vision()
        np.testing.assert_equal([0., 0., 0., 0., 0., 3.], de.vision)
        de.move_factor = (1, 0)
        de.get_vision()
        np.testing.assert_equal([1., 0., 3., 0., 0., 3.], de.vision)
        de.move_factor = (0, 1)
        de.get_vision()
        np.testing.assert_equal([3., 0., 1., 0., 0., 3.], de.vision)
        de.move_factor = (0, 0)
        de.zoom_factor -= 1
        de.get_vision()
        np.testing.assert_equal([3., 0., 3., 0., 0., 1.], de.vision)
        de.zoom_factor = 1
        de.get_vision()
        np.testing.assert_equal([3., 0., 3., 0., 3., 0.], de.vision)
        de.move_factor = (de.size / 2 - 1, 0)
        de.get_vision()
        np.testing.assert_equal([0., 3., 3., 0., 3., 0.], de.vision)
        de.move_factor = (de.size / 2 - 1, de.size / 2 - 1)
        de.get_vision()
        np.testing.assert_equal([0., 3., 0., 3., 3., 0.], de.vision)

    def test_get_state(self):
        de = EnvironmentDummySoft.DummyEnv()
        de.init_env()

        de.dummy_boat_model = 0
        de.dummy_surface_model = 0
        de.vision = [0, 0, 0, 0, 0, 0]
        assert de.get_current_state() == 0

        de.dummy_boat_model = 0
        de.dummy_surface_model = 0
        de.vision = [0, 0, 0, 0, 0, 1]
        assert de.get_current_state() == 1

        de.dummy_boat_model = 0
        de.dummy_surface_model = 1
        de.vision = [0, 0, 0, 0, 0, 0]
        assert de.get_current_state() == 4096

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

        de.zoom_factor = 1
        de.fit_dummy_model()
        assert de.dummy_boat_model == 1
        assert de.dummy_surface_model == 1

        de.sub_grid = np.array([[EnvironmentDummySoft.Piece.HOUSE, EnvironmentDummySoft.Piece.GROUND],
                                [EnvironmentDummySoft.Piece.GROUND, EnvironmentDummySoft.Piece.HOUSE]])

        de.zoom_factor = 3
        de.fit_dummy_model()
        assert de.dummy_boat_model == 0
        assert de.dummy_surface_model == 0

        de.zoom_factor = 1
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

        assert de.get_reward(EnvironmentDummySoft.Action.DOWN, False) == -1
        de.zoom_factor = 1
        assert de.get_reward(EnvironmentDummySoft.Action.DOWN, False) == -3
        de.history.append((0, 1, 2))
        assert de.get_reward(EnvironmentDummySoft.Action.DOWN, False) == -13

        de.history.append((0, 2, 2))
        de.marked.append((0, 0, 1))
        assert de.get_reward(EnvironmentDummySoft.Action.MARK, False) == 17

        de.sub_grid = np.array([[EnvironmentDummySoft.Piece.WATER, EnvironmentDummySoft.Piece.BOAT],
                                [EnvironmentDummySoft.Piece.BOAT, EnvironmentDummySoft.Piece.HOUSE]])
        assert de.get_reward(EnvironmentDummySoft.Action.MARK, False) == 2
        de.marked.append((0, 2, 1))
        de.marked.append((0, 0, 1))
        assert de.get_reward(EnvironmentDummySoft.Action.MARK, False) == -103

        de.sub_grid = np.array([[EnvironmentDummySoft.Piece.WATER, EnvironmentDummySoft.Piece.BOAT],
                                [EnvironmentDummySoft.Piece.BOAT, EnvironmentDummySoft.Piece.WATER]])
        de.grid = np.array([[EnvironmentDummySoft.Piece.WATER, EnvironmentDummySoft.Piece.BOAT],
                            [EnvironmentDummySoft.Piece.BOAT, EnvironmentDummySoft.Piece.WATER]])
        de.marked_map = np.array([[False, True],
                                  [False, False]])
        de.zoom_factor = 3
        assert de.get_reward(EnvironmentDummySoft.Action.DOWN, True) == -11
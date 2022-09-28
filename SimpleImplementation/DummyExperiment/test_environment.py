"""
unit test for the file containing the implementation of the agent
"""
import numpy as np
import EnvironmentDummy


class TestEnv:
    """test class"""

    def test_get_vision(self):
        de = EnvironmentDummy.DummyEnv()
        de.init_env()
        de.reload_env()

        temp1 = de.fit_dummy_model_surface
        temp2 = de.compute_sub_grid
        de.fit_dummy_model_surface = lambda x: True
        de.compute_sub_grid = lambda x, z: 1
        de.get_vision()

        np.testing.assert_equal([4., 2., 4., 2., 2., 4., 2.], de.vision)
        de.move_factor = (1, 1)
        de.get_vision()
        np.testing.assert_equal([2., 2., 2., 2., 2., 4., 2.], de.vision)
        de.move_factor = (1, 0)
        de.history.append((0, 0, de.zoom_factor))
        de.get_vision()
        np.testing.assert_equal([0., 2., 4., 2., 2., 4., 2.], de.vision)
        de.move_factor = (0, 1)
        de.get_vision()
        np.testing.assert_equal([4., 2., 0., 2., 2., 4., 2.], de.vision)
        de.move_factor = (0, 0)
        de.zoom_factor -= 1
        de.get_vision()
        np.testing.assert_equal([4., 2., 4., 2., 2., 0., 2.], de.vision)
        de.zoom_factor = 1
        de.get_vision()
        np.testing.assert_equal([4., 2., 4., 2., 4., 2., 2.], de.vision)
        de.move_factor = (de.size / 2 - 1, 0)
        de.get_vision()
        np.testing.assert_equal([2., 4., 4., 2., 4., 2., 2.], de.vision)
        de.move_factor = (de.size / 2 - 1, de.size / 2 - 1)
        de.get_vision()
        np.testing.assert_equal([2., 4., 2., 4., 4., 2., 2.], de.vision)

        de.fit_dummy_model_surface = temp1
        de.compute_sub_grid = temp2

    def test_get_state(self):
        de = EnvironmentDummy.DummyEnv()
        de.init_env()

        de.dummy_boat_model = 0
        de.vision = [0, 0, 0, 0, 0, 0, 0]
        assert de.get_current_state() == 0

        de.dummy_boat_model = 0
        de.vision = [0, 0, 0, 0, 0, 0, 1]
        assert de.get_current_state() == 1

        de.dummy_boat_model = 1
        de.vision = [0, 0, 0, 0, 0, 0, 0]
        assert de.get_current_state() == 62500

    def test_fit_dummy_model(self):
        temp = EnvironmentDummy.np.random.binomial

        EnvironmentDummy.np.random.binomial = lambda n, m: 1 if m > 0.5 else 0

        de = EnvironmentDummy.DummyEnv()
        de.init_env()
        sub_grid = np.array([[EnvironmentDummy.Piece.WATER, EnvironmentDummy.Piece.BOAT],
                             [EnvironmentDummy.Piece.BOAT, EnvironmentDummy.Piece.WATER]])
        de.fit_dummy_model_boat(sub_grid)
        assert de.dummy_boat_model == 0
        assert de.fit_dummy_model_surface(sub_grid) == 1

        de.zoom_factor = 1
        de.fit_dummy_model_boat(sub_grid)
        assert de.dummy_boat_model == 1
        assert de.fit_dummy_model_surface(sub_grid) == 1

        sub_grid = np.array([[EnvironmentDummy.Piece.HOUSE, EnvironmentDummy.Piece.GROUND],
                             [EnvironmentDummy.Piece.GROUND, EnvironmentDummy.Piece.HOUSE]])

        de.fit_dummy_model_boat(sub_grid)
        assert de.dummy_boat_model == 0
        assert de.fit_dummy_model_surface(sub_grid) == 0

        de.zoom_factor = 1
        de.fit_dummy_model_boat(sub_grid)
        assert de.dummy_boat_model == 0
        assert de.fit_dummy_model_surface(sub_grid) == 0

        EnvironmentDummy.np.random.binomial = temp

    def test_get_reward(self):
        de = EnvironmentDummy.DummyEnv()

        temp = de.compute_sub_grid
        de.compute_sub_grid = de.fit_dummy_model_surface = lambda x, y: np.array([[EnvironmentDummy.Piece.WATER,
                                                                                   EnvironmentDummy.Piece.BOAT],
                                                                                  [EnvironmentDummy.Piece.BOAT,
                                                                                   EnvironmentDummy.Piece.WATER]])

        de.init_env()
        de.history.append((0, 1, 2))

        assert de.get_reward(EnvironmentDummy.Action.DOWN, False) == -1
        de.zoom_factor = 1
        assert de.get_reward(EnvironmentDummy.Action.DOWN, False) == -1
        de.history.append((0, 1, 2))
        assert de.get_reward(EnvironmentDummy.Action.DOWN, False) == -11

        de.history.append((0, 2, 2))
        de.marked.append((0, 0, 1))
        assert de.get_reward(EnvironmentDummy.Action.MARK, False) == 19
        de.compute_sub_grid = de.fit_dummy_model_surface = lambda x, y: np.array([[EnvironmentDummy.Piece.WATER,
                                                                                   EnvironmentDummy.Piece.BOAT],
                                                                                  [EnvironmentDummy.Piece.BOAT,
                                                                                   EnvironmentDummy.Piece.HOUSE]])
        assert de.get_reward(EnvironmentDummy.Action.MARK, False) == 9
        de.marked.append((0, 2, 1))
        de.marked.append((0, 0, 1))
        assert de.get_reward(EnvironmentDummy.Action.MARK, False) == -101
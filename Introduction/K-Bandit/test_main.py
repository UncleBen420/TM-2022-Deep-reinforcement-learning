'''
unit test for main file
'''
import numpy as np

from main import incremental_mean

class TestClass:
    '''class'''

    def test_incremental_mean(self):
        '''test1'''

        self.test_array_1 = np.array([1., 1., 1., 1.])
        self.test_array_2 = np.array([0., 0., 0., 0.])
        self.test_array_3 = np.array([-1., 1., -1., 1.])

        np.testing.assert_array_almost_equal(incremental_mean(self.test_array_1),
                                             [1., 1., 1., 1.])
        np.testing.assert_array_almost_equal(incremental_mean(self.test_array_2),
                                             [0., 0., 0., 0.])
        np.testing.assert_array_almost_equal(incremental_mean(self.test_array_3),
                                             [-1., 0., -0.3333333333333333, 0.])

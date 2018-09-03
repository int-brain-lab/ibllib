import unittest
from alf.extractors import *
import numpy as np
import ibllib.io.raw_data_loaders as raw


class TestExtractors(unittest.TestCase):

    def setUp(self):
        self.test_session = "/home/nico/Projects/IBL/IBL-github/iblrig/\
pybpod_data/test_mouse/2018-07-31/1/"

    def test_feedbackType(self):
        ft = get_trials_feedbackType(self.test_session, save=False)
        self.assertTrue(isinstance(ft, np.ndarray))
        self.assertTrue(ft.dtype == np.int64)
        # with self.assertRaises(ValueError):
        #     get_trials_feedbackType(self.test_session, save=False)

    def test_contrastLR(self):
        lr = get_trials_contrastLR(self.test_session, save=False)
        self.assertTrue(len(lr) == 2)
        self.assertTrue(isinstance(lr, tuple))
        self.assertTrue(isinstance(lr[0], np.ndarray))
        self.assertTrue(isinstance(lr[1], np.ndarray))
        self.assertTrue(lr[0].dtype == np.float64)
        self.assertTrue(lr[1].dtype == np.float64)

    def test_choice(self):
        c = get_trials_choice(self.test_session, save=False)
        self.assertTrue(isinstance(c, np.ndarray))
        self.assertTrue(c.dtype == np.int64)

if __name__ == '__main__':
    unittest.main()

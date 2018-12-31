import unittest
import numpy as np
import logging

import alf.extractors as ex


class TestExtractors(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger('ibllib').setLevel(50)

    def test_interpolation(self):
        # straight test that it returns an usable function
        ta = np.array([0., 1., 2., 3., 4., 5.])
        tb = np.array([0., 1.1, 2.0, 2.9, 4., 5.])
        finterp = ex.time_interpolation(ta, tb)
        self.assertTrue(np.all(finterp(ta) == tb))
        # next test if sizes are not similar
        tc = np.array([0., 1.1, 2.0, 2.9, 4., 5., 6.])
        finterp = ex.time_interpolation(ta, tc)
        self.assertTrue(np.all(finterp(ta) == tb))

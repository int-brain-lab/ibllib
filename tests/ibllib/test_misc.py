import unittest
import logging
import time

import numpy as np

from ibllib.misc import version, print_progress, bincount2D


class TestPrintProgress(unittest.TestCase):

    def test_simple_print(self):
        print('waitbar')
        for p in range(10):
            time.sleep(0.05)
            print_progress(p, 9)


class TestVersionTags(unittest.TestCase):

    def test_compare_version_tags(self):
        self.assert_eq('3.2.3', '3.2.3')
        self.assert_eq('3.2.3', '3.2.03')
        self.assert_g('3.2.3', '3.2.1')
        self.assert_l('3.2.1', '3.2.3')
        self.assert_g('3.2.11', '3.2.2')
        self.assert_l('3.2.1', '3.2.11')

    def assert_eq(self, v0, v_):
        self.assertTrue(version.eq(v0, v_))
        self.assertTrue(version.ge(v0, v_))
        self.assertTrue(version.le(v0, v_))
        self.assertFalse(version.gt(v0, v_))
        self.assertFalse(version.lt(v0, v_))

    def assert_l(self, v0, v_):
        self.assertFalse(version.eq(v0, v_))
        self.assertFalse(version.ge(v0, v_))
        self.assertTrue(version.le(v0, v_))
        self.assertFalse(version.gt(v0, v_))
        self.assertTrue(version.lt(v0, v_))

    def assert_g(self, v0, v_):
        self.assertFalse(version.eq(v0, v_))
        self.assertTrue(version.ge(v0, v_))
        self.assertFalse(version.le(v0, v_))
        self.assertTrue(version.gt(v0, v_))
        self.assertFalse(version.lt(v0, v_))


class TestLoggingSystem(unittest.TestCase):

    def test_levels(self):
        # logger = logger_config('ibllib')
        logger = logging.getLogger('ibllib')
        logger.critical('IBLLIB This is a critical message')
        logger.error('IBLLIB This is an error message')
        logger.warning('IBLLIB This is a warning message')
        logger.info('IBLLIB This is an info message')
        logger.debug('IBLLIB This is a debug message')
        logger = logging.getLogger()
        logger.critical('ROOT This is a critical message')
        logger.error('ROOT This is an error message')
        logger.warning('ROOT This is a warning message')
        logger.info('ROOT This is an info message')
        logger.debug('ROOT This is a debug message')


class TestMisc(unittest.TestCase):

    def test_bincount_2d(self):
        # first test simple with indices
        x = np.array([0, 1, 1, 2, 2, 3, 3, 3])
        y = np.array([3, 2, 2, 1, 1, 0, 0, 0])
        r, xscale, yscale = bincount2D(x, y, xbin=1, ybin=1)
        r_ = np.zeros_like(r)
        # sometimes life would have been simpler in c:
        for ix, iy in zip(x, y):
            r_[iy, ix] += 1
        self.assertTrue(np.all(np.equal(r_, r)))
        # test with negative values
        y = np.array([3, 2, 2, 1, 1, 0, 0, 0]) - 5
        r, xscale, yscale = bincount2D(x, y, xbin=1, ybin=1)
        self.assertTrue(np.all(np.equal(r_, r)))
        # test unequal bins
        r, xscale, yscale = bincount2D(x / 2, y / 2, xbin=1, ybin=2)
        r_ = np.zeros_like(r)
        for ix, iy in zip(np.floor(x / 2), np.floor((y / 2 + 2.5) / 2)):
            r_[int(iy), int(ix)] += 1
        self.assertTrue(np.all(r_ == r))
        # test with weights
        w = np.ones_like(x) * 2
        r, xscale, yscale = bincount2D(x / 2, y / 2, xbin=1, ybin=2, weights=w)
        self.assertTrue(np.all(r_ * 2 == r))
        # test aggregation instead of binning
        x = np.array([0, 1, 1, 2, 2, 4, 4, 4])
        y = np.array([4, 2, 2, 1, 1, 0, 0, 0])
        r, xscale, yscale = bincount2D(x, y)
        self.assertTrue(np.all(xscale == yscale) and np.all(xscale == np.array([0, 1, 2, 4])))


if __name__ == "__main__":
    unittest.main(exit=False)

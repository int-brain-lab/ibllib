import unittest

import numpy as np

from brainbox.misc import bincount2D


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

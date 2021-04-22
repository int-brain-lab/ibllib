import unittest

import brainbox.numerical as bnum
import numpy as np


class TestBetweeenSorted(unittest.TestCase):

    def test_between_sorted_single_time(self):
        # test single time, falling right on edges
        t = np.arange(100)
        bounds = [10, 25]
        ind = bnum.between_sorted(t, bounds)
        assert np.all(t[ind] == np.arange(int(np.ceil(bounds[0])), int(np.floor(bounds[1] + 1))))
        # test single time in between edges
        bounds = [10.4, 25.2]
        ind = bnum.between_sorted(t, bounds)
        assert np.all(t[ind] == np.arange(np.ceil(bounds[0]), np.floor(bounds[1]) + 1))
        # out of bounds
        ind = bnum.between_sorted(t, [-5, 800])
        assert np.all(ind)

    def test_between_sorted_multiple_times(self):
        t = np.arange(100)
        # non overlapping ranges
        bounds = np.array([[10.4, 25.2], [67.2, 86.4]])
        ind_ = np.logical_or(bnum.between_sorted(t, bounds[0]), bnum.between_sorted(t, bounds[1]))
        ind = bnum.between_sorted(t, bounds)
        assert np.all(ind == ind_)
        # overlapping ranges
        bounds = np.array([[10.4, 83.2], [67.2, 86.4]])
        ind_ = np.logical_or(bnum.between_sorted(t, bounds[0]), bnum.between_sorted(t, bounds[1]))
        ind = bnum.between_sorted(t, bounds)
        assert np.all(ind == ind_)
        # one range contains the other
        bounds = np.array([[10.4, 83.2], [34, 78]])
        ind_ = np.logical_or(bnum.between_sorted(t, bounds[0]), bnum.between_sorted(t, bounds[1]))
        ind = bnum.between_sorted(t, bounds)
        assert np.all(ind == ind_)
        # case when one range starts exactly where another stops
        bounds = np.array([[10.4, 83.2], [83.2, 84]])
        ind = bnum.between_sorted(t, bounds)
        ind_ = np.logical_or(bnum.between_sorted(t, bounds[0]), bnum.between_sorted(t, bounds[1]))
        assert np.all(ind == ind_)

    def test_between_sorted_out_of_range(selfs):
        # np searchsorted was returning out of range index when the start time
        # was greater than the max or the end time lower than the min
        bounds = np.array([[-10.4, -3], [10.4, 20], [34, 78]])
        array = np.arange(30)
        assert np.sum(bnum.between_sorted(array, bounds)) == 10


class TestIsmember(unittest.TestCase):

    def test_ismember2d(self):
        b = np.reshape([0, 0, 0, 1, 1, 1], [3, 2])
        locb = np.array([0, 1, 0, 2, 2, 1])
        lia = np.array([True, True, True, True, True, True, False, False])
        a = np.r_[b[locb, :], np.array([[2, 1], [1, 2]])]
        lia_, locb_ = bnum.ismember2d(a, b)
        assert np.all(lia == lia_) & np.all(locb == locb_)

    def test_ismember2d_uuids(self):
        nb = 20
        na = 500
        np.random.seed(42)
        a = np.random.randint(0, nb + 3, na)
        b = np.arange(nb)
        lia, locb = bnum.ismember(a, b)
        bb = np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max,
                               size=(nb, 2), dtype=np.int64)
        aa = np.zeros((na, 2), dtype=np.int64)
        aa[lia, :] = bb[locb, :]
        lia_, locb_ = bnum.ismember2d(aa, bb)
        assert np.all(lia == lia_) & np.all(locb == locb_)
        bb[:, 0] = 0
        aa[:, 0] = 0
        # if the first column is equal, the distinction is to be made on the second\
        assert np.unique(bb[:, 1]).size == nb
        lia_, locb_ = bnum.ismember2d(aa, bb)
        assert np.all(lia == lia_) & np.all(locb == locb_)

    def test_ismember(self):
        def _check_ismember(a, b, lia_, locb_):
            lia, locb = bnum.ismember(a, b)
            self.assertTrue(np.all(a[lia] == b[locb]))
            self.assertTrue(np.all(lia_ == lia))
            self.assertTrue(np.all(locb_ == locb))

        b = np.array([0, 1, 3, 4, 4])
        a = np.array([1, 4, 5, 4])
        lia_ = np.array([True, True, False, True])
        locb_ = np.array([1, 3, 3])
        _check_ismember(a, b, lia_, locb_)

        b = np.array([0, 4, 3, 1, 4])
        a = np.array([1, 4, 5, 4])
        lia_ = np.array([True, True, False, True])
        locb_ = np.array([3, 1, 1])
        _check_ismember(a, b, lia_, locb_)

        b = np.array([0, 1, 3, 4])
        a = np.array([1, 4, 5])
        lia_ = np.array([True, True, False])
        locb_ = np.array([1, 3])
        _check_ismember(a, b, lia_, locb_)

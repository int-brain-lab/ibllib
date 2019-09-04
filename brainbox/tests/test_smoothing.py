import pytest
import brainbox.singlecell.regression.smoothing as bsrs
import numpy as np


def test_times2frames():
    assert bsrs.times2frames(np.array([2.3, 2.45, 2.46, 2.7]), 0.1, 2) == np.array([2, 4, 4, 7])


def test_gaussian_window():
    assert bsrs.gaussian_window(9, sigma=2) == \
        np.array([0.02699548, 0.0647588, 0.12098536, 0.17603266, 0.19947114, 0.17603266,
                  0.12098536, 0.0647588, 0.02699548])


def test_frame_smoothing_1():
    with pytest.raises(ValueError):
        bsrs.frame_smoothing(np.array([2, 5, 6, 7]))


def test_frame_smoothing_2():
    with pytest.raises(ValueError):
        # FIXME:
        assert bsrs.frame_smoothing(np.array([2, 5, 6, 7]), window_type='garbage') == ValueError


def test_frame_smoothing_3():
    assert bsrs.frame_smoothing(frames=np.array([2, 5, 6, 7]), window=[0.1, 0.2, 0.5, 0.2, 0.1]) \
        == np.array([0.1, 0.2, 0.5, 0.3, 0.4, 0.8, 0.9, 0.8, 0.3, 0.1])


def test_smoothing_times():
    assert bsrs.smoothing_times(times=np.array([2.3, 2.45, 2.46, 2.7]), frame_length=0.1,
                                window=[0.1, 0.5, 0.5, 0.1]) == \
        np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0.1, 0.5, 0.7, 1.1, 1., 0.3, 0.5, 0.5, 0.1])

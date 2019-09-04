import cross_val
import numpy as np


def test_train_test_windows():

    np.random.seed(42)

    window = 3

    for _ in range(10):
        times = np.random.randint(100, size=10)
        train, test = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        train2, test2 = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        assert train.shape == (2, 8)
        assert test.shape == (2, 2)
        recombined = np.concatenate((train, test), axis=1)
        recombined2 = np.concatenate((train2, test2), axis=1)
        assert np.array_equal(np.sort(recombined), np.sort(recombined2))

    for _ in range(10):
        times = np.random.randint(100, size=10)
        train, _ = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        train2, _ = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        assert not np.array_equal(train, train2)

    window = (2, 4)

    for _ in range(10):
        times = np.random.randint(100, size=10)
        train, test = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        train2, test2 = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        assert train.shape == (2, 8)
        assert test.shape == (2, 2)
        recombined = np.concatenate((train, test), axis=1)
        recombined2 = np.concatenate((train2, test2), axis=1)
        assert np.array_equal(np.sort(recombined), np.sort(recombined2))

    for _ in range(10):
        times = np.random.randint(100, size=10)
        train, _ = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        train2, _ = cross_val.train_test_windows(test_size=0.2, times=times, window=window)
        assert not np.array_equal(train, train2)


def test_intervals():

    np.random.seed(42)

    window = 3
    for _ in range(10):
        times = np.sort(np.random.randint(100, size=10))
        indices = cross_val.intervals(times=times, window=window)
        assert indices.shape == (2, 10)
        assert np.all(indices[0] < indices[1])
        assert np.all(indices[1] - indices[0] == window * 2)

    window = (3, 2)
    for _ in range(10):
        times = np.sort(np.random.randint(100, size=10))
        indices = cross_val.intervals(times=times, window=window)
        assert indices.shape == (2, 10)
        assert np.all(indices[0] < indices[1])
        assert np.all(indices[1] - indices[0] == window[0] + window[1])

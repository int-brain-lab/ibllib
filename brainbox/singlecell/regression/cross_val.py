from sklearn.model_selection import train_test_split
import numpy as np


def train_test_windows(test_size, times, window):
    indices = intervals(times, window)
    train, test = train_test_split(range(times.size), shuffle=True, test_size=test_size)
    return indices[:, train], indices[:, test]


def intervals(times, window):
    ret = np.zeros((2, times.size))

    if isinstance(window, int):
        ret[0] = times - window
        ret[1] = times + window
    elif isinstance(window, tuple):
        ret[0] = times - window[0]
        ret[1] = times + window[1]
    else:
        raise TypeError('Window type not valid, use int or tuple')

    return ret


print(test_val_split(0.8, np.array([4, 7, 8, 11, 12, 19, 23, 50, 51]), 2))
print(test_val_split(0.8, np.array([4, 7, 8, 11, 12, 19, 23, 50, 51]), (3, 4)))

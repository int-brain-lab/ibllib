from sklearn.model_selection import train_test_split
import numpy as np


def train_test_windows(test_size, times, window):
    """
    Returns test and train split for index windows around times

    Parameters:
    -----------
        test_size : float
            between 0 and 1, which proportion of windows is to be used for testing
        times : numpy-array
            array of time points around which to pick a window of indices
        window : int or tuple
            determines window size around times, if int gives symmetric window, if tuple uses the
            first number as left window size, second as right window size

    Returns:
    --------
        indices : numpy-array of dim (2, n)
            two sets of numpy arrays, training and test, 2 numbers for window start index and end
            index
    """

    indices = intervals(times, window)
    train, test = train_test_split(range(times.size), shuffle=True, test_size=test_size)
    return indices[:, train], indices[:, test]


def intervals(times, window):
    """
    Returns test and train split for index windows around times

    Parameters:
    -----------
        test_size : float
            between 0 and 1, which proportion of windows is to be used for testing
        times : numpy-array
            array of time points around which to pick a window of indices
        window : int or tuple
            determines window size around times, if int gives symmetric window, if tuple uses the
            first number as left window size, second as right window size

    Returns:
    --------
        indices : numpy-array of dim (2, n)
            two sets of numpy arrays, training and test, 2 numbers for window start index and end
            index
    """
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

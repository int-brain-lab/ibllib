import numpy as np


def rms(x, axis=None):
    if not axis:
        axis = x.ndim - 1
    return np.sqrt(np.mean(x ** 2, axis=axis))

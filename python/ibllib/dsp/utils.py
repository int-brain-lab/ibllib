import numpy as np


def rms(x, axis=0):
    return np.sqrt(np.mean(x ** 2, axis=axis))

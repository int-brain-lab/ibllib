import numpy as np


def freduce(x):
    """
    Reduces a spectrum to positive frequencies only
    :param x: numpy.ndarray
    :return: numpy.ndarray
    """
    # support for multidimensional not done. If/when needed...
    siz = list(x.shape)
    siz[0] = int(np.floor(siz[0] / 2 + 1))
    return x[:siz[0]]


def fexpand(x, ns=1):
    """
    Reconstructs full spectrum from positive frequencies
    :param x: numpy.ndarray
    :return: numpy.ndarray
    """
    # support for multidimensional not done. If/when needed...
    siz = list(x.shape)
    dec = int(ns % 2)
    return np.concatenate((x, np.conj(x[1:siz[0] - dec])), axis=0)

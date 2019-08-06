'''
Single-cell functions.
'''

import numpy as np
from brainbox.population import xcorr


def acorr(spike_times, bin_size=None, window_size=None):
    """Compute the auto-correlogram of a neuron.

    Parameters
    ----------

    :param spike_times: Spike times in seconds.
    :type spike_times: array-like
    :param bin_size: Size of the bin, in seconds.
    :type bin_size: float
    :param window_size: Size of the window, in seconds.
    :type window_size: float

    Returns an `(winsize_samples,)` array with the auto-correlogram.

    """
    xc = xcorr(spike_times, np.zeros_like(spike_times), bin_size=bin_size, window_size=window_size)
    return xc[0, 0, :]

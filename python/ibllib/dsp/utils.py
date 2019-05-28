import numpy as np

from ibllib.misc import print_progress


def rms(x, axis=None):
    """
    Root mean square of array along axis

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    if not axis:
        axis = x.ndim - 1
    return np.sqrt(np.mean(x ** 2, axis=axis))


class WindowGenerator(object):
    """
    `wg = WindowGenerator(ns, nswin, overlap)`

    Provide sliding windows indices generator for signal processing applications.
    For straightforward spectrogram / periodogram implementation, prefer scipy methods !

    Example of implementations in test_dsp.py.
    """
    def __init__(self, ns, nswin, overlap):
        """
        :param ns: number of sample of the signal along the direction to be windowed
        :param nswin: number of samples of the window
        :return: dsp.WindowGenerator object:
        """
        self.ns = int(ns)
        self.nswin = int(nswin)
        self.overlap = int(overlap)
        self.nwin = int(np.ceil(float(ns - nswin) / float(nswin - overlap))) + 1
        self.iw = None

    @property
    def firstlast(self):
        """
        Generator that yields first and last index of windows

        :return: tuple of [first_index, last_index] of the window
        """
        self.iw = 0
        first = 0
        while True:
            last = first + self.nswin
            last = min(last, self.ns)
            yield (first, last)
            if last == self.ns:
                break
            first += self.nswin - self.overlap
            self.iw += 1

    @property
    def slice(self):
        """
        Generator that yields slices of windows

        :return: a slice of the window
        """
        for first, last in self.firstlast:
            yield slice(first, last)

    def slice_array(self, sig, axis=-1):
        """
        Provided an array or sliceable object, generator that yields
        slices corresponding to windows. Especially useful when working on memmpaps

        :param sig: array
        :param axis: (optional, -1) dimension along which to provide the slice
        :return: array slice Generator
        """
        for first, last in self.firstlast:
            yield np.take(sig, np.arange(first, last), axis=axis)

    def tscale(self, fs):
        """
        Returns the time scale associated with Window slicing (middle of window)
        :param fs: sampling frequency (Hz)
        :return: time axis scale
        """
        return np.array([(first + (last - first - 1) / 2) / fs for first, last in self.firstlast])

    def print_progress(self):
        """
        Prints progress using a terminal progress bar
        """
        print_progress(self.iw, self.nwin)

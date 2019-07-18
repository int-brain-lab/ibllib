'''
Set of functions for processing data from one form into another,
for example taking spike times and then binning them into non-overlapping
bins or convolving with a gaussian kernel.
'''
import numpy as np
from scipy import interpolate
from brainbox.core import Bunch, TimeSeries


def syncbin(timeseries, binwidth, offsets=None, interp=step):
    """Takes a tuple of timeseries bunches, aligns them to one another given an optional offset,
    and bins them into non-overlapping temporal bins of a given width.

    :param timeseries: A group of time series to perform alignment. Must have time stamps.
    :type timeseries: tuple of TimeSeries objects
    :param binwidth: Width of non-overlapping bins in which to put time series
    :type binwidth: float
    :param offsets: tuple of offsets for time stamps of each TimeSeries object, defaults to None
    :type offsets: tuple of floats, optional
    :return: TimeSeries object with each row representing binned and synchronized values of all
    input TimeSeries
    """
    # Adjust each timeseries by the associated offset if necessary then load into a list
    if offsets is not None:
        tstamps = [ts.times + os for ts, os in zip(timeseries, offsets)]
    else:
        tstamps = [ts.times for ts in timeseries]

    tbounds = np.array([np.amin(ts), np.amax(ts) for ts in tstamps])
    tmin, tmax = np.amin(tbounds[:, 0]), np.amax(tbounds[:, 1])
    return syncd
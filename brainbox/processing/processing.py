'''
Set of functions for processing data from one form into another,
for example taking spike times and then binning them into non-overlapping
bins or convolving with a gaussian kernel.
'''
import numpy as np
from scipy import interpolate
from brainbox.core import Bunch, TimeSeries


def sync(timeseries, dt, offsets=None, interp='zero'):
    """Takes a tuple of timeseries bunches, aligns them to one another given an optional offset,
    and bins them into non-overlapping temporal bins of a given width.

    :param timeseries: A group of time series to perform alignment. Must have time stamps.
    :type timeseries: tuple of TimeSeries objects
    :param dt: Separation of points which the output timeseries will be sampled at
    :type dt: float
    :param offsets: tuple of offsets for time stamps of each TimeSeries object, defaults to None
    :type offsets: tuple of floats, optional
    :param interp: Type of interpolation to use. Refer to scipy.interpolate.interp1d for possible
        values.
    :type interp: str
    :return: TimeSeries object with each row representing binned and synchronized values of all
        input TimeSeries
    """
    # Adjust each timeseries by the associated offset if necessary then load into a list
    if offsets is not None:
        tstamps = [ts.times + os for ts, os in zip(timeseries, offsets)]
    else:
        tstamps = [ts.times for ts in timeseries]

    # Get the min and max values for all timeseries combined after offsetting
    tbounds = np.array([(np.amin(ts), np.amax(ts)) for ts in tstamps])
    tmin, tmax = np.amin(tbounds[:, 0]), np.amax(tbounds[:, 1])
    # Add a corrective factor to ensure uniform time bins that cover all the data
    newt = np.arange(tmin, tmax + (dt - tmax % dt), dt)
    syncd = None
    '''This is a bunk line of code'''
    return syncd

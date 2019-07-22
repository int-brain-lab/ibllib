'''
Set of functions for processing data from one form into another,
for example taking spike times and then binning them into non-overlapping
bins or convolving with a gaussian kernel.
'''
import numpy as np
import pandas as pd
from scipy import interpolate
from brainbox import core


def sync(timeseries, dt, offsets=None, interp='zero', fillval=np.nan):
    """Takes a tuple of timeseries bunches, aligns them to one another given an optional offset,
    and synchronizes them into samples separated by a given delta_t. Uses scipy's interpolation
    library to perform interpolation. See scipy.interp1d for more information regarding interp and
    fillval parameters.

    :param timeseries: A group of time series to perform alignment. Must have time stamps.
    :type timeseries: tuple of TimeSeries objects
    :param dt: Separation of points which the output timeseries will be sampled at
    :type dt: float
    :param offsets: tuple of offsets for time stamps of each TimeSeries object, defaults to None
    :type offsets: tuple of floats, optional
    :param interp: Type of interpolation to use. Refer to scipy.interpolate.interp1d for possible
        values, defaults to np.nan
    :type interp: str
    :param fillval: Fill values to use when interpolating outside of range of data. See interp1d
        for possible values, defaults to '
    :return: TimeSeries object with each row representing synchronized values of all
        input TimeSeries
    """
    if isinstance(timeseries, core.TimeSeries):
        timeseries = [timeseries]
    # Adjust each timeseries by the associated offset if necessary then load into a list
    if offsets is not None:
        tstamps = [ts.times + os for ts, os in zip(timeseries, offsets)]
    else:
        tstamps = [ts.times for ts in timeseries]

    # If all input timeseries have column names, put them together for the output TS
    if all([ts.columns is not None for ts in timeseries]):
        colnames = []
        for ts in timeseries:
            colnames.extend(ts.columns)
    else:
        colnames = None
    # Get the min and max values for all timeseries combined after offsetting
    tbounds = np.array([(np.amin(ts), np.amax(ts)) for ts in tstamps])
    tmin, tmax = np.amin(tbounds[:, 0]), np.amax(tbounds[:, 1])
    # Add a corrective factor to ensure uniform time bins that cover all the data
    newt = np.arange(tmin, tmax + (dt - tmax % dt), dt)
    tsinterps = [interpolate.interp1d(ts.times, ts.values, kind=interp,
                                      fill_value=fillval, axis=0) for ts in timeseries]
    syncd = core.TimeSeries(newt, np.hstack([tsi(newt) for tsi in tsinterps]), columns=colnames)
    return syncd


def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None):
    """
    Computes a 2D histogram by aggregating values in a 2D array.

    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin: bin size along 2nd dimension (set to 0 to aggregate according to unique values)
    :param ybin: bin size along 1st dimension (set to 0 to aggregate according to unique values)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    # if no bounds provided, use min/max of vectors
    if not xlim:
        xlim = [np.min(x), np.max(x)]
    if not ylim:
        ylim = [np.min(y), np.max(y)]

    # create the indices on which to aggregate: binning is different that aggregating
    if xbin:
        xscale = np.arange(xlim[0], xlim[1] + xbin / 2, xbin)
        xind = (np.floor((x - xlim[0]) / xbin)).astype(np.int64)
    else:  # if bin size = 0 , aggregate over unique values
        xscale, xind = np.unique(x, return_inverse=True)
    if ybin:
        yscale = np.arange(ylim[0], ylim[1] + ybin / 2, ybin)
        yind = (np.floor((y - ylim[0]) / ybin)).astype(np.int64)
    else:  # if bin size = 0 , aggregate over unique values
        yscale, yind = np.unique(y, return_inverse=True)

    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)
    return r, xscale, yscale


def bin_spikes(spikes, binsize, interval_indices=False):

    if type(spikes) is not core.TimeSeries:
        raise TypeError('Input spikes need to be in TimeSeries object format')

    if not hasattr(spikes, 'clusters'):
        raise AttributeError('Input spikes need to have a clusters attribute. Make sure you set\
                              columns=(\'clusters\',)) when constructing spikes.')

    rates, tbins, clusters = bincount2D(spikes.times, spikes.clusters, binsize)
    if interval_indices:
        intervals = pd.interval_range(tbins[0], tbins[-1], freq=binsize, closed='left')
        return core.TimeSeries(times=intervals, values=rates.T[:-1], columns=clusters)
    else:
        return core.TimeSeries(times=tbins, values=rates.T, columns=clusters)

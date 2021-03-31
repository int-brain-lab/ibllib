'''
Processes data from one form into another, e.g. taking spike times and binning them into
non-overlapping bins and convolving spike times with a gaussian kernel.
'''

import numpy as np
import pandas as pd
from scipy import interpolate
from brainbox import core


def sync(dt, times=None, values=None, timeseries=None, offsets=None, interp='zero',
         fillval=np.nan):
    """
    Function for resampling a single or multiple time series to a single, evenly-spaced, delta t
    between observations. Uses interpolation to find values.

    Can be used on raw numpy arrays of timestamps and values using the 'times' and 'values' kwargs
    and/or on brainbox.core.TimeSeries objects passed to the 'timeseries' kwarg. If passing both
    TimeSeries objects and numpy arrays, the offsets passed should be for the TS objects first and
    then the numpy arrays.

    Uses scipy's interpolation library to perform interpolation.
    See scipy.interp1d for more information regarding interp and fillval parameters.

    :param dt: Separation of points which the output timeseries will be sampled at
    :type dt: float
    :param timeseries: A group of time series to perform alignment or a single time series.
        Must have time stamps.
    :type timeseries: tuple of TimeSeries objects, or a single TimeSeries object.
    :param times: time stamps for the observations in 'values']
    :type times: np.ndarray or list of np.ndarrays
    :param values: observations corresponding to the timestamps in 'times'
    :type values: np.ndarray or list of np.ndarrays
    :param offsets: tuple of offsets for time stamps of each time series. Offsets for passed
        TimeSeries objects first, then offsets for passed numpy arrays. defaults to None
    :type offsets: tuple of floats, optional
    :param interp: Type of interpolation to use. Refer to scipy.interpolate.interp1d for possible
        values, defaults to np.nan
    :type interp: str
    :param fillval: Fill values to use when interpolating outside of range of data. See interp1d
        for possible values, defaults to np.nan
    :return: TimeSeries object with each row representing synchronized values of all
        input TimeSeries. Will carry column names from input time series if all of them have column
        names.
    """
    #########################################
    # Checks on inputs and input processing #
    #########################################

    # Initialize a list to contain times/values pairs if no TS objs are passed
    if timeseries is None:
        timeseries = []
    # If a single time series is passed for resampling, wrap it in an iterable
    elif isinstance(timeseries, core.TimeSeries):
        timeseries = [timeseries]
    # Yell at the user if they try to pass stuff to timeseries that isn't a TimeSeries object
    elif not all([isinstance(ts, core.TimeSeries) for ts in timeseries]):
        raise TypeError('All elements of \'timeseries\' argument must be brainbox.core.TimeSeries '
                        'objects. Please uses \'times\' and \'values\' for np.ndarray args.')
    # Check that if something is passed to times or values, there is a corresponding equal-length
    # argument for the other element.
    if (times is not None) or (values is not None):
        if len(times) != len(values):
            raise ValueError('\'times\' and \'values\' must have the same number of elements.')
        if type(times[0]) is np.ndarray:
            if not all([t.shape == v.shape for t, v in zip(times, values)]):
                raise ValueError('All arrays in \'times\' must match the shape of the'
                                 ' corresponding entry in \'values\'.')
            # If all checks are passed, convert all times and values args into TimeSeries objects
            timeseries.extend([core.TimeSeries(t, v) for t, v in zip(times, values)])
        else:
            # If times and values are only numpy arrays and lists of arrays, pair them and add
            timeseries.append(core.TimeSeries(times, values))

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

    #################
    # Main function #
    #################

    # Get the min and max values for all timeseries combined after offsetting
    tbounds = np.array([(np.amin(ts), np.amax(ts)) for ts in tstamps])
    if not np.all(np.isfinite(tbounds)):
        # If there is a np.inf or np.nan in the time stamps for any of the timeseries this will
        # break any further code so we check for all finite values and throw an informative error.
        raise ValueError('NaN or inf encountered in passed timeseries.\
                          Please either drop or fill these values.')
    tmin, tmax = np.amin(tbounds[:, 0]), np.amax(tbounds[:, 1])
    if fillval == 'extrapolate':
        # If extrapolation is enabled we can ensure we have a full coverage of the data by
        # extending the t max to be an whole integer multiple of dt above tmin.
        # The 0.01% fudge factor is to account for floating point arithmetic errors.
        newt = np.arange(tmin, tmax + 1.0001 * (dt - (tmax - tmin) % dt), dt)
    else:
        newt = np.arange(tmin, tmax, dt)
    tsinterps = [interpolate.interp1d(ts.times, ts.values, kind=interp, fill_value=fillval, axis=0)
                 for ts in timeseries]
    syncd = core.TimeSeries(newt, np.hstack([tsi(newt) for tsi in tsinterps]), columns=colnames)
    return syncd


def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None):
    """
    Computes a 2D histogram by aggregating values in a 2D array.

    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin:
        scalar: bin size along 2nd dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param ybin:
        scalar: bin size along 1st dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    # if no bounds provided, use min/max of vectors
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    def _get_scale_and_indices(v, bin, lim):
        # if bin is a nonzero scalar, this is a bin size: create scale and indices
        if np.isscalar(bin) and bin != 0:
            scale = np.arange(lim[0], lim[1] + bin / 2, bin)
            ind = (np.floor((v - lim[0]) / bin)).astype(np.int64)
        # if bin == 0, aggregate over unique values
        else:
            scale, ind = np.unique(v, return_inverse=True)
        return scale, ind

    xscale, xind = _get_scale_and_indices(x, xbin, xlim)
    yscale, yind = _get_scale_and_indices(y, ybin, ylim)
    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)

    # if a set of specific values is requested output an array matching the scale dimensions
    if not np.isscalar(xbin) and xbin.size > 1:
        _, iout, ir = np.intersect1d(xbin, xscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ny, xbin.size))
        r[:, iout] = _r[:, ir]
        xscale = xbin

    if not np.isscalar(ybin) and ybin.size > 1:
        _, iout, ir = np.intersect1d(ybin, yscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ybin.size, r.shape[1]))
        r[iout, :] = _r[ir, :]
        yscale = ybin

    return r, xscale, yscale


def bin_spikes(spikes, binsize, interval_indices=False):
    """
    Wrapper for bincount2D which is intended to take in a TimeSeries object of spike times
    and cluster identities and spit out spike counts in bins of a specified width binsize, also in
    another TimeSeries object. Can either return a TS object with each row labeled with the
    corresponding interval or the value of the left edge of the bin.

    :param spikes: Spike times and cluster identities of sorted spikes
    :type spikes: TimeSeries object with \'clusters\' column and timestamps
    :param binsize: Width of the non-overlapping bins in which to bin spikes
    :type binsize: float
    :param interval_indices: Whether to use intervals as the time stamps for binned spikes, rather
        than the left edge value of the bins, defaults to False
    :type interval_indices: bool, optional
    :return: Object with 2D array of shape T x N, for T timesteps and N clusters, and the
        associated time stamps.
    :rtype: TimeSeries object
    """
    if type(spikes) is not core.TimeSeries:
        raise TypeError('Input spikes need to be in TimeSeries object format')

    if not hasattr(spikes, 'clusters'):
        raise AttributeError('Input spikes need to have a clusters attribute. Make sure you set '
                             'columns=(\'clusters\',)) when constructing spikes.')

    rates, tbins, clusters = bincount2D(spikes.times, spikes.clusters, binsize)
    if interval_indices:
        intervals = pd.interval_range(tbins[0], tbins[-1], freq=binsize, closed='left')
        return core.TimeSeries(times=intervals, values=rates.T[:-1], columns=clusters)
    else:
        return core.TimeSeries(times=tbins, values=rates.T, columns=clusters)


def get_units_bunch(spks_b, *args):
    '''
    Returns a bunch, where the bunch keys are keys from `spks` with labels of spike information
    (e.g. unit IDs, times, features, etc.), and the values for each key are arrays with values for
    each unit: these arrays are ordered and can be indexed by unit id.

    Parameters
    ----------
    spks_b : bunch
        A spikes bunch containing fields with spike information (e.g. unit IDs, times, features,
        etc.) for all spikes.
    features : list of strings (optional positional arg)
        A list of names of labels of spike information (which must be keys in `spks`) that specify
        which labels to return as keys in `units`. If not provided, all keys in `spks` are returned
        as keys in `units`.

    Returns
    -------
    units_b : bunch
        A bunch with keys of labels of spike information (e.g. cluster IDs, times, features, etc.)
        whose values are arrays that hold values for each unit. The arrays for each key are ordered
        by unit ID.

    Examples
    --------
    1) Create a units bunch given a spikes bunch, and get the amps for unit #4 from the units
    bunch.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        (*Note, if there is no 'alf' directory, make 'alf' directory from 'ks2' output directory):
        >>> e_spks.ks2_to_alf(path_to_ks_out, path_to_alf_out)
        >>> spks_b = aio.load_object(path_to_alf_out, 'spikes')
        >>> units_b = bb.processing.get_units_bunch(spks_b)
        # Get amplitudes for unit 4.
        >>> amps = units_b['amps']['4']

    TODO add computation time estimate?
    '''

    # Initialize `units`
    units_b = core.Bunch()
    # Get the keys to return for `units`:
    if not args:
        feat_keys = list(spks_b.keys())
    else:
        feat_keys = args[0]
    # Get unit id for each spike and number of units. *Note: `n_units` might not equal `len(units)`
    # because some clusters may be empty (due to a "wontfix" bug in ks2).
    spks_unit_id = spks_b['clusters']
    n_units = np.max(spks_unit_id)
    units = np.unique(spks_b['clusters'])
    # For each key in `units`, iteratively get each unit's values and add as a key to a bunch,
    # `feat_bunch`. After iterating through all units, add `feat_bunch` as a key to `units`:
    for feat in feat_keys:
        # Initialize `feat_bunch` with a key for each unit.
        feat_bunch = core.Bunch((str(unit), np.array([])) for unit in np.arange(n_units))
        for unit in units:
            unit_idxs = np.where(spks_unit_id == unit)[0]
            feat_bunch[str(unit)] = spks_b[feat][unit_idxs]
        units_b[feat] = feat_bunch
    return units_b


def filter_units(units_b, t, **kwargs):
    '''
    Filters units according to some parameters. **kwargs are the keyword parameters used to filter
    the units.

    Parameters
    ----------
    units_b : bunch
        A bunch with keys of labels of spike information (e.g. cluster IDs, times, features, etc.)
        whose values are arrays that hold values for each unit. The arrays for each key are ordered
        by unit ID.
    t : float
        Duration of time over which to calculate the firing rate and false positive rate.

    Keyword Parameters
    ------------------
    min_amp : float
        The minimum mean amplitude (in V) of the spikes in the unit. Default value is 50e-6.
    min_fr : float
        The minimum firing rate (in Hz) of the unit. Default value is 0.5.
    max_fpr : float
        The maximum false positive rate of the unit (using the fp formula in Hill et al. (2011)
        J Neurosci 31: 8699-8705). Default value is 0.2.
    rp : float
        The refractory period (in s) of the unit. Used to calculate `max_fp`. Default value is
        0.002.

    Returns
    -------
    filt_units : ndarray
        The ids of the filtered units.

    See Also
    --------
    get_units_bunch

    Examples
    --------
    1) Filter units according to the default parameters.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        (*Note, if there is no 'alf' directory, make 'alf' directory from 'ks2' output directory):
        >>> e_spks.ks2_to_alf(path_to_ks_out, path_to_alf_out)
        # Get a spikes bunch, units bunch, and filter the units.
        >>> spks_b = aio.load_object(path_to_alf_out, 'spikes')
        >>> units_b = bb.processing.get_units_bunch(spks_b, ['times', 'amps', 'clusters'])
        >>> T = spks_b['times'][-1] - spks_b['times'][0]
        >>> filtered_units = bb.processing.filter_units(units_b, T)

    2) Filter units with no minimum amplitude, a minimum firing rate of 1 Hz, and a max false
    positive rate of 0.2, given a refractory period of 2 ms.
        >>> filtered_units  = bb.processing.filter_units(units_b, T, min_amp=0, min_fr=1)

    TODO: `units_b` input arg could eventually be replaced by `clstrs_b` if the required metrics
          are in `clstrs_b['metrics']`
    '''

    # Set params
    params = {'min_amp': 50e-6, 'min_fr': 0.5, 'max_fpr': 0.2, 'rp': 0.002}  # defaults
    params.update(kwargs)  # update from **kwargs

    # Iteratively filter the units for each filter param #
    # -------------------------------------------------- #
    units = np.asarray(list(units_b.amps.keys()))
    # Remove empty clusters
    empty_cl = np.where([len(units_b.amps[unit]) == 0 for unit in units])[0]
    filt_units = np.delete(units, empty_cl)
    for param in params.keys():
        if param == 'min_amp':  # return units above with amp > `'min_amp'`
            mean_amps = np.asarray([np.mean(units_b.amps[unit]) for unit in filt_units])
            filt_idxs = np.where(mean_amps > params['min_amp'])[0]
            filt_units = filt_units[filt_idxs]
        elif param == 'min_fr':  # return units with fr > `'min_fr'`
            fr = np.asarray([len(units_b.amps[unit]) /
                            (units_b.times[unit][-1] - units_b.times[unit][0])
                            for unit in filt_units])
            filt_idxs = np.where(fr > params['min_fr'])[0]
            filt_units = filt_units[filt_idxs]
        elif param == 'max_fpr':  # return units with fpr < `'max_fpr'`
            fpr = np.zeros_like(filt_units, dtype='float')
            for i, unit in enumerate(filt_units):
                n_spks = len(units_b.amps[unit])
                n_isi_viol = len(np.where(np.diff(units_b.times[unit]) < params['rp'])[0])
                # fpr is min of roots of solved quadratic equation (Hill, et al. 2011).
                c = (t * n_isi_viol) / (2 * params['rp'] * n_spks**2)  # 3rd term in quadratic
                fpr[i] = np.min(np.abs(np.roots([-1, 1, c])))  # solve quadratic
            filt_idxs = np.where(fpr < params['max_fpr'])[0]
            filt_units = filt_units[filt_idxs]
    return filt_units.astype(int)

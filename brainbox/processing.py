'''
Processes data from one form into another, e.g. taking spike times and binning them into
non-overlapping bins and convolving spike times with a gaussian kernel.
'''

import os
import math
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import interpolate, sparse, signal
from scipy.interpolate import interp1d
from brainbox import core
from iblutil.util import Bunch
import brainbox.io.one as bbone
import brainbox.behavior.wheel as wh


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


def compute_cluster_average(spike_clusters, spike_var):
    """
    Quickish way to compute the average of some quantity across spikes in each cluster given
    quantity for each spike

    :param spike_clusters: cluster idx of each spike
    :param spike_var: variable of each spike (e.g spike amps or spike depths)
    :return: cluster id, average of quantity for each cluster, no. of spikes per cluster
    """
    clust, inverse, counts = np.unique(spike_clusters, return_inverse=True, return_counts=True)
    _spike_var = sparse.csr_matrix((spike_var, (inverse, np.zeros(inverse.size, dtype=int))))
    spike_var_avg = np.ravel(_spike_var.toarray()) / counts

    return clust, spike_var_avg, counts


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
    units_b = Bunch()
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
        feat_bunch = Bunch((str(unit), np.array([])) for unit in np.arange(n_units))
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



# ----------- Start of trial event averaging functions -----------

def event_timing_by_trial_type(eid):
  """
  Returns trial timing dataframes for all, left correct, left incorrect,
  right correct, and right incorrect trials respectively in the given eid.

  Parameters
  ----------
  eid : string
    The eid we're returning trial timings by type for

  Returns
  -------
  trdf, left_corr, left_inc, right_corr, right_inc : Dataframe
    Five dataframes containing the event timing for all, left correct,
    left incorrect, right correct, and right incorrect trials respectively in the given session.

  Examples
  --------
  1) Get trial timing dataframes for an example session
    >>> eid = "0802ced5-33a3-405e-8336-b65ebc5cb07c"
    >>> trdf, left_corr, left_inc, right_corr, right_inc = event_timing_by_trial_type(eid)
  """
  trdf = bbone.load_trials_df(eid, maxlen=2., t_before=0.6, t_after=0.6,
                                      wheel_binsize=0.02, ret_abswheel=False,
                                      ret_wheel=True)
  left_corr = trdf.loc[(trdf['contrastLeft'] > 0) & (trdf['feedbackType'] == 1)]
  left_inc = trdf.loc[(trdf['contrastLeft'] > 0) & (trdf['feedbackType'] == -1)]
  right_corr = trdf.loc[(trdf['contrastRight'] > 0) & (trdf['feedbackType'] == 1)]
  right_inc = trdf.loc[(trdf['contrastRight'] > 0) & (trdf['feedbackType'] == -1)]
  ret = [trdf, left_corr, left_inc, right_corr, right_inc]
  # reset indices of each returned dataframe
  for i in range(len(ret)):
    ret[i] = ret[i].reset_index()
  return tuple(ret)

# Using the wheel time and position data we'll find the first time the wheel exceeds 2 degrees in
# either direction for each trial and add that to the data frame
def append_session_wheel_movements(wheel_times, pos, trdf,
                                   threshold=(2 * math.pi / 180), freq=1000):
  """
  Appends the column "first_wheel_move" to the given trial timing dataframe containing the
  first wheel moves for each trial.

  Parameters
  ----------
  wheel_times : np.ndarray of floats
    shape: (# times)
    list of times the wheel moved
  pos : np.ndarray of floats
    shape: (# times)
    list of wheel positions at each time in wheel_times
  trdf : Dataframe
    dataframe of trial event timings. Must contain at least stimOn and feedback times for each
    trial.
  threshold : float
    the smallest degree difference from starting wheel position that is considered the first move
  freq : int
    the recording frequency of the wheel movements

  Returns
  -------
  trdf : Dataframe
    the given dataframe with the time of the first wheel move in each trial appended as the
    column "first_wheel_move".

  Examples
  --------
  1) Append first wheel movements to the trial timing dataframe for an example eid
    >>> import brainbox.io.one as bbone
    >>> import brainbox.behavior.wheel as wh
    >>> eid = "0802ced5-33a3-405e-8336-b65ebc5cb07c"
    >>> freq = 1000
    >>> wheel = one.load_object(eid, 'wheel', collection='alf',
    >>>                                       attribute=['position', 'timestamps'])
    >>> pos, times = wh.interpolate_position(wheel.timestamps, wheel.position, freq=freq)
    >>> all_trials_df = bbone.load_trials_df(eid, maxlen=2.0, t_before=0.6, t_after=0.6,
    >>>                                       wheel_binsize=0.02, ret_abswheel=False,
    >>>                                       ret_wheel=True)
    >>> all_trials_df = append_session_wheel_movements(t, pos, all_trials_df, freq=freq)
  """
  wheel_move_times = []
  for idx, row in trdf.iterrows():
    # Get indices of stimon and feedback events in the array of wheel times
    start_idx = np.searchsorted(wheel_times, row['stimOn_times'], "right")
    feedback_idx = np.searchsorted(wheel_times, row['feedback_times'], "right") # UNITS: indices
    wheel_diff_from_start = pos[start_idx:feedback_idx] - pos[start_idx] # UNITS: degrees
    # Get first index where the wheel is moved more than threshold degrees since stimon
    wheel_move_idx = np.argwhere(abs(wheel_diff_from_start) > threshold)[0]
    # Convert index of first wheel move to s
    if len(wheel_move_idx) > 0:
      wheel_move_time = (wheel_move_idx[0] / freq) # UNITS: seconds
      #if wheel_move_time > too_soon_thresh: # Don't add if wheel move is too close to stimon
      wheel_move_times.append(row['stimOn_times'] + wheel_move_time)
  trdf['first_wheel_move'] = np.array(wheel_move_times)
  return trdf


def all_session_event_timings_by_type(eids, include_wheel=True):
  """
  Returns trial timing dataframes for all, left correct, left incorrect,
  right correct, and right incorrect trials respectively for each given eid.
  Appends first trial wheel moves to each dataframe iff include_wheel

  Parameters
  ----------
  eids : string
    The eids we're returning trial timings by type for
  include_wheel : bool
    True iff we append "first_wheel_move" fields for each trial in the returned dataframes

  Returns
  -------
  (# eids x [trdf, left_corr, left_inc, right_corr, right_inc]) : List of Dataframe
    Five dataframes containing the event timing for all, left correct,
    left incorrect, right correct, and right incorrect trials respectively for each given session.

  Examples
  --------
  1) Get a 2d array of [all_trials, left_corr, left_incorr, right_corr,
     right_incorr] for each session
  >>> trial_timing_dfs = all_session_event_timings_by_type(eids, include_wheel=True)
  >>> trial_timing_dfs = trial_timing_dfs[:, 1:] # exclude df for all trials together
  """
  all_trial_timing_dfs = []
  tqdm.write("Building list of trial timings...")
  for eid in tqdm(eids):
    all_trials, left_corr, left_inc, right_corr, right_inc = event_timing_by_trial_type(eid)

    if include_wheel:
      # Since one of our events is first_wheel_move, append the correponding times to the
      # trial timing dataframes
      wheel = one.load_object(eid, 'wheel', collection='alf', attribute=['position', 'timestamps'])
      pos, t = wh.interpolate_position(wheel.timestamps, wheel.position, freq=1000)
      all_trials = append_session_wheel_movements(t, pos, all_trials)
      left_corr = append_session_wheel_movements(t, pos, left_corr)
      left_inc = append_session_wheel_movements(t, pos, left_inc)
      right_corr = append_session_wheel_movements(t, pos, right_corr)
      right_inc = append_session_wheel_movements(t, pos, right_inc)

    sess_trial_timing_dfs = [all_trials, left_corr, left_inc, right_corr, right_inc]
    all_trial_timing_dfs.append(sess_trial_timing_dfs)
  return all_trial_timing_dfs

def avg_session_event_timings(trdf, event_names):
  """
  Returns cumulative and average times for each given event in the given trial timing dataframe.

  Parameters
  ----------
  trdf : Dataframe
    dataframe of trial event timings. Must contain at least trial_start and trial_end times
    for each trial.
  event_names : list of strings
    a list of the events to average accross sessions and return

  Returns
  -------
  cumulative_timestamps, normalized_timestamps : np.ndarray of float
    shape: (# given events)
    the average seconds since trial start to each given event accross trials and
    the average time to the given events since trial start normalized between 0.0 and 1.0
    respectively

  Examples
  --------
  1) Return average and cumulative event timings for stimOn and feedback over all trials in an
     example eid.
    >>> eid = "0802ced5-33a3-405e-8336-b65ebc5cb07c"
    >>> event_names = ["stimOn_times", "feedback_times"]
    >>> all_trials_df = bbone.load_trials_df(eid, maxlen=2.0, t_before=0.6, t_after=0.6,
                                         wheel_binsize=0.02, ret_abswheel=False,
                                         ret_wheel=True)
    >>> cumulative_timestamps, normalized_timestamps = \
    >>>  avg_session_event_timings(all_trials_df, event_names)
  """
  num_events = len(event_names)
  timestamps = np.empty((trdf.shape[0], num_events + 1))
  for i, (_, row) in enumerate(trdf.iterrows()):
    trial_start = row['trial_start']
    for j in range(num_events):
      timestamps[i, j] = row[event_names[j]] - trial_start # UNITS: seconds since trial start
    timestamps[i, num_events] = row['trial_end'] - trial_start
  cumulative_timestamps = np.cumsum(timestamps, axis=1) # UNITS: total seconds (accross trials)
  # UNITS: normalized seconds (divided by trial end)
  normalized_timestamps = np.divide(cumulative_timestamps, cumulative_timestamps[:, [num_events]])
  return np.mean(cumulative_timestamps, axis=0), np.mean(normalized_timestamps, axis=0)


def avg_event_timings(eids, data_path, event_names, one, show_errors=True):
  """
  Returns and saves the average timings for the given events accross all trials in the given eids
  to the given path

  Parameters
  ----------
  eids : list of strings
    list of all the sessions to calculate average event timings from
  data_path : string
    the output path average event timings will be saved to
  event_names : list of strings
    a list of the events to average accross sessions
  show_errors : bool
    Whether to print errors that occur while calculating average timings

  Returns
  -------
  mean_times : np.ndarray of floats
    shape: (# given events)
    the time to each event since trial_start averaged accross all trials in the given eids
    normalized between 0.0 and 1.0

  Examples
  --------
  1) Compute average and normalized event timings for an example list of eids.
    # To compute avg number of values for stim on to wheel move, wheel move to feedback, etc...
    >>> data_path = "./data/event_avgs/testing"
    >>> event_names = ["stimOn_times", "feedback_times"]
    # takes ~15 mins for all eids in the BWM depending on download speed
    >>> avg_times = avg_event_timings(eids, data_path, event_names, show_errors=False)
    # convert from 0.0 to 1.0 scale to 0 to SCALED_LEN
    >>> avg_event_lengths = np.array(avg_times * SCALED_LEN, dtype="int")
    # convert to the number of values from each event to the next event
    # (instead of normalized time since trial_start)
    >>> for i in range(len(avg_event_lengths) - 1, 0, -1):
    >>>   avg_event_lengths[i] -= avg_event_lengths[i - 1]
    >>> start_to_stimon_len = avg_event_lengths[0]
    >>> stimon_to_feedback_len = avg_event_lengths[1]
  """
  # Compute the average trial lengths for all experiment ids, saving the data along the way
  all_normalized_times = []

  count = 0
  for eid in eids:
    tqdm.write("Averaging trials from eid: " + eid)
    try:
      all_trials_df = bbone.load_trials_df(eid, maxlen=2.0, t_before=0.6, t_after=0.6,
                                           wheel_binsize=0.02, ret_abswheel=False,
                                           ret_wheel=True)
      if "first_wheel_move" in event_names:
        wheel = one.load_object(eid, 'wheel', collection='alf',
                                              attribute=['position', 'timestamps'])
        pos, t = wh.interpolate_position(wheel.timestamps, wheel.position, freq=1000)
        all_trials_df = append_session_wheel_movements(t, pos, all_trials_df)
      _, normalized_time = avg_session_event_timings(all_trials_df, event_names)
    except Exception:
      if show_errors:
        print("Failed to average timings")
        print(traceback.format_exc())
      continue

    tqdm.write("Normalized times: " + str(np.round(normalized_time * 100) / 100))
    all_normalized_times.append(normalized_time)
    count += 1

  all_normalized_times = np.array(all_normalized_times)
  np.save(data_path + 'normalized_trial_lengths.npy', all_normalized_times)
  mean_times = np.mean(all_normalized_times, axis=0)
  print("Average timing for events: " + str(mean_times))
  print("Num eids averaged: " + str(count))
  return mean_times

def causal_gaussian_smoothing(data, num_pts_from_center=21, sigma=1.5):
  """
  Smooths the given data with a causal gaussian filter of the given parameters.

  Parameters
  ----------
  data : np.ndarray
    The data to smooth
  num_pts_from_center : int
    The number of points in the gaussian filter from the center to use for smoothing
  sigma : float
    Size of gaussian

  Returns
  -------
  conv : np.ndarray
    Given data smoothed using a 1d causal gaussian filter with the given parameters

  Examples
  --------
  1) Smooth binned firing rates using a causal gaussian with default paramaters.
    >>> spike_binsize = 0.01
    >>> st = spikes["times"]
    >>> clu = spikes["clusters"]
    # rates in spikes per 0.01 second
    >>> rates, times, clusters = bincount2D(st, clu, spike_binsize)
    >>> smoothed_rates = np.zeros(rates.shape)
    >>> for clu_num in len(range(rates)):
    >>>  smoothed_rates[clu_num] = causal_gaussian_smoothing(rates[clu_num])

  """
  gaussian = signal.windows.gaussian(num_pts_from_center, sigma)
  # Set future half of gaussian to 0.0 so it doesn't affect smoothing
  gaussian[:(num_pts_from_center // 2)] = 0.0
  conv = signal.convolve(data, gaussian, mode="same")
  return conv

def avg_trial_firing_rates_in_window(rates, times, window_start_times, window_end_times):
  """
  Returns a (# clusters, # trials) array containing the average firing rates between
  the window start and end times in each trial

  Parameters
  ----------
  rates : np.ndarray of float
    shape: (# clusters, # times)
    array of firing rates for each cluster at the given times
  times : np.ndarray of float
    shape: (# times)
    the recorded times for each cluster firing rate
  window_start_times : np.ndarray of float
    shape: (# trials)
    the start of the window in seconds to average for each trial
  window_end_times : np.ndarray of float
    shape: (# trials)
    the end of the window in seconds to average for each trial

  Returns
  -------
  trial_avg_rates : np.ndarray of float
    shape: (# clusters, # trials)
    the average firing rate for each cluster in each trial between the given window start and
    end times

  Examples
  --------
  1) Baseline binned firing rates using the average from the window of stimon to 0.4 seconds before.
    >>>  spike_binsize = 0.01
    >>> st = spikes["times"]
    >>> clu = spikes["clusters"]
    # rates in spikes per 0.01 second
    >>> rates, times, clusters = bincount2D(st, clu, spike_binsize)
    >>> baseline_window_size = 0.4 # seconds
    >>> baseline_end_times = np.array(trdf["stimOn_times"])
    >>> baseline_start_times = baseline_end_times - baseline_window_size
    >>> baseline_avg_rates = avg_trial_firing_rates_in_window(rates, times, baseline_start_times, \
    >>>                                                      baseline_end_times)
  """
  rates = np.array(rates, dtype=float)
  num_clusters = len(rates)
  num_trials = len(window_start_times)
  trial_avg_rates = np.zeros((num_clusters, num_trials))

  for trial_num in range(num_trials):
    window_start_idx = np.argmax(times >= window_start_times[trial_num])
    window_end_idx = np.argmax(times >= window_end_times[trial_num])
    window_len = window_end_times[trial_num] - window_start_times[trial_num]
    trial_window_firing_rates = rates[: , window_start_idx:window_end_idx]
    for clu_num in range(num_clusters):
      # average spikes per sec
      trial_avg_rates[clu_num][trial_num] = np.sum(trial_window_firing_rates[clu_num]) / window_len
  return trial_avg_rates

# Default values from baseline procedure in Steinmetz, Nicholas A., et al.
# "Distributed coding of choice, action and engagement across the mouse brain."
def get_trial_baselined_firing_rates(trdf, rates, times, baseline_window_size=0.4, normalize=True,
                                     baseline_norm_constant=0.5):
  """
  Returns a (# clusters, # times) array containing the given firing rates for each cluster
  divided by its average firing rate during the baseline period
  (stim_On - baseline_window_size to stim_On) in each trial.

  Parameters
  ----------
  trdf : Dataframe
  rates : np.ndarray of float
    shape: (# clusters, # times)
    array of firing rates for each cluster at the given times
  times : np.ndarray of float
    shape: (# times)
    the recorded times for each cluster firing rate
  baseline_window_size : float
    the time in seconds before stim on to use for the baseline window
  normalize : bool
    True iff we divide each baselined firing rate by the trial baseline average +
    baseline_norm_constant
  baseline_norm_constant : float
    a constant added to the denominator in baselining so data for clusters with near-zero
    firing rates during the baseline period doesn't blow up

  Returns
  -------
  baselined_rates : np.ndarray
    shape: (# clusters, # times)
    The given firing rates for each cluster baselined by subtracting then
    dividing by each cluster's firing rate during the given baseline period in each trial

  Examples
  --------
  1)
  >>> ssl = bbone.SpikeSortingLoader(pid=pid, one=one)
  >>> spikes, clusters, _ = ssl.load_spike_sorting()
  >>> trdf = bbone.load_trials_df(eid, maxlen=2., t_before=0.6, t_after=0.6,
  >>>                                     wheel_binsize=0.02, ret_abswheel=False,
  >>>                                     ret_wheel=True)
  >>> st = spikes["times"]
  >>> clu = spikes["clusters"]
  >>> clu_num = 60
  >>> rates, times, clusters = bincount2D(st, clu, spike_binsize) # rates in spikes per 0.01 second
  >>> rates = (rates / spike_binsize) # rates in spikes per second at each 0.01s bin
  >>> baselined_rates = get_trial_baselined_firing_rates(trdf, rates[clu_num], times)
  """
  rates = np.array(rates, dtype=float)
  baselined_rates = np.zeros((rates.shape), dtype=float)
  trial_start_times = np.array(trdf["trial_start"])
  trial_end_times = np.array(trdf["trial_end"])
  baseline_end_times = np.array(trdf["stimOn_times"])
  baseline_start_times = baseline_end_times - baseline_window_size
  baseline_avg_rates = avg_trial_firing_rates_in_window(rates, times,
                                                        baseline_start_times, baseline_end_times)
  for trial_num in range(len(baseline_start_times)):
    trial_start_idx = np.argmax(times >= trial_start_times[trial_num])
    trial_end_idx = np.argmax(times >= trial_end_times[trial_num])

    for clu_num in range(len(rates)):
      # average spikes per sec
      clu_trial_baseline_firing_rate = baseline_avg_rates[clu_num][trial_num]
      base_sum = \
          ((rates[clu_num][trial_start_idx:trial_end_idx]) - clu_trial_baseline_firing_rate)
      baselined_rates[clu_num][trial_start_idx:trial_end_idx] = base_sum
      if normalize:
        baselined_rates[clu_num][trial_start_idx:trial_end_idx] /= \
         (clu_trial_baseline_firing_rate + baseline_norm_constant)

  return baselined_rates

def get_clu_fano_factors(clu_spike_times, hist_win=0.01, fr_win=0.5, n_bins=10):
  clu_fano_factors = np.empty((len(clu_spike_times), n_bins))
  for idx, st in enumerate(clu_spike_times):
    ff, ffs, fr = bb.metrics.firing_rate_fano_factor(st, hist_win=hist_win, fr_win=fr_win,
                                                         n_bins=n_bins)
    clu_fano_factors[idx] = ffs
  return clu_fano_factors

############################### Resampling Functions ###############################

# Create a re-sampled set of trial timings for a dataframe based on some set of allowed lengths
# for each trial event
def resample_trial_timing_linear(trdf, avg_event_idxs, event_names, scaled_len=250):
  """
  description description

  Parameters
  ----------
  trdf : Dataframe
    dataframe of trial event timings. Must contain at least trial_start and trial_end times
    for each trial.
  avg_event_idxs : list of int
    The average indices of each given event normalized from 0 to scaled_len
  event_names : list of string
    A list of the names of given events
  scaled_len : int
    The length of the array to resample real time data into

  Returns
  -------
  resampled_times : np.ndarray of float
    shape: (# trials, scaled_len)
    The array containing times resampled to have the given number of values between each event
    up to scaled_len for each trial.

  Examples
  --------
  1)
  >>> trdf = bbone.load_trials_df(eid, maxlen=2., t_before=0.6, t_after=0.6,
  >>>                                     wheel_binsize=0.02, ret_abswheel=False,
  >>>                                     ret_wheel=True)
  >>> avg_event_idxs = [37, 113]
  >>> event_names = ["stimOn_times", "feedback"]
  >>> scaled_len = 250
  >>> resampled_trial_timing = resample_trial_timing_linear(trdf, avg_event_idxs,
  >>>                                                      event_names, scaled_len)
  """
  resampled_times = np.empty((len(trdf), scaled_len))
  avg_event_idxs_copy = avg_event_idxs.copy()
  avg_event_idxs_copy.insert(0, 0)
  avg_event_idxs_copy.append(scaled_len)
  event_names_copy = event_names.copy()
  event_names_copy.insert(0, "trial_start")
  event_names_copy.append("trial_end")

  for i, (_, trial) in enumerate(trdf.iterrows()):
    for j in range(1, len(avg_event_idxs_copy)):
      # Create time arrays between each event with the corresponding number of allocated values
      prev_idx = avg_event_idxs_copy[j - 1]
      cur_idx = avg_event_idxs_copy[j]
      resampled_times[i, prev_idx:cur_idx] = \
        np.linspace(trial[event_names_copy[j - 1]], trial[event_names_copy[j]], cur_idx - prev_idx)

  return resampled_times

############################### Main Function ###############################

def interp_time_data_around_resampled_events(resampled_timing, times, time_series_data):
  """
  Linearly interpolates given time series data to match a resampled timing.

  Parameters
  ----------
  resampled_timing : np.ndarray of float
    shape: (# trials, fixed_len)
    The array containing times resampled to some fixed length for each trial
  times : np.ndarray of float
    shape: (# times)
    The recorded times for each data point in time_series_data
  time_series_data : np.ndarray of number
    shape: (# times)
    Any data recorded at the given times

  Returns
  --------
  interpolated_data : np.ndarray of float
    shape: (# trials, scaled_len)
    The given time series data interpolated to the resampled time

  Examples
  --------
  1)
  >>> clu_num = 60
  >>> rates, times, clusters = bincount2D(st, clu, spike_binsize)
  >>> baselined_rates = get_trial_baselined_firing_rates(trdf, rates[clu_num], times)
  >>> trial_interp_rates = interp_time_data_around_resampled_events(resampled_trial_timing, times,
  >>>                                                               baselined_rates)
  """
  cluster_interpolator = interp1d(times, time_series_data)
  interpolated_data = cluster_interpolator(resampled_timing)
  return interpolated_data

# Trial_event_timings must have at least "trial_start", "trial_end", and all of the event times
# of the given names for each trial
def average_cluster_data_around_events(trial_event_timings, avg_event_idxs, event_names,
                                       times, clu_2_time_series_data, interp_method="linear",
                                       scaled_len=250):
  """
  Interpolates time-series data for each cluster around given events using their indices in a
  resampled array of values, actual timings, and the given method.

  Parameters
  ----------
  trial_event_timings : Dataframe
    dataframe of trial event timings. Must contain at least trial_start and trial_end times
    for each trial.
  avg_event_idxs : list of int
    The average indices of each given event normalized from 0 to scaled_len
  A list of the names of given events
  scaled_len : int
    The length of the array to resample real time data into
  times : np.ndarray of float
    shape: (# times)
    The recorded times for each data point in clu_2_time_series_data
  clu_2_time_series_data np.ndarray of float
    shape: (# clusters, data length)
    An array of arbitrary time series data for some clusters
  interp_method : string
    The interpolation method to use. Defaults to "linear".

  Returns
  --------
  clu_2_event_avgs : np.ndarray of float
    shape: (# clusters, scaled_len)
    An array containing the given time series data interpolated around the given events into
    resampled arrays of length scaled_len for each cluster. Data is averaged over all
    given trials.

  Examples
  --------
  1)
  >>> ssl = bbone.SpikeSortingLoader(pid=pid, one=one)
  >>> spikes, clusters, _ = ssl.load_spike_sorting()
  >>> trdf = bbone.load_trials_df(eid, maxlen=2., t_before=0.6, t_after=0.6,
  >>>                                     wheel_binsize=0.02, ret_abswheel=False,
  >>>                                     ret_wheel=True)
  >>> st = spikes["times"]
  >>> clu = spikes["clusters"]
  >>> clu_num = 60
  >>> rates, times, clusters = bincount2D(st, clu, spike_binsize) # rates in spikes per 0.01 second
  >>> rates = (rates / spike_binsize) # rates in spikes per second at each 0.01s bin
  >>> avg_event_idxs = [37, 113]
  >>> event_names = ["stimOn_times", "feedback"]
  >>> scaled_len = 250
  >>> clu_event_avg = average_cluster_data_around_events(trdf, avg_event_idxs, event_names, times,
  >>>                                                    rates[clu_num], scaled_len)
  """
  # for each set of trials, make a matrix with the times for each trial and event window
  # we'll pre-interpolate these. Note that pre-computing this saves a *ton* of time, since
  # doing this for each cluster individually would be just repeating. Also by having it all
  # in one matrix we can average in one step.
  if interp_method == "linear":
    resampled_trial_timing = resample_trial_timing_linear(trial_event_timings, avg_event_idxs, \
                                                          event_names)
  else:
    raise NotImplementedError("Interpolation method " + interp_method + " not yet implemented")

  # Initialize dictionary mapping event avgs to num_trial_types x scaled_len matrices
  num_clusters = len(clu_2_time_series_data)
  clu_2_event_avgs = np.zeros((num_clusters, scaled_len))

  for clu_num in range(num_clusters):
    trial_rates = interp_time_data_around_resampled_events(resampled_trial_timing, \
      times, clu_2_time_series_data[clu_num])
    np.mean(trial_rates, axis=0)
    clu_2_event_avgs[clu_num] = np.mean(trial_rates, axis=0)

  return clu_2_event_avgs

def event_average_session_firing_rates(pid, trial_timing_dfs, event_names, avg_event_idxs, one,
                                   spike_binsize=0.01, scaled_len=250, norm_method="baseline",
                                   normalize=True, fr_cutoff=0.1):
  """
  Interpolates and normalizes all clusters' firing rates in the given insertion around
  given events into resampled arrays of length scaled_len.

  Parameters
  ----------
  pid : string
    the given probe insertion ID
  trial_timing_dfs : list of Dataframe
    a list of dataframes containing the event timings for each trial type to evaluate
  event_names : list of string
    A list of the names of given events
  avg_event_idxs : list of int
    The average indices of each given event normalized from 0 to scaled_len
  scaled_len : int
    The length of the array to resample real time data into
  spike_binsize : float
    Time in seconds between to use for each spike bin. Defaults to 0.01.
  norm_method : string
    The method to use when normalize firing rates. Currently options are "baseline",
    "fano_factor", or None
  normalize : bool
    True iff we divide baselined firing rates by the trial baseline average +
    baseline_norm_constant.
  fr_cutoff : float
    All clusters with a session averaged firing rate of less than fr_cutoff will be removed
    from the output.

  Returns
  -------
  {"avgs" : clu_event_avgs, "clusters" : filtered_clusters} : dict of np.ndarray
  A dictionary containing "avgs": the normalized firing rates averaged over trial and interpolated
  around the given events into arrays of length scaled_len for each cluster that meets the
  given fr_cutoff", and "clusters": a list of the corresponding cluster numbers for each index
  in the average array after filtering.

  Examples
  --------
  1)
  >>> eid = "0802ced5-33a3-405e-8336-b65ebc5cb07c"
  >>> pid = "7d999a68-0215-4e45-8e6c-879c6ca2b771"
  >>> outpath = "./data/"

  # Load all trial data
  >>> all_trials, left_corr, left_inc, right_corr, right_inc = event_timing_by_trial_type(eid)

  # Precomputed avg value indices for stimon, first wheel move, and feedback when SCALED_LEN = 250
  # If computing from scratch use avg_session_event_timings.
  >>> avg_event_lengths = [37, 51, 62]  # UNITS: values
  >>> avg_event_idxs = list(np.cumsum(avg_event_lengths)) # [37, 88, 150]
  >>> event_names = ["stimOn_times", "first_wheel_move", "feedback_times"]

  # Since one of our events is first_wheel_move, append the correponding times to the
  # trial timing dataframes
  >>> wheel = one.load_object(eid, 'wheel', collection='alf', attribute=['position', 'timestamps'])
  >>> pos, t = wh.interpolate_position(wheel.timestamps, wheel.position, freq=1000)
  >>> all_trials = append_session_wheel_movements(t, pos, all_trials)
  >>> left_corr = append_session_wheel_movements(t, pos, left_corr)
  >>> left_inc = append_session_wheel_movements(t, pos, left_inc)
  >>> right_corr = append_session_wheel_movements(t, pos, right_corr)
  >>> right_inc = append_session_wheel_movements(t, pos, right_inc)
  >>> trial_timing_dfs = [all_trials, left_corr, left_inc]

  # Toy example with a single session
  >>> baselined_event_avgs = event_average_session_firing_rates(pid, trial_timing_dfs,
  >>>                                event_names, avg_event_idxs, \
  >>>                                scaled_len=SCALED_LEN, norm_method="baseline")
  """
  # Load all spiking data
  ssl = bbone.SpikeSortingLoader(pid=pid, one=one)
  spikes, clusters, _ = ssl.load_spike_sorting()

  tqdm.write("Computing averages for pid: " + pid)

  num_trial_types = len(trial_timing_dfs)
  st = spikes["times"]
  clu = spikes["clusters"]
  rates, times, clusters = bincount2D(st, clu, spike_binsize) # rates in spikes per 0.01 second
  rates = (rates / spike_binsize) # rates in spikes per second at each 0.01s bin
  sess_avg_rates = np.mean(rates, axis=1)
  filtered_idxs = np.argwhere(sess_avg_rates > fr_cutoff).flatten()
  # All clusters with a session avg firing rate of > 0.1 Hz
  filtered_clusters = clusters[filtered_idxs]
  num_filtered_clusters = len(filtered_clusters)
  filtered_rates = rates[filtered_clusters]
  smoothed_rates = np.zeros((num_trial_types, num_filtered_clusters, len(filtered_rates[0])))

  # Initialize dictionary mapping event avgs to num_trial_types x scaled_len matrices
  clu_event_avgs = np.zeros((num_trial_types, num_filtered_clusters, scaled_len))

  for i in range(num_trial_types):
    if norm_method == "fano_factor":
      smoothed_rates[i] = get_clu_fano_factors(st[filtered_idxs], n_bins=len(filtered_rates[0]))
    else:
      # Smooth firing rates
      for j in range(num_filtered_clusters):
        smoothed_rates[i][j] = causal_gaussian_smoothing(filtered_rates[j]) # UNITS: spikes per second

      if norm_method == "baseline":
        smoothed_rates[i] = get_trial_baselined_firing_rates(trial_timing_dfs[i], smoothed_rates[i],
                                                            times, normalize=normalize)
      elif norm_method != None:
        raise NotImplementedError("Normalization method " + norm_method + " not yet implemented")

    clu_event_avgs[i] = average_cluster_data_around_events(trial_timing_dfs[i], avg_event_idxs,
                                                           event_names, times, smoothed_rates[i])

  return {"avgs" : clu_event_avgs, "clusters" : filtered_clusters}


def event_average_all_session_firing_rates(outpath, pids, sess_trial_timing_dfs, event_names, avg_event_idxs,
                                       spike_binsize=0.01, scaled_len=250, norm_method="baseline", normalize=True,
                                       fr_cutoff=0.1, use_existing=True, show_errors=True):
  """
  Interpolates and normalizes all clusters' firing rates in the given insertions around
  given events into resampled arrays of length scaled_len. Saves output to the given path.

  Parameters
  ----------
  outpath : string
    The path to save all outputs to
  pids : list of strings
    The given probe insertion IDs
  sess_trial_timing_dfs : list of Dataframe
    a list of dataframes containing the event timings for each trial type to evaluate for each pid
  event_names : list of string
    A list of the names of given events
  avg_event_idxs : list of int
    The average indices of each given event normalized from 0 to scaled_len
  scaled_len : int
    The length of the array to resample real time data into
  spike_binsize : float
    Time in seconds between to use for each spike bin. Defaults to 0.01.
  norm_method : string
    The method to use when normalize firing rates. Currently options are "baseline",
    "fano_factor", or None
  normalize : bool
    True iff we divide baselined firing rates by the trial baseline average +
    baseline_norm_constant.
  fr_cutoff : float
    All clusters with a session averaged firing rate of less than fr_cutoff will be removed
    from the output.
  use_existing : bool
    Uses existing averages if they exist in the given outpath instead of recomputing
  show_errors : bool
    Whether to show errors that occur while computing the ouput for each pid

  Examples
  --------
  1)
  # See event_average_session_firing_rates
  >>> event_average_all_session_firing_rates(outpath, pids, sess_trial_timing_dfs, event_names, \
                                         avg_event_idxs, scaled_len=250)
  """
  for idx, pid in tqdm(enumerate(pids)):
    fname = outpath + "event_avgs_" + pid + ".npy"
    if use_existing and os.path.isfile(fname):
      tqdm.write("Avgs file for pid: " + pid + " already exists!")
    else:
      try:
        clu_event_avgs = event_average_session_firing_rates(pid, sess_trial_timing_dfs[idx], \
                            event_names, avg_event_idxs, spike_binsize, scaled_len,
                            norm_method, normalize, fr_cutoff)
        np.save(fname, clu_event_avgs)
      except Exception:
        if show_errors:
          print("Error while calculating averages")
          print(traceback.format_exc())
        continue

import signal
import numpy as np
from scipy.interpolate import interp1d


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
        trial_window_firing_rates = rates[:, window_start_idx:window_end_idx]
        for clu_num in range(num_clusters):
            # average spikes per sec
            trial_avg_rates[clu_num][trial_num] = \
                np.sum(trial_window_firing_rates[clu_num]) / window_len
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
    baseline_avg_rates = avg_trial_firing_rates_in_window(rates, times, baseline_start_times,
                                                          baseline_end_times)
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
    avg_event_idxs : np.ndarray of int
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
    avg_event_idxs_copy = np.insert(avg_event_idxs, 0, 0)
    avg_event_idxs_copy = np.append(avg_event_idxs_copy, scaled_len)
    event_names_copy = event_names.copy()
    event_names_copy.insert(0, "trial_start")
    event_names_copy.append("trial_end")

    def resample_trial(trial_row):
        idx = trial_row.name
        for j in range(1, len(avg_event_idxs_copy)):
            # Create time arrays between each event with the given number of allocated values
            prev_idx = avg_event_idxs_copy[j - 1]
            cur_idx = avg_event_idxs_copy[j]
            resampled_times[idx, prev_idx:cur_idx] = \
                np.linspace(trial_row[event_names_copy[j - 1]], trial_row[event_names_copy[j]],
                            cur_idx - prev_idx)

    trdf.apply(resample_trial, axis=1)
    return resampled_times


def interp_to_resampled_timing(resampled_timing, times, time_series_data):
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

'''
Computes task related output
'''

import numpy as np
from sklearn.metrics import roc_curve, auc
from brainbox.core import Bunch
from brainbox.population import xcorr


def _get_spike_counts_in_bins(spike_times, spike_clusters, intervals=None):
    """Return the number of spikes in a sequence of time intervals, for each neuron.
    :param spike_times: times of spikes, in seconds
    :type spike_times: 1D array
    :param spike_clusters: spike neurons
    :type spike_clusters: 1D array, same length as spike_times
    :type intervals: the times of the events onsets and offsets
    :param interval: 2D array
    :rtype: 2D array of shape `(n_neurons, n_intervals)`
    """
    # Check inputs.
    assert spike_times.ndim == spike_clusters.ndim == 1
    assert spike_times.shape == spike_clusters.shape
    intervals = np.atleast_2d(intervals)
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2
    n_intervals = intervals.shape[0]

    # For each neuron and each interval, the number of spikes in the interval.
    neuron_ids = np.unique(spike_clusters)
    n_neurons = len(neuron_ids)
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[(t0 <= spike_times) & (spike_times < t1)],
            minlength=neuron_ids.max() + 1)
        counts[:, j] = x[neuron_ids]
    return counts  # value (i, j) is the number of spikes of neuron `neurons[i]` in interval #j


def responsive_units(spike_times, spike_clusters, event_times, pre_time=0.5, post_time=0.5):
    """
    Determine responsive neurons by doing a Wilcoxon's Signed Rank test between a baseline period
    before a certain task event (e.g. stimulus onset) and a period after the task event

    Parameters
    ----------
    spike_times : ndarray
        spike times (in seconds)
    spike_clusters : ndarray
        cluster ids corresponding to each event in `spikes`
    event_times : ndarray
        times (in seconds) of the events from the two groups
    pre_time : float
        time (in seconds) to precede the event times to get the baseline
    post_time : float
        time (in seconds) to follow the event times

    Returns
    -------
    significant_units : ndarray
        an array with the indices of clusters that are significatly modulated
    p_values : ndarray
        the p-values of all the clusters
    cluster_ids : ndarray
        cluster ids of the p-values

    Examples
    --------

    """






def calculate_roc(spike_times, spike_clusters, event_times, event_groups,
                  pre_time=0, post_time=0.25):
    """
    Calcluate area under the ROC curve that indicates how well the activity of the neuron
    distiguishes between the events from the two groups. A value of 0.5 indicates no distinction
    and a value of 1 indicates maximum distinction

    Parameters
    ----------
    spike_times : ndarray
        spike times (in seconds)
    spike_clusters : ndarray
        cluster ids corresponding to each event in `spikes`
    event_times : ndarray
        times (in seconds) of the events from the two groups
    event_groups : ndarray
        group identities of the events as either 0 or 1
    pre_time : float
        time (in seconds) to precede the event times
    post_time : float
        time (in seconds) to follow the event times

    Returns
    -------
    auc_roc : ndarray
        an array of the area under the ROC curve for every neuron

    Examples
    --------

    """




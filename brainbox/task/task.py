'''
Computes task related output
'''

import numpy as np
from scipy.stats import ranksums, sem, t
from sklearn.metrics import roc_auc_score
from brainbox.core import Bunch
from brainbox.population import xcorr


def _get_spike_counts_in_bins(spike_times, spike_clusters, intervals):
    """
    Return the number of spikes in a sequence of time intervals, for each neuron.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    intervals : 2D array of shape (n_events, 2)
        the start and end times of the events

    Returns
    ---------
    counts : 2D array of shape (n_neurons, n_events)
        the spike counts of all neurons ffrom scipy.stats import sem, tor all events
        value (i, j) is the number of spikes of neuron `neurons[i]` in interval #j
    neuron_ids : 1D array
        list of neuron ids
    """

    # Check input
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2

    # For each neuron and each interval, the number of spikes in the interval.
    neuron_ids = np.unique(spike_clusters)
    n_neurons = len(neuron_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[(t0 <= spike_times) & (spike_times < t1)],
            minlength=neuron_ids.max() + 1)
        counts[:, j] = x[neuron_ids]
    return counts, neuron_ids


def responsive_units(spike_times, spike_clusters, event_times, pre_time=0.5, post_time=0.5):
    """
    Determine responsive neurons by doingfrom scipy.stats import sem, t a Wilcoxon's Signed Rank test between a baseline period
    before a certain task event (e.g. stimulus onset) and a period after the task event

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    event_times : 1D array
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
    """

    # Get spike counts for baseline and event timewindow
    baseline_times = np.column_stack(((event_times - pre_time), event_times))
    baseline_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters,
                                                             baseline_times)
    times = np.column_stack((event_times, (event_times + post_time)))
    spike_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times)

    # Do statistics
    p_values = np.empty(spike_counts.shape[0])
    for i in range(spike_counts.shape[0]):
        if (np.sum(baseline_counts[i, :]) == 0) and (np.sum(spike_counts[i, :]) == 0):
            p_values[i] = 1
        else:
            _, p_values[i] = ranksums(baseline_counts[i, :], spike_counts[i, :])
    significant_units = cluster_ids[p_values < 0.05]

    return significant_units, p_values, cluster_ids


def calculate_roc(spike_times, spike_clusters, event_times, event_groups,
                  pre_time=0, post_time=0.25, bootstrap=False, n_bootstrap=1000):
    """
    Calcluate area under the ROC curve that indicates how well the activity of the neuron
    distiguishes between two events (e.g. movement to the right vs left). A value of 0.5 indicates
    the neuron cannot distiguish between the two events. A value of 0 or 1 indicates maximum
    distinction. Significance is determined by bootstrapping the ROC curves. If 0.5 is not
    included in the 95th percentile of the bootstrapped distribution, the neuron is deemed
    to be significant.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    event_times : 1D array
        times (in seconds) of the events from the two groups
    event_groups : 1D array
        group identities of the events as either 0 or 1
    pre_time : float
        time (in seconds) to precede the event times
    post_time : float
        time (in seconds) to follow the event times

    Returns
    -------
    auc_roc : 1D array
        an array of the area under the ROC curve for every neuron
    cluster_ids : 1D array
        cluster ids of the AUC values
    significant_units : 1D array
        neurons which show significant discrimination between the two events
        (only works with bootstrap=True)
    """

    # Get spike counts
    times = np.column_stack(((event_times - pre_time), (event_times + post_time)))
    spike_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times)

    # Calculate area under the ROC curve per neuron
    auc_roc = np.empty(spike_counts.shape[0])
    significant_units = np.empty(0)
    for i in range(spike_counts.shape[0]):
        auc_roc[i] = roc_auc_score(event_groups, spike_counts[i, :])

        if bootstrap is True:
            # Bootstrap ROC curves
            auc_roc_bootstrap = np.empty(n_bootstrap)
            for j in range(n_bootstrap):
                this_sample = np.random.choice(np.arange(len(event_groups)),
                                               int(len(event_groups)/2))
                auc_roc_bootstrap[j] = roc_auc_score(event_groups[this_sample],
                                                     spike_counts[i, this_sample])
            if not ((np.percentile(auc_roc_bootstrap, 5) < 0.5)
                    and (np.percentile(auc_roc_bootstrap, 95) > 0.5)):
                significant_units = np.append(significant_units, cluster_ids[i])

    return auc_roc, cluster_ids, significant_units

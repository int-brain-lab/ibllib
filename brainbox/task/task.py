'''
Computes task related output
'''

import numpy as np
from scipy.stats import ranksums, wilcoxon, ttest_ind, ttest_rel
from ._statsmodels import multipletests
from sklearn.metrics import roc_auc_score


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
    cluster_ids : 1D array
        list of cluster ids
    """

    # Check input
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2

    # For each neuron and each interval, the number of spikes in the interval.
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[(t0 <= spike_times) & (spike_times < t1)],
            minlength=cluster_ids.max() + 1)
        counts[:, j] = x[cluster_ids]
    return counts, cluster_ids


def responsive_units(spike_times, spike_clusters, event_times,
                     pre_time=[0.5, 0], post_time=[0, 0.5], alpha=0.05):
    """
    Determine responsive neurons by doing a Wilcoxon Signed-Rank test between a baseline period
    before a certain task event (e.g. stimulus onset) and a period after the task event.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    event_times : 1D array
        times (in seconds) of the events from the two groups
    pre_time : two-element array
        time (in seconds) preceding the event to get the baseline (e.g. [0.5, 0.2] would be a
        window starting 0.5 seconds before the event and ending at 0.2 seconds before the event)
    post_time : two-element array
        time (in seconds) to follow the event times
    alpha : float
        alpha to use for statistical significance

    Returns
    -------
    significant_units : ndarray
        an array with the indices of clusters that are significatly modulated
    stats : 1D array
        the statistic of the test that was performed
    p_values : ndarray
        the p-values of all the clusters
    cluster_ids : ndarray
        cluster ids of the p-values
    """

    # Get spike counts for baseline and event timewindow
    baseline_times = np.column_stack(((event_times - pre_time[0]), (event_times - pre_time[1])))
    baseline_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters,
                                                             baseline_times)
    times = np.column_stack(((event_times + post_time[0]), (event_times + post_time[1])))
    spike_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times)

    # Do statistics
    p_values = np.empty(spike_counts.shape[0])
    stats = np.empty(spike_counts.shape[0])
    for i in range(spike_counts.shape[0]):
        if np.sum(baseline_counts[i, :] - spike_counts[i, :]) == 0:
            p_values[i] = 1
            stats[i] = 0
        else:
            stats[i], p_values[i] = wilcoxon(baseline_counts[i, :], spike_counts[i, :])

    # Perform FDR correction for multiple testing
    sig_units, p_values, _, _ = multipletests(p_values, alpha, method='fdr_bh')
    significant_units = cluster_ids[sig_units]

    return significant_units, stats, p_values, cluster_ids


def differentiate_units(spike_times, spike_clusters, event_times, event_groups,
                        pre_time=0, post_time=0.5, test='ranksums', alpha=0.05):
    """
    Determine units which significantly differentiate between two task events
    (e.g. stimulus left/right) by performing a statistical test between the spike rates
    elicited by the two events. Default is a Wilcoxon Rank Sum test.

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
        time (in seconds) to precede the event times to get the baseline
    post_time : float
        time (in seconds) to follow the event times
    test : string
        which statistical test to use, options are:
            'ranksums'      Wilcoxon Rank Sums test
            'signrank'      Wilcoxon Signed Rank test (for paired observations)
            'ttest'         independent samples t-test
            'paired_ttest'  paired t-test
    alpha : float
        alpha to use for statistical significance

    Returns
    -------
    significant_units : 1D array
        an array with the indices of clusters that are significatly modulated
    stats : 1D array
        the statistic of the test that was performed
    p_values : 1D array
        the p-values of all the clusters
    cluster_ids : ndarray
        cluster ids of the p-values
    """

    # Check input
    assert test in ['ranksums', 'signrank', 'ttest', 'paired_ttest']
    if (test == 'signrank') or (test == 'paired_ttest'):
        assert np.sum(event_groups == 0) == np.sum(event_groups == 1), \
            'For paired tests the number of events in both groups needs to be the same'

    # Get spike counts for the two events
    times_1 = np.column_stack(((event_times[event_groups == 0] - pre_time),
                               (event_times[event_groups == 0] + post_time)))
    counts_1, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times_1)
    times_2 = np.column_stack(((event_times[event_groups == 1] - pre_time),
                               (event_times[event_groups == 1] + post_time)))
    counts_2, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times_2)

    # Do statistics
    p_values = np.empty(len(cluster_ids))
    stats = np.empty(len(cluster_ids))
    for i in range(len(cluster_ids)):
        if (np.sum(counts_1[i, :]) == 0) and (np.sum(counts_2[i, :]) == 0):
            p_values[i] = 1
            stats[i] = 0
        else:
            if test == 'ranksums':
                stats[i], p_values[i] = ranksums(counts_1[i, :], counts_2[i, :])
            elif test == 'signrank':
                stats[i], p_values[i] = wilcoxon(counts_1[i, :], counts_2[i, :])
            elif test == 'ttest':
                stats[i], p_values[i] = ttest_ind(counts_1[i, :], counts_2[i, :])
            elif test == 'paired_ttest':
                stats[i], p_values[i] = ttest_rel(counts_1[i, :], counts_2[i, :])

    # Perform FDR correction for multiple testing
    sig_units, p_values, _, _ = multipletests(p_values, alpha, method='fdr_bh')
    significant_units = cluster_ids[sig_units]

    return significant_units, stats, p_values, cluster_ids


def roc_single_event(spike_times, spike_clusters, event_times,
                     pre_time=[0.5, 0], post_time=[0, 0.5]):
    """
    Determine how well neurons respond to a certain task event by calculating the area under the
    ROC curve between a baseline period before the event and a period after the event.
    Values of > 0.5 indicate the neuron respons positively to the event and < 0.5 indicate
    a negative response.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    event_times : 1D array
        times (in seconds) of the events from the two groups
    pre_time : two-element array
        time (in seconds) preceding the event to get the baseline (e.g. [0.5, 0.2] would be a
        window starting 0.5 seconds before the event and ending at 0.2 seconds before the event)
    post_time : two-element array
        time (in seconds) to follow the event times

    Returns
    -------
    auc_roc : 1D array
        the area under the ROC curve
    cluster_ids : 1D array
        cluster ids of the p-values
    """

    # Get spike counts for baseline and event timewindow
    baseline_times = np.column_stack(((event_times - pre_time[0]), (event_times - pre_time[1])))
    baseline_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters,
                                                             baseline_times)
    times = np.column_stack(((event_times + post_time[0]), (event_times + post_time[1])))
    spike_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times)

    # Calculate area under the ROC curve per neuron
    auc_roc = np.empty(spike_counts.shape[0])
    for i in range(spike_counts.shape[0]):
        auc_roc[i] = roc_auc_score(np.concatenate((np.zeros(baseline_counts.shape[1]),
                                                   np.ones(spike_counts.shape[1]))),
                                   np.concatenate((baseline_counts[i, :], spike_counts[i, :])))

    return auc_roc, cluster_ids


def roc_between_two_events(spike_times, spike_clusters, event_times, event_groups,
                           pre_time=0, post_time=0.25):
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
    """

    # Get spike counts
    times = np.column_stack(((event_times - pre_time), (event_times + post_time)))
    spike_counts, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times)

    # Calculate area under the ROC curve per neuron
    auc_roc = np.empty(spike_counts.shape[0])
    for i in range(spike_counts.shape[0]):
        auc_roc[i] = roc_auc_score(event_groups, spike_counts[i, :])

    return auc_roc, cluster_ids

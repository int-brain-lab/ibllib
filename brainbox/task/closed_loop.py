'''
Computes task related output
'''

import numpy as np
from scipy.stats import ranksums, wilcoxon, ttest_ind, ttest_rel
from ._statsmodels import multipletests
from sklearn.metrics import roc_auc_score
import pandas as pd
from brainbox.population.decode import get_spike_counts_in_bins


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
    baseline_counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters,
                                                            baseline_times)
    times = np.column_stack(((event_times + post_time[0]), (event_times + post_time[1])))
    spike_counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)

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
    counts_1, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times_1)
    times_2 = np.column_stack(((event_times[event_groups == 1] - pre_time),
                               (event_times[event_groups == 1] + post_time)))
    counts_2, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times_2)

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
    baseline_counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters,
                                                            baseline_times)
    times = np.column_stack(((event_times + post_time[0]), (event_times + post_time[1])))
    spike_counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)

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
    spike_counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)

    # Calculate area under the ROC curve per neuron
    auc_roc = np.empty(spike_counts.shape[0])
    for i in range(spike_counts.shape[0]):
        auc_roc[i] = roc_auc_score(event_groups, spike_counts[i, :])

    return auc_roc, cluster_ids


def _get_biased_probs(n: int, idx: int = -1, prob: float = 0.5) -> list:
    n_1 = n - 1
    z = n_1 + prob
    p = [1 / z] * (n_1 + 1)
    p[idx] *= prob
    return p


def _draw_contrast(
    contrast_set: list, prob_type: str = "biased", idx: int = -1, idx_prob: float = 0.5
) -> float:
    if prob_type == "biased":
        p = _get_biased_probs(len(contrast_set), idx=idx, prob=idx_prob)
        return np.random.choice(contrast_set, p=p)
    elif prob_type == "uniform":
        return np.random.choice(contrast_set)


def _draw_position(position_set, stim_probability_left):
    return int(
        np.random.choice(
            position_set, p=[stim_probability_left, 1 - stim_probability_left]
        )
    )


def generate_pseudo_blocks(n_trials, factor=60, min_=20, max_=100, first5050=90):
    """
    Generate a pseudo block structure

    Parameters
    ----------
    n_trials : int
        how many trials to generate
    factor : int
        factor of the exponential
    min_ : int
        minimum number of trials per block
    max_ : int
        maximum number of trials per block
    first5050 : int
        amount of trials with 50/50 left right probability at the beginning

    Returns
    ---------
    probabilityLeft : 1D array
        array with probability left per trial
    """

    block_ids = []
    while len(block_ids) < n_trials:
        x = np.random.exponential(factor)
        while (x <= min_) | (x >= max_):
            x = np.random.exponential(factor)
        if (len(block_ids) == 0) & (np.random.randint(2) == 0):
            block_ids += [0.2] * int(x)
        elif (len(block_ids) == 0):
            block_ids += [0.8] * int(x)
        elif block_ids[-1] == 0.2:
            block_ids += [0.8] * int(x)
        elif block_ids[-1] == 0.8:
            block_ids += [0.2] * int(x)
    return np.array([0.5] * first5050 + block_ids[:n_trials - first5050])


def generate_pseudo_stimuli(n_trials, contrast_set=[0.06, 0.12, 0.25, 1], first5050=90):
    """
    Generate a block structure with stimuli

    Parameters
    ----------
    n_trials : int
        number of trials to generate
    contrast_set : 1D array
        the contrasts that are presented. The default is [0.06, 0.12, 0.25, 1].
    first5050 : int
        Number of 50/50 trials at the beginning of the session. The default is 90.

    Returns
    -------
    p_left : 1D array
        probability of left stimulus
    contrast_left : 1D array
        contrast on the left
    contrast_right : 1D array
        contrast on the right

    """

    # Initialize vectors
    contrast_left = np.empty(n_trials)
    contrast_left[:] = np.nan
    contrast_right = np.empty(n_trials)
    contrast_right[:] = np.nan

    # Generate block structure
    p_left = generate_pseudo_blocks(n_trials)

    for i in range(n_trials):

        # Draw position and contrast for this trial
        position = _draw_position([-1, 1], p_left[i])
        contrast = _draw_contrast(contrast_set, 'uniform')

        # Add to trials
        if position == -1:
            contrast_left[i] = contrast
        elif position == 1:
            contrast_right[i] = contrast

    return p_left, contrast_left, contrast_right


def generate_pseudo_session(trials, generate_choices=True):
    """
    Generate a complete pseudo session with biased blocks, all stimulus contrasts, choices and
    rewards and omissions. Biased blocks and stimulus contrasts are generated using the same
    statistics as used in the actual task. The choices of the animal are generated using the
    actual psychometrics of the animal in the session. For each synthetic trial the choice is
    determined by drawing from a Bernoulli distribution that is biased according to the proportion
    of times the animal chose left for the stimulus contrast, side, and block probability.
    No-go trials are ignored in the generating of the synthetic choices.

    Parameters
    ----------
    trials : DataFrame
        Pandas dataframe with columns as trial vectors loaded using ONE
    generate_choices : bool
        whether to generate the choices (runs faster without)

    Returns
    -------
    pseudo_trials : DataFrame
        a trials dataframe with synthetically generated trials
    """

    # Get contrast set presented to the animal
    contrast_set = np.unique(trials['contrastLeft'][~np.isnan(trials['contrastLeft'])])
    signed_contrast = trials['contrastRight'].copy()
    signed_contrast[np.isnan(signed_contrast)] = -trials['contrastLeft'][
        ~np.isnan(trials['contrastLeft'])]

    # Generate synthetic session
    pseudo_trials = pd.DataFrame()
    pseudo_trials['probabilityLeft'] = generate_pseudo_blocks(trials.shape[0])

    # For each trial draw stimulus contrast and side and generate a synthetic choice
    for i in range(pseudo_trials.shape[0]):

        # Draw position and contrast for this trial
        position = _draw_position([-1, 1], pseudo_trials['probabilityLeft'][i])
        contrast = _draw_contrast(contrast_set, 'uniform')
        signed_stim = contrast * np.sign(position)

        if generate_choices:
            # Generate synthetic choice by drawing from Bernoulli distribution
            trial_select = ((signed_contrast == signed_stim) & (trials['choice'] != 0)
                            & (trials['probabilityLeft'] == pseudo_trials['probabilityLeft'][i]))
            p_right = (np.sum(trials['choice'][trial_select] == 1)
                       / trials['choice'][trial_select].shape[0])
            this_choice = [-1, 1][np.random.binomial(1, p_right)]

            # Add to trials
            if position == -1:
                pseudo_trials.loc[i, 'contrastLeft'] = contrast
                if this_choice == -1:
                    pseudo_trials.loc[i, 'feedbackType'] = -1
                elif this_choice == 1:
                    pseudo_trials.loc[i, 'feedbackType'] = 1
            elif position == 1:
                pseudo_trials.loc[i, 'contrastRight'] = contrast
                if this_choice == -1:
                    pseudo_trials.loc[i, 'feedbackType'] = 1
                elif this_choice == 1:
                    pseudo_trials.loc[i, 'feedbackType'] = -1
            pseudo_trials.loc[i, 'choice'] = this_choice
        else:
            if position == -1:
                pseudo_trials.loc[i, 'contrastLeft'] = contrast
            elif position == 1:
                pseudo_trials.loc[i, 'contrastRight'] = contrast
        pseudo_trials.loc[i, 'stim_side'] = position
    pseudo_trials['signed_contrast'] = pseudo_trials['contrastRight']
    pseudo_trials.loc[pseudo_trials['signed_contrast'].isnull(),
                      'signed_contrast'] = -pseudo_trials['contrastLeft']
    return pseudo_trials

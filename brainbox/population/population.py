'''
Population functions.

Code from https://github.com/cortex-lab/phylib/blob/master/phylib/stats/ccg.py by C. Rossant.
Code for decoding by G. Meijer
'''

import numpy as np
import scipy as sp
import types
from itertools import groupby
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle as sklearn_shuffle


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
    assert np.all(np.diff(spike_times) >= 0), "Spike times need to be sorted"

    intervals_idx = np.searchsorted(spike_times, intervals)

    # For each neuron and each interval, the number of spikes in the interval.
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[intervals_idx[j, 0]:intervals_idx[j, 1]],
            minlength=cluster_ids.max() + 1)
        counts[:, j] = x[cluster_ids]
    return counts, cluster_ids


def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.

    Implicitely assume that:

    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.

    This is not checked for performance reasons.

    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    bbins = np.bincount(indices)
    arr[:len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    return arr[steps:] - arr[:len(arr) - steps]


def _create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=np.int32)


def _symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    correlograms[..., 0] = np.maximum(
        correlograms[..., 0], correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, correlograms))


def _generate_pseudo_blocks(n_trials, factor=60, min_=20, max_=100):
    block_ids = []
    while len(block_ids) < n_trials:
        x = np.random.exponential(factor)
        while (x <= min_) | (x >= max_):
            x = np.random.exponential(factor)
        if (len(block_ids) == 0) & (np.random.randint(2) == 0):
            block_ids += [0] * int(x)
        elif (len(block_ids) == 0):
            block_ids += [1] * int(x)
        elif block_ids[-1] == 0:
            block_ids += [1] * int(x)
        elif block_ids[-1] == 1:
            block_ids += [0] * int(x)
    return np.array(block_ids[:n_trials])


def xcorr(spike_times, spike_clusters, bin_size=None, window_size=None):
    """Compute all pairwise cross-correlograms among the clusters appearing in `spike_clusters`.

    Parameters
    ----------

    :param spike_times: Spike times in seconds.
    :type spike_times: array-like
    :param spike_clusters: Spike-cluster mapping.
    :type spike_clusters: array-like
    :param bin_size: Size of the bin, in seconds.
    :type bin_size: float
    :param window_size: Size of the window, in seconds.
    :type window_size: float

    Returns an `(n_clusters, n_clusters, winsize_samples)` array with all pairwise
    cross-correlograms.

    """
    assert np.all(np.diff(spike_times) >= 0), ("The spike times must be increasing.")
    assert spike_times.ndim == 1
    assert spike_times.shape == spike_clusters.shape

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds

    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds
    winsize_bins = 2 * int(.5 * window_size / bin_size) + 1

    # Take the cluster order into account.
    clusters = np.unique(spike_clusters)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_times, dtype=np.bool)

    correlograms = _create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Interval between spike i and spike i+shift.
        spike_diff = _diff_shifted(spike_times, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = np.round(spike_diff / bin_size).astype(np.int64)

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins / 2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = np.ravel_multi_index(
            (spike_clusters_i[:-shift][m], spike_clusters_i[+shift:][m], d), correlograms.shape)

        # Increment the matching spikes in the correlograms array.
        _increment(correlograms.ravel(), indices)

        shift += 1

    return _symmetrize_correlograms(correlograms)


def decode(spike_times, spike_clusters, event_times, event_groups, pre_time=0, post_time=0.5,
           classifier='bayes-multinomial', cross_validation='kfold', num_splits=5, prob_left=None,
           custom_validation=None, n_neurons='all', iterations=1, shuffle=False, phase_rand=False,
           pseudo_blocks=False):
    """
    Use decoding to classify groups of trials (e.g. stim left/right). Classification is done using
    the population vector of summed spike counts from the specified time window. Cross-validation
    is achieved using n-fold cross validation or leave-one-out cross validation. Decoders can
    decode any number of groups. When providing the classfier with an imbalanced dataset (not
    the same number of trials in each group) the chance level will not be 1/groups. In that case,
    to compare the classification performance against change one has to either determine chance
    level by decoding a shuffled dataset or use the 'auroc' metric as readout (this metric is
    robust against imbalanced datasets)

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    event_times : 1D or 2D array
        times (in seconds) of the events from the two groups, if it's a 2D array every row will
    event_groups : 1D or 2D array
        group identities of the events, can be any number of groups, accepts integers and strings
    pre_time : float
        time (in seconds) preceding the event times
    post_time : float
        time (in seconds) following the event times
    classifier : string or sklearn object
        which decoder to use, either input a scikit learn clf object directly or a string.
        When it's a string options are (all classifiers are used with default options):
            'bayes-multinomal'      Naive Bayes with multinomial likelihood
            'bayes-gaussian'        Naive Bayes with gaussian likelihood
            'forest'                Random forest
            'regression'            Logistic regression
            'lda'                   Linear Discriminant Analysis
    cross_validation : string of generator object
        which cross-validation method to use
        you can input the .split method of any sklearn cross-validation method
        or a string. If it's a string, options are:
            'none'              No cross-validation
            'kfold'             K-fold cross-validation
            'kfold-interleaved' K-fold cross validation with interleaved trial selection
            'leave-one-out'     Leave out the trial that is being decoded
            'block'             Leave out the block the to-be-decoded trial is in
    num_splits : integer
        ** only for 'kfold' cross-validation **
        Number of splits to use for k-fold cross validation, a value of 5 means that the decoder
        will be trained on 4/5th of the data and used to predict the remaining 1/5th. This process
        is repeated five times so that all data has been used as both training and test set.
    prob_left : 1D array
        ** only for 'block' cross-validation **
        the probability of the stimulus appearing on the left for each trial in event_times
    custom_validation : generator
        ** only for 'custom' cross-validation **
        a generator object with the splits to be used for cross validation using this format:
            (
                (split1_train_idxs, split1_test_idxs),
                (split2_train_idxs, split2_test_idxs),
                (split3_train_idxs, split3_test_idxs),
             ...)
    n_neurons : string or integer
        number of neurons to randomly subselect from the population (default is 'all')
    iterations : int
        number of times to repeat the decoding (especially usefull when subselecting neurons)
    shuffle : boolean
        whether to shuffle the trial labels each decoding iteration
    phase_rand : boolean
        whether to use phase randomization of the activity over trials to use as a "chance"
        predictor
    pseudo_blocks : boolean
        whether to generate pseudo blocks with the same statistics as the actual blocks
        to estimate chance level

    Returns
    -------
    results : dict
        dictionary with decoding results

        accuracy : float
            accuracy of the classifier in percentage correct
        f1 : float
            F1 score of the classifier
        auroc : float
            the area under the ROC curve of the classification performance
        confusion_matrix : 2D array
            normalized confusion matrix
        predictions : 2D array with dimensions iterations x trials
            predicted group label for all trials in every iteration
        probabilities : 2D array with dimensions iterations x trials
            classification probability for all trials in every iteration
    """

    # Check input
    if type(classifier) == str:
        assert classifier in ['bayes-multinomial', 'bayes-gaussian', 'forest', 'regression', 'lda']
    assert (type(cross_validation) == str) or (isinstance(cross_validation, types.GeneratorType))
    if type(cross_validation) == str:
        assert cross_validation in ['none', 'kfold', 'kfold-interleaved', 'leave-one-out', 'block']
    assert event_times.shape[0] == event_groups.shape[0]
    if cross_validation == 'block':
        assert event_times.shape[0] == prob_left.shape[0]

    # Get matrix of all neuronal responses
    times = np.column_stack(((event_times - pre_time), (event_times + post_time)))
    pop_vector, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times)
    pop_vector = pop_vector.T

    # Exclude last trial if the number of trials is even and phase shuffling
    if (phase_rand is True) & (event_groups.shape[0] % 2 == 0):
        event_groups = event_groups[:-1]
        pop_vector = pop_vector[:-1]

    # Initialize classifier
    if type(classifier) == str:
        if classifier == 'forest':
            clf = RandomForestClassifier()
        elif classifier == 'bayes-multinomial':
            clf = MultinomialNB()
        elif classifier == 'bayes-gaussian':
            clf = GaussianNB()
        elif classifier == 'regression':
            clf = LogisticRegression(solver='liblinear')
        elif classifier == 'lda':
            clf = LinearDiscriminantAnalysis()
    else:
        clf = classifier

    # Pre-allocate variables
    acc = np.zeros(iterations)
    f1 = np.zeros(iterations)
    auroc = np.zeros(iterations)
    conf_matrix_norm = np.zeros((np.shape(np.unique(event_groups))[0],
                                 np.shape(np.unique(event_groups))[0],
                                 iterations))
    pred = np.zeros([iterations, pop_vector.shape[0]])
    prob = np.zeros([iterations, pop_vector.shape[0]])

    for i in range(iterations):

        # Pre-allocate variables for this iteration
        y_pred = np.zeros(event_groups.shape)
        y_probs = np.zeros(event_groups.shape)

        # Get neurons to use for this iteration
        if n_neurons == 'all':
            sub_pop_vector = pop_vector
        else:
            use_neurons = np.random.choice(pop_vector.shape[1], n_neurons, replace=False)
            sub_pop_vector = pop_vector[:, use_neurons]

        # Shuffle trail labels if necessary
        if shuffle is True:
            event_groups = sklearn_shuffle(event_groups)

        # Perform phase randomization of activity over trials if necessary
        if phase_rand is True:
            if i == 0:
                original_pop_vector = sub_pop_vector
            rand_pop_vector = np.empty(original_pop_vector.shape)
            frequencies = int((original_pop_vector.shape[0] - 1) / 2)
            fsignal = sp.fft.fft(original_pop_vector, axis=0)
            power = np.abs(fsignal[1:1 + frequencies])
            phases = 2 * np.pi * np.random.rand(frequencies)
            for k in range(original_pop_vector.shape[1]):
                newfsignal = fsignal[0, k]
                newfsignal = np.append(newfsignal, np.exp(1j * phases) * power[:, k])
                newfsignal = np.append(newfsignal, np.flip(np.exp(-1j * phases) * power[:, k]))
                newsignal = sp.fft.ifft(newfsignal)
                rand_pop_vector[:, k] = np.abs(newsignal.real)
            sub_pop_vector = rand_pop_vector

        # Generate pseudo blocks if necessary
        if pseudo_blocks:
            event_groups = _generate_pseudo_blocks(event_groups.shape[0])

        if cross_validation == 'none':

            # Fit the model on all the data and predict
            clf.fit(sub_pop_vector, event_groups)
            y_pred = clf.predict(sub_pop_vector)

            #  Get the probability of the prediction for ROC analysis
            probs = clf.predict_proba(sub_pop_vector)
            y_probs = probs[:, 1]  # keep positive only

        else:
            # Perform cross-validation
            if cross_validation == 'leave-one-out':
                cv = LeaveOneOut().split(sub_pop_vector)
            elif cross_validation == 'kfold':
                cv = KFold(n_splits=num_splits).split(sub_pop_vector)
            elif cross_validation == 'kfold-interleaved':
                cv = KFold(n_splits=num_splits, shuffle=True).split(sub_pop_vector)
            elif cross_validation == 'block':
                block_lengths = [sum(1 for i in g) for k, g in groupby(prob_left)]
                blocks = np.repeat(np.arange(len(block_lengths)), block_lengths)
                cv = LeaveOneGroupOut().split(sub_pop_vector, groups=blocks)
            elif cross_validation == 'custom':
                cv = cross_validation

            # Loop over the splits into train and test
            for train_index, test_index in cv:

                # Fit the model to the training data
                clf.fit(sub_pop_vector[train_index], event_groups[train_index])

                # Predict the test data
                y_pred[test_index] = clf.predict(sub_pop_vector[test_index])

                # Get the probability of the prediction for ROC analysis
                probs = clf.predict_proba(sub_pop_vector[test_index])
                y_probs[test_index] = probs[:, 1]  # keep positive only

        # Calculate performance metrics and confusion matrix
        acc[i] = accuracy_score(event_groups, y_pred)
        f1[i] = f1_score(event_groups, y_pred)
        auroc[i] = roc_auc_score(event_groups[~np.isnan(y_probs)], y_probs[~np.isnan(y_probs)])
        conf_matrix = confusion_matrix(event_groups, y_pred)
        conf_matrix_norm[:, :, i] = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Add prediction and probability to matrix
        pred[i, :] = y_pred
        prob[i, :] = y_probs

    # Make integers from arrays when there's only one iteration
    if iterations == 1:
        acc = acc[0]
        f1 = f1[0]
        auroc = auroc[0]

    # Add to results dictionary
    if cross_validation == 'kfold':
        results = dict({'accuracy': acc, 'f1': f1, 'auroc': auroc,
                        'predictions': pred, 'probabilities': prob,
                        'confusion_matrix': conf_matrix_norm,
                        'n_groups': np.shape(np.unique(event_groups))[0],
                        'classifier': classifier, 'cross_validation': '%d-fold' % num_splits,
                        'iterations': iterations, 'shuffle': shuffle, 'phase_rand': phase_rand,
                        'pseudo_blocks': pseudo_blocks})

    else:
        results = dict({'accuracy': acc, 'f1': f1, 'auroc': auroc,
                        'predictions': pred, 'probabilities': prob,
                        'confusion_matrix': conf_matrix_norm,
                        'n_groups': np.shape(np.unique(event_groups))[0],
                        'classifier': classifier, 'cross_validation': cross_validation,
                        'iterations': iterations, 'shuffle': shuffle, 'phase_rand': phase_rand,
                        'pseudo_blocks': pseudo_blocks})
    return results


def lda_project(spike_times, spike_clusters, event_times, event_groups, pre_time=0, post_time=0.5,
                cross_validation='kfold', num_splits=5, prob_left=None, custom_validation=None):
    """
    Use linear discriminant analysis to project population vectors to the line that best separates
    the two groups. When cross-validation is used, the LDA projection is fitted on the training
    data after which the test data is projected to this projection.

    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    event_times : 1D array
        times (in seconds) of the events from the two groups
    event_groups : 1D array
        group identities of the events, can be any number of groups, accepts integers and strings
    pre_time : float
        time (in seconds) preceding the event times
    post_time : float
        time (in seconds) following the event times
    cross_validation : string
        which cross-validation method to use, options are:
            'none'              No cross-validation
            'kfold'             K-fold cross-validation
            'leave-one-out'     Leave out the trial that is being decoded
            'block'             Leave out the block the to-be-decoded trial is in
            'custom'            Any custom cross-validation provided by the user
    num_splits : integer
        ** only for 'kfold' cross-validation **
        Number of splits to use for k-fold cross validation, a value of 5 means that the decoder
        will be trained on 4/5th of the data and used to predict the remaining 1/5th. This process
        is repeated five times so that all data has been used as both training and test set.
    prob_left : 1D array
        ** only for 'block' cross-validation **
        the probability of the stimulus appearing on the left for each trial in event_times
    custom_validation : generator
        ** only for 'custom' cross-validation **
        a generator object with the splits to be used for cross validation using this format:
            (
                (split1_train_idxs, split1_test_idxs),
                (split2_train_idxs, split2_test_idxs),
                (split3_train_idxs, split3_test_idxs),
             ...)
    n_neurons : int
        Group size of number of neurons to be sub-selected

    Returns
    -------
    lda_projection : 1D array
        the position along the LDA projection axis for the population vector of each trial

    """

    # Check input
    assert cross_validation in ['none', 'kfold', 'leave-one-out', 'block', 'custom']
    assert event_times.shape[0] == event_groups.shape[0]
    if cross_validation == 'block':
        assert event_times.shape[0] == prob_left.shape[0]
    if cross_validation == 'custom':
        assert isinstance(custom_validation, types.GeneratorType)

    # Get matrix of all neuronal responses
    times = np.column_stack(((event_times - pre_time), (event_times + post_time)))
    pop_vector, cluster_ids = _get_spike_counts_in_bins(spike_times, spike_clusters, times)
    pop_vector = pop_vector.T

    # Initialize
    lda = LinearDiscriminantAnalysis()
    lda_projection = np.zeros(event_groups.shape)

    if cross_validation == 'none':
        # Find the best LDA projection on all data and transform those data
        lda_projection = lda.fit_transform(pop_vector, event_groups)

    else:
        # Perform cross-validation
        if cross_validation == 'leave-one-out':
            cv = LeaveOneOut().split(pop_vector)
        elif cross_validation == 'kfold':
            cv = KFold(n_splits=num_splits).split(pop_vector)
        elif cross_validation == 'block':
            block_lengths = [sum(1 for i in g) for k, g in groupby(prob_left)]
            blocks = np.repeat(np.arange(len(block_lengths)), block_lengths)
            cv = LeaveOneGroupOut().split(pop_vector, groups=blocks)
        elif cross_validation == 'custom':
            cv = custom_validation

        # Loop over the splits into train and test
        for train_index, test_index in cv:

            # Find LDA projection on the training data
            lda.fit(pop_vector[train_index], [event_groups[j] for j in train_index])

            # Project the held-out test data to projection
            lda_projection[test_index] = lda.transform(pop_vector[test_index]).T[0]

    return lda_projection

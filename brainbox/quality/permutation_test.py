"""
Quality control for arbitrary metrics, using permutation testing.

Written by Sebastian Bruijns
"""

import numpy as np
import time
# TODO: take in eids and download data yourself?


def permut_test(data1, data2, metric, n_permut=1000, plot=False):
    """
    Compute the probability of observating metric difference for datasets, via permutation testing.

    We're taking absolute values of differences, order of dataset input shouldn't matter
    We're only computing means, what if we want to apply a more complicated function to the
    permutation result?
    Pay attention to always give one list (even if its just one dataset, but then it doesn't make
    sense anyway...)

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : array-like (optional)
        A subset of all units for which to create the bar plot. (If `None`, all units are used)
    feat_names : list of strings (optional)
        A list of n

    Returns
    -------
    p_vals_b : bunch
        A bunch with `feat_n
    cv_b : bunch
        A bunch with `feat_names` as keys, c

    See Also
    --------
    aasdf

    Examples
    --------
    asdf
    """
    # Calculate metrics and true difference between groups
    metrics1 = [metric(d) for d in data1]
    metrics2 = [metric(d) for d in data2]
    true_diff = np.abs(np.mean(metrics1) - np.mean(metrics2))

    # Prepare permutations
    size1 = len(metrics1)
    diffs = np.concatenate((metrics1, metrics2))
    permutations = np.zeros((n_permut, diffs.size), dtype=np.int32)

    # Create permutations, could be parallelized or vectorized in principle, but unclear how
    indizes = np.arange(diffs.size)
    for i in range(n_permut):
        np.random.shuffle(indizes)
        permutations[i] = indizes

    permut_diffs = np.abs(np.mean(diffs[permutations[:, :size1]], axis=1) - np.mean(diffs[permutations[:, size1:]], axis=1))
    p = len(permut_diffs[permut_diffs > true_diff]) / n_permut

    return p


if __name__ == '__main__':
    rng = np.random.RandomState(2)
    data1 = rng.normal(0, 1, (3, 5))
    data2 = rng.normal(0.5, 1, (2, 5))
    t = time.time()
    p = permut_test(data1, data2, np.mean)
    print(time.time() - t)
    print(p)

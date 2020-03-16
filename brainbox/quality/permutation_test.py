"""
Quality control for arbitrary metrics, using permutation testing.

Written by Sebastian Bruijns
"""

import numpy as np
import time
import matplotlib.pyplot as plt
# TODO: take in eids and download data yourself?


def permut_test(data1, data2, metric, n_permut=1000, show=False, title=None):
    """
    Compute the probability of observating metric difference for datasets, via permutation testing.

    We're taking absolute values of differences, because the order of dataset input shouldn't
    matter
    We're only computing means, what if we want to apply a more complicated function to the
    permutation result?
    Pay attention to always give one list (even if its just one dataset, but then it doesn't make
    sense anyway...)

    Parameters
    ----------
    data1 : array-like
        First data set, list or array of data-entities to use for permutation test
        (make data2 optional and then permutation test more similar to tuning sensitivity?)
    data2 : array-like
        Second data set, also list or array of data-entities to use for permutation test
    metric : function, array-like -> float
        Metric to use for permutation test, will be used to reduce elements of data1 and data2
        to one number
    n_permut : integer (optional)
        Number of perumtations to use for test
    plot : Boolean (optional)
        Whether or not to show a plot of the permutation distribution and a marker for the position
        of the true difference in relation to this distribution

    Returns
    -------
    p : float
        p-value of true difference in permutation distribution

    See Also
    --------
    TODO:

    Examples
    --------
    TODO:
    """
    # Calculate metrics and true difference between groups
    print('data1')
    print(data1)
    metrics1 = [metric(d) for d in data1]
    print('metrics1')
    print(metrics1)
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

    permut_diffs = np.abs(np.mean(diffs[permutations[:, :size1]], axis=1) -
                          np.mean(diffs[permutations[:, size1:]], axis=1))
    p = len(permut_diffs[permut_diffs > true_diff]) / n_permut

    if show or title:
        plot_permut_test(permut_diffs=permut_diffs, true_diff=true_diff, p=p, title=title)

    return p


def plot_permut_test(permut_diffs, true_diff, p, title=None):
    """Plot permutation test result."""
    n, _, _ = plt.hist(permut_diffs)
    plt.plot(true_diff, np.max(n) / 20, '*r', markersize=12)

    # Prettify plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("p = {}".format(p))

    if title:
        plt.savefig(title + '.png')
    plt.close()


if __name__ == '__main__':
    rng = np.random.RandomState(2)
    data1 = rng.normal(0, 1, (23, 5))
    data2 = rng.normal(0.1, 1, (32, 5))
    t = time.time()
    p = permut_test(data1, data2, np.mean, plot=True)
    print(time.time() - t)
    print(p)

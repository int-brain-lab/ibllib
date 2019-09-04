import numpy as np


def _preprocess(array, smoothing_sd=25, n_pca_dims=20):
    """
    Preprocess neural data for cca analysis with smoothing and pca

    :param array: array of shape (T, n_features)
    :type array: array-like
    :param smoothing_sd: gaussian smoothing kernel standard deviation (ms)
    :type smoothing_sd: float
    :param pca_dims:
    :type pca_dims: int
    :return: preprocessed neural data
    :rtype: array-like, shape (T, pca_dims)
    """
    pass


def cca(array_0, array_1, n_cca_dims=10, **preprocess_kwargs):
    """

    :param array_0:
    :param array_1:
    :param n_cca_dims:
    :return: sklearn cca object
    """
    # preprocess data
    # initialize sklearn cca object
    # fit cca
    pass


def get_cca_projection(cca, array_0, array_1):
    """
    Project data into CCA dimensions

    :param cca:
    :param array_0:
    :param array_1:
    :return:
    """
    pass


def get_correlations(cca, array_0, array_1):
    """

    :param cca:
    :param array_0:
    :param array_1:
    :return:
    """
    pass


def shuffle_analysis(array_0, array_1, n_shuffles=100, **cca_kwargs):
    """
    Perform CCA on shuffled data

    :param array_0:
    :param array_1:
    :param n_shuffles:
    :return:
    """
    pass


def plot_correlations(corrs, shuffled=None):
    """
    Correlation vs CCA dimension

    :param corrs:
    :param shuffled:
    :return:
    """
    pass


if __name__ == '__main__':
    pass

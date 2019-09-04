import numpy as np


def preprocess(array, smoothing_sd=25, n_pca_dims=20):
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
    # smooth data
    # pca
    pass


def fit_cca(array_0, array_1, n_cca_dims=10):
    """

    :param array_0:
    :param array_1:
    :param n_cca_dims:
    :return: sklearn cca object
    """
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

    from pathlib import Path
    from oneibl.one import ONE
    import alf.io as ioalf
    from brainbox.processing import bincount2D

    BIN_SIZE = 0.025  # seconds
    SMOOTH_SIZE = 0.025  # seconds; standard deviation of gaussian kernel
    PCA_DIMS = 20
    CCA_DIMS = PCA_DIMS

    # get the data from flatiron
    subject = 'KS005'
    date = '2019-08-30'
    number = 1

    one = ONE()
    eid = one.search(subject=subject, date=date, number=number)
    D = one.load(eid[0], download_only=True)
    session_path = Path(D.local_path[0]).parent
    spikes = ioalf.load_object(session_path, 'spikes')
    # clusters = ioalf.load_object(session_path, 'clusters')
    # channels = ioalf.load_object(session_path, 'channels')
    # trials = ioalf.load_object(session_path, '_ibl_trials')

    # bin spikes
    binned_spikes, _, _ = bincount2D(spikes['times'], spikes['clusters'], BIN_SIZE)
    # extract 2 populations
    data = [binned_spikes[:100, :].T, binned_spikes[100:200, :].T]

    # preprocess data
    for pop in data:
        # TODO: DOES THIS WORK???
        pop = preprocess(pop, n_pca_dims=PCA_DIMS, smoothing_sd=SMOOTH_SIZE)

    # split trials
    idxs_trial = {'train': None, 'test': None}
    # get train/test indices into spike arrays
    idxs_time = {'train': None, 'test': None}

    # fit cca
    cca = fit_cca(
        data[0][idxs_time['train'], :], data[0][idxs_time['train'], :], n_cca_dims=CCA_DIMS)

    # plot cca correlations
    corrs = get_correlations(cca, data[0][idxs_time['test'], :], data[0][idxs_time['test'], :])
    plot_correlations(corrs)

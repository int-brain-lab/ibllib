import numpy as np
import matplotlib.pylab as plt



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


def plot_correlations(corrs, errors=None, ax=None, **plot_kwargs):
    """
    Correlation vs CCA dimension

    :param corrs: correlation values for the CCA dimensions
    :type corrs: 1-D vector
    :param errors: error values
    :type shuffled: 1-D array of size len(corrs)
    :param ax: axis to plot on (default None)
    :type ax: matplotlib axis object
    :return: axis if specified, or plot if axis = None
    """
    # evaluate if np.arrays are passed
    assert type(corrs) is np.ndarray, "'corrs' is not a numpy array."
    if errors is not None:
        assert type(errors) is np.ndarray, "'errors' is not a numpy array."
    # create axis if no axis is passed
    if ax is None:
        ax = plt.gca()
    # get the data for the x and y axis
    y_data = corrs
    x_data = range(1, (len(corrs) + 1))
    # create the plot object
    cplot = ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data-errors, y_data+errors, **plot_kwargs, alpha=0.2)
    # change y and x labels and ticks
    ax.set_xticks(x_data)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("CCA dimension")
    return cplot


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

import numpy as np
import matplotlib.pylab as plt
from brainbox.processing import bincount2D


def _smooth(data, sd):
    from scipy.signal import gaussian
    from scipy.signal import convolve
    n_bins = data.shape[0]
    w = n_bins - 1 if n_bins % 2 == 0 else n_bins
    window = gaussian(w, std=sd)
    for j in range(data.shape[1]):
        data[:, j] = convolve(data[:, j], window, mode='same', method='auto')
    return data


def _pca(data, n_pcs):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_pcs)
    pca.fit(data)
    data_pc = pca.transform(data)
    return data_pc


def preprocess(data, smoothing_sd=25, n_pcs=20):
    """
    Preprocess neural data for cca analysis with smoothing and pca

    :param data: array of shape (n_samples, n_features)
    :type data: array-like
    :param smoothing_sd: gaussian smoothing kernel standard deviation (ms)
    :type smoothing_sd: float
    :param n_pcs: number of pca dimensions to retain
    :type n_pcs: int
    :return: preprocessed neural data
    :rtype: array-like, shape (n_samples, pca_dims)
    """
    if smoothing_sd > 0:
        data = _smooth(data, sd=smoothing_sd)
    if n_pcs > 0:
        data = _pca(data, n_pcs=n_pcs)
    return data


def split_trials(trial_ids, n_splits=5, rng_seed=0):
    """
    Assign each trial to testing or training fold

    :param trial_ids:
    :type trial_ids: array-like
    :param n_splits: one split used for testing; remaining splits used for training
    :type n_splits: int
    :param rng_seed: set random state for shuffling trials
    :type rng_seed: int
    :return: list of dicts of indices with keys `train` and `test`
    """
    from sklearn.model_selection import KFold
    shuffle = True if rng_seed is not None else False
    kf = KFold(n_splits=n_splits, random_state=rng_seed, shuffle=shuffle)
    kf.get_n_splits(trial_ids)
    idxs = [None for _ in range(n_splits)]
    for i, t0 in enumerate(kf.split(trial_ids)):
        idxs[i] = {'train': t0[0], 'test': t0[1]}
    return idxs


def split_timepoints(trial_ids, idxs_trial):
    """
    Assign each time point to testing or training fold

    :param trial_ids: trial id for each timepoint
    :type trial_ids: array-like
    :param idxs_trial: list of dicts that define which trials are in `train` or `test` folds
    :type idxs_trial: list
    :return: list of dicts that define which time points are in `train` and `test` folds
    """
    idxs_time = [None for _ in range(len(idxs_trial))]
    for i, idxs in enumerate(idxs_trial):
        idxs_time[i] = {
            dtype: np.where(np.isin(trial_ids, idxs[dtype]))[0] for dtype in idxs.keys()}
    return idxs_time


def fit_cca(data_0, data_1, n_cca_dims=10):
    """
    Initialize and fit CCA sklearn object

    :param data_0: shape (n_samples, n_features_0)
    :type data_0: array-like
    :param data_1: shape (n_samples, n_features_1)
    :type data_1: array-like
    :param n_cca_dims: number of CCA dimensions to fit
    :type n_cca_dims: int
    :return: sklearn cca object
    """
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=n_cca_dims, max_iter=1000)
    cca.fit(data_0, data_1)
    return cca


def get_cca_projection(cca, data_0, data_1):
    """
    Project data into CCA dimensions

    :param cca:
    :param data_0:
    :param data_1:
    :return: tuple; (data_0 projection, data_1 projection)
    """
    x_scores, y_scores = cca.transform(data_0, data_1)
    return x_scores, y_scores


def get_correlations(cca, data_0, data_1):
    """

    :param cca:
    :param data_0:
    :param data_1:
    :return:
    """
    x_scores, y_scores = get_cca_projection(cca, data_0, data_1)
    corrs_tmp = np.corrcoef(x_scores.T, y_scores.T)
    corrs = np.diagonal(corrs_tmp, offset=data_0.shape[1])
    return corrs


def shuffle_analysis(data_0, data_1, n_shuffles=100, **cca_kwargs):
    """
    Perform CCA on shuffled data

    :param data_0:
    :param data_1:
    :param n_shuffles:
    :return:
    """
    # TODO
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
    ax.plot(x_data, y_data, **plot_kwargs)
    if errors is not None:
        ax.fill_between(x_data, y_data - errors, y_data + errors, **plot_kwargs, alpha=0.2)
    # change y and x labels and ticks
    ax.set_xticks(x_data)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("CCA dimension")
    return ax


def plot_pairwise_correlations(means, stderrs=None, n_dims=None, region_strs=None, **kwargs):
    """
    Plot CCA correlations for multiple pairs of regions
    
    :param means: list of lists; means[i][j] contains the mean corrs between regions i, j
    :param stderrs: list of lists; stderrs[i][j] contains std errors of corrs between regions i, j
    :param n_dims: number of CCA dimensions to plot
    :param region_strs: list of strings identifying each region
    :param kwargs: keyword arguments for plot
    :return: matplotlib figure handle
    """
    n_regions = len(means)

    fig, axes = plt.subplots(n_regions - 1, n_regions - 1, figsize=(12, 12))
    for r in range(n_regions - 1):
        for c in range(n_regions - 1):
            axes[r, c].axis('off')

    # get max correlation to standardize y axes
    max_val = 0
    for r in range(1, n_regions):
        for c in range(r):
            tmp = means[i][j]
            if tmp is not None:
                max_val = np.max([max_val, np.max(tmp)])

    for r in range(1, n_regions):
        for c in range(r):
            ax = axes[r - 1, c]
            ax.axis('on')
            ax = plot_correlations(means[i][j][:n_dims], stderrs[i][j][:n_dims], ax=ax, **kwargs)
            ax.axhline(y=0, xmin=0.05, xmax=0.95, linestyle='--', color='k')
            if region_strs is not None:
                ax.text(
                    x=0.95, y=0.95, s=str('%s-%s' % (region_strs[c], region_strs[r])),
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)
            ax.set_ylim([-0.05, max_val + 0.05])
            if not ax.is_first_col():
                ax.set_ylabel('')
                ax.set_yticks([])
            if not ax.is_last_row():
                ax.set_xlabel('')
                ax.set_xticks([])
    plt.tight_layout()
    plt.show()

    return fig


def bin_spikes_trials(spikes, trials, T_BIN=0.01):
    """
    Binarizes the spike times into a raster and assigns a trial number to each bin
    :param spikes: spikes object
    :type spikes: Bunch
    :param trials: trials object
    :type trials: Bunch
    :param T_BIN: size, in s, of the bins
    :type T_BIN: float
    :return: a matrix (SpikeCounts, bins), and a vector of bins size with trial ID
    """
    binned_spikes, bin_times, _ = bincount2D(spikes['times'], spikes['clusters'], T_BIN)
    trial_start_times = trials['intervals'][:, 0]
    binned_trialIDs = np.digitize(bin_times, trial_start_times)
    # correct, as index 0 is whatever happens before the first trial
    binned_trialIDs_corrected = binned_trialIDs - 1

    return binned_spikes, binned_trialIDs_corrected


def split_by_area(binned_spikes, cl_brainAcronyms, active_clusters, brain_areas):
    """
    This function converts a matrix of binned spikes into a list of matrices,
    with the clusters grouped by brain areas
    :param binned_spikes: binned spike data of shape (Clusters, timebins)
    :type binned_spikes: numpy.ndarray
    :param cl_brainAcronyms: brain region for each cluster
    :type cl_brainAcronyms: pandas.core.frame.DataFrame
    :param brain_areas: list of brain areas to select
    :type brain_areas: numpy.ndarray
    :param active_clusters: list of clusterIDs
    :type active_clusters: numpy.ndarray
    :return: list of numpy.ndarrays of size brain_areas
    """
    # TODO: check that this is doing what it is suppossed to!!!

    # TODO: check that input is as expected
    #   
    # initialize list
    listof_bs = []
    for b_area in brain_areas:
        # get the ids of clusters in the area
        cl_in_area = cl_brainAcronyms.loc[cl_brainAcronyms['brainAcronyms'] == b_area].index
        # get the indexes of the clusters that are in that area
        cl_idx_in_area = np.isin(active_clusters, cl_in_area)
        bs_in_area = binned_spikes[cl_idx_in_area, :]
        listof_bs.append(bs_in_area)
 
    return listof_bs


if __name__ == '__main__':

    from pathlib import Path
    from oneibl.one import ONE
    import alf.io as ioalf

    BIN_SIZE = 0.025  # seconds
    SMOOTH_SIZE = 0.025  # seconds; standard deviation of gaussian kernel
    PCA_DIMS = 20
    CCA_DIMS = PCA_DIMS
    N_SPLITS = 5
    RNG_SEED = 0

    # get the data from flatiron
    # subject = 'ZM_1735'
    # date = '2019-08-01'
    # number = 1

    # one = ONE()
    # eid = one.search(subject=subject, date=date, number=number)
    # D = one.load(eid[0], download_only=True)
    # session_path = Path(D.local_path[0]).parent
    # session_path = "/home/hmvergara/Downloads/FlatIron/mnt/s0/Data/Subjects/ZM_1735/2019-08-01/001/alf/"
    session_path = "/home/hmvergara/Downloads/FlatIron/KS005/2019-08-29/001/alf/"
    spikes = ioalf.load_object(session_path, 'spikes')
    clusters = ioalf.load_object(session_path, 'clusters')
    # channels = ioalf.load_object(session_path, 'channels')
    trials = ioalf.load_object(session_path, '_ibl_trials')

    # # bin spikes and get trial IDs associated with them
    # binned_spikes, binned_trialIDs = bin_spikes_trials(spikes, trials, T_BIN=0.01)

    # # extract 2 populations
    # data = [binned_spikes[:100, :].T, binned_spikes[100:200, :].T]

    # # preprocess data
    # for i, pop in enumerate(data):
    #     data[i] = preprocess(pop, n_pcs=PCA_DIMS, smoothing_sd=SMOOTH_SIZE)

    # # split trials
    # idxs_trial = split_trials(np.unique(binned_trialIDs), n_splits=N_SPLITS, rng_seed=RNG_SEED)
    # # get train/test indices into spike arrays
    # idxs_time = split_timepoints(binned_trialIDs, idxs_trial)

    # # fit cca
    # cca = fit_cca(
    #     data[0][idxs_time['train'], :], data[1][idxs_time['train'], :], n_cca_dims=CCA_DIMS)

    # # plot cca correlations
    # corrs = get_correlations(cca, data[0][idxs_time['test'], :], data[1][idxs_time['test'], :])
    # plot_correlations(corrs)

    ## Generate a matrix of correlations (biggest correlation found) between all pairwise areas
    # bin spikes and get trial IDs associated with them
    binned_spikes, binned_trialIDs = bin_spikes_trials(spikes, trials, T_BIN=0.01)
    # split trials
    idxs_trial = split_trials(np.unique(binned_trialIDs), n_splits=N_SPLITS, rng_seed=RNG_SEED)
    # get train/test indices into spike arrays
    idxs_time = split_timepoints(binned_trialIDs, idxs_trial)
    # Define areas
    brain_areas = np.unique(clusters.brainAcronyms)
    # [subset for testing] DELETE
    brain_areas = brain_areas[1:4]

    # Split/create populations
    # bin_spikes_trials does not return info for innactive clusters
    active_clusters = np.unique(spikes['clusters'])
    split_binned_spikes = split_by_area(binned_spikes, clusters.brainAcronyms,
                                        active_clusters, brain_areas)

    # Create empty "matrix" to store cca objects
    cca_mat = [[None for _ in range(len(brain_areas))] for _ in range(len(brain_areas))]

    # For each pair of populations:
    for i in range(len(brain_areas)):
        pop1 = split_binned_spikes[i].T
        for j in range(len(brain_areas)):
            if j > i:
                pop2 = split_binned_spikes[j].T
                # fit cca
                cca = fit_cca(
                    pop1[idxs_time['train'], :], pop2[idxs_time['train'], :], n_cca_dims=CCA_DIMS)
                # populate matrix
                cca_mat[i][j] = cca
                # print progress
                print("{} / {}".format(i, j))

    # Plot matrix

import numpy as np

from pathlib import Path
from oneibl.one import ONE
import alf.io as ioalf


def filter_trials(trials, choice, stim_side, stim_contrast):
    """
    Filter trials by choice and presented stimulus constrast/side

    :param trials: dict with keys `choice`, `contrastLeft`, `contrastRight`
    :type trials: dict
    :param choice: subject's choice (1=left, -1=right)
    :type choice: int
    :param stim_side: where stimulus was presented (1=left, -1=right)
    :type stim_side: int
    :param stim_contrast: contrast of grating stimulus
    :type stim_contrast: float
    :return: subset of trials filtered by choice and stimulus contrast/side
    :rtype: np.ndarray
    """
    contrast = 'contrastLeft' if stim_side == 1 else 'contrastRight'
    trial_ids = np.where(
        (trials['choice'] == choice) & (trials[contrast] == stim_contrast))[0]
    return trial_ids


def calculate_peths(
        spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
        post_time=0.5, bin_size=0.025, smoothing=0.025):
    """
    Calcluate peri-event time histograms; return means and standard deviations
    for each time point across specified clusters

    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating peths
    :type cluster_ids: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param bin_size: width of time windows (in seconds) to bin spikes
    :type bin_size: float
    :param smoothing: standard deviation (in seconds) of Gaussian kernel for
        smoothing peths; use `smoothing=0` to skip smoothing
    :type smoothing: float
    :return: (psth_means, psth_stds)
    :rtype: tuple with two elements, each of shape `(n_trials, n_clusters, n_bins)`
    """

    from scipy.signal import gaussian
    from scipy.signal import convolve

    # initialize containers
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)

    ids = np.unique(cluster_ids)

    # bin spikes
    for i, t_0 in enumerate(align_times):

        # define bin edges
        ts_pre = t_0 - np.arange(n_bins_pre, 0, -1) * bin_size
        ts_post = t_0 + np.arange(n_bins_post + 1) * bin_size
        ts = np.concatenate([ts_pre, ts_post])

        # filter spikes
        idxs = (spike_times > ts[0]) & \
               (spike_times <= ts[-1]) & np.isin(spike_clusters, cluster_ids)
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - ts[0]) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # smooth
        if smoothing > 0:
            for j in range(r.shape[0]):
                r[j, :] = convolve(r[j, :], window, mode='same', method='auto')

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

    # average
    peth_means = np.mean(binned_spikes, axis=0)
    peth_stds = np.std(binned_spikes, axis=0)

    return peth_means, peth_stds


def plot_peths(
        means, stds=None, marker_idx=0, bin_size=1.0, linewidth=1,
        onset_label='event', **kwargs):
    """
    Plot peths with optional errorbars

    :param means: peth means
    :type means: np.ndarray of shape `(n_clusters, n_bins)`
    :param stds: peth standard deviations
    :type stds: np.ndarray of shape `(n_clusters, n_bins)`
    :param marker_idx: index into bin dimension for drawing dashed line
    :type marker_idx: int
    :param bin_size: bin size of peth (in seconds) for labeling time axis
    :type bin_size: float
    :param linewidth: linewidth of peth means
    :type linewidth: float
    :param onset_label: name of alignment type, i.e. 'go cue', 'stimulus', etc.
        the x-axis is labeled as 'Time from `onset_label` onset (s)'
    :type onset_label: str
    :param kwargs: additional arguments into matplotlib.axis.Axes.plot function
    :return: figure handle
    :rtype: matplotlib figure handle
    """

    import matplotlib.pyplot as plt
    from ibllib.dsp import rms

    scale = 1.0 / rms(means.flatten())
    n_clusts, n_bins = means.shape
    ts = (np.arange(n_bins) - marker_idx) * bin_size

    fig = plt.figure(figsize=(3, 0.5 * n_clusts))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plot individual traces
    for i in range(n_clusts):
        # extract mean and offset
        ys = i + means[i, :] * scale
        # plot mean
        ax.plot(ts, ys, linewidth=linewidth, **kwargs)
        # plot standard error
        if stds is not None:
            ax.fill_between(
                ts, ys - stds[i, :] * scale, ys + stds[i, :] * scale,
                linewidth=linewidth, alpha=0.5)
    # plot vertical line at marker_idx (time = 0 seconds)
    ax.axvline(x=0, ymin=0.02, ymax=0.98, linestyle='--', color='k')
    # add labels
    ax.set_xlabel('Time from %s onset (s)' % onset_label, fontsize=12)
    ax.set_ylim([-0.5, n_clusts - 0.5])
    ax.set_yticks([])
    ax.set_yticklabels([])

    plt.show()

    return fig


if __name__ == '__main__':

    BIN_SIZE = 0.025  # seconds
    SMOOTH_SIZE = 0.025  # seconds; standard deviation of gaussian kernel
    PRE_TIME = 0.2  # seconds to plot before event onset
    POST_TIME = 0.5  # seconds to plot after event onset

    # get the data from flatiron
    one = ONE()
    eid = one.search(subject='ZM_1735', date='2019-08-01', number=1)
    D = one.load(eid[0], clobber=False, download_only=True)
    session_path = Path(D.local_path[0]).parent

    # load objects
    spikes = ioalf.load_object(session_path, 'spikes')
    trials = ioalf.load_object(session_path, '_ibl_trials')

    # filter trials by choice and contrast/side
    trial_ids = filter_trials(trials, choice=1, stim_contrast=1.0, stim_side=1)

    # align to go cue
    align_times = trials['goCue_times'][trial_ids]

    # define subset of clusters to plot
    cluster_ids = np.unique(spikes['clusters'])[np.arange(10)]

    # calculate peths of specified trials/clusters, aligned to desired event
    peth_means, peth_stds = calculate_peths(
        spikes['times'], spikes['clusters'], cluster_ids, align_times,
        pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
        smoothing=SMOOTH_SIZE)

    # plot peths along with standard error
    fig = plot_peths(
        peth_means, peth_stds / np.sqrt(len(trial_ids)),
        marker_idx=int(PRE_TIME / BIN_SIZE), bin_size=BIN_SIZE,
        onset_label='go cue')

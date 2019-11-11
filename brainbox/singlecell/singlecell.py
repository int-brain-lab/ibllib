'''
Computes properties of single-cells, e.g. the autocorrelation and firing rate.
'''

import numpy as np
from brainbox.population import xcorr


def acorr(spike_times, bin_size=None, window_size=None):
    """Compute the auto-correlogram of a neuron.

    Parameters
    ----------

    :param spike_times: Spike times in seconds.
    :type spike_times: array-like
    :param bin_size: Size of the bin, in seconds.
    :type bin_size: float
    :param window_size: Size of the window, in seconds.
    :type window_size: float

    Returns an `(winsize_samples,)` array with the auto-correlogram.

    """
    xc = xcorr(spike_times, np.zeros_like(spike_times), bin_size=bin_size, window_size=window_size)
    return xc[0, 0, :]


def firing_rate(spks, unit, t='all', hist_win=0.01, fr_win=0.5):
    '''
    Computes the instantaneous firing rate of a unit over time by computing a histogram of spike
    counts over a specified window of time `hist_win`, and summing this histogram over a sliding
    window of specified time `fr_win`, over a specified period of total time `t`.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    unit : int
        The unit number for which to calculate the firing rate.
    t : str or pair of floats (optional)
        The total time period for which the instantaneous firing rate is returned. Default: the
        time period from `unit`'s first to last spike (in s).
    hist_win : float (optional)
        The time window (in s) to use for computing spike counts.
    fr_win : float (optional)
        The time window (in s) to use as a moving slider to compute the instantaneous firing rate.

    Returns
    -------
    fr : ndarray
        The instantaneous firing rate over time (in hz).

    See Also
    --------
    metrics.firing_rate_coeff_var
    plot.firing_rate

    Examples
    --------
    1) Compute the instantaneous firing rate for unit1 from the time of its first to last spike,
    and compute the instantaneous firing rate for unit2 from the first to second minute.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch and calculate the firing rates.
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> fr = bb.singlecell.firing_rate(spks, 1)
        >>> fr_2 = bb.singlecell.firing_rate(spks, 2, t=[60, 120])
    '''

    # Get unit timestamps.
    unit_idxs = np.where(spks['clusters'] == unit)
    ts = ts = spks['times'][unit_idxs]
    if t != 'all':
        t_first = np.where(ts > t[0])[0][0]
        t_last = np.where(ts < t[1])[0][-1]
        ts = ts[t_first:t_last]
    # Compute histogram of spike counts.
    t_tot = ts[-1] - ts[0]
    n_bins_hist = np.int(t_tot / hist_win)
    counts = np.histogram(ts, n_bins_hist)[0]
    # Compute moving average of spike counts to get instantaneous firing rate in s.
    n_bins_fr = np.int(t_tot / fr_win)
    step_sz = np.int(len(counts) / n_bins_fr)
    fr = np.array([np.sum(counts[step:(step + step_sz)])
                   for step in range(len(counts) - step_sz)]) / fr_win
    return fr

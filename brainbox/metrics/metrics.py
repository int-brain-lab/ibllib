"""
Computes metrics for assessing quality of single units.

Run the following to set-up the workspace to run the docstring examples:
>>> import brainbox as bb
>>> import alf.io as aio
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import ibllib.ephys.spikes as e_spks
# (*Note, if there is no 'alf' directory, make 'alf' directory from 'ks2' output directory):
>>> e_spks.ks2_to_alf(path_to_ks_out, path_to_alf_out)
# Load the alf spikes bunch and clusters bunch, and get a units bunch.
>>> spks_b = aio.load_object(path_to_alf_out, 'spikes')
>>> clstrs_b = aio.load_object(path_to_alf_out, 'clusters')
>>> units_b = bb.processing.get_units_bunch(spks_b)  # may take a few mins to compute

TODO add spikemetrics as dependency?
TODO metrics that could be added: iso_dist, l_ratio, d_prime, nn_hit, nn_miss, sil
"""

import time

import numpy as np
import scipy.ndimage.filters as filters
import scipy.stats as stats

import brainbox as bb
from brainbox.core import Bunch
from brainbox.processing import bincount2D
from ibllib.io import spikeglx

# Parameters to be used in `quick_unit_metrics`
METRICS_PARAMS = {
    'presence_window': 10,
    'refractory_period': 0.0015,
    'min_isi': 0.0001,
    'spks_per_bin_for_missed_spks_est': 10,
    'std_smoothing_kernel_for_missed_spks_est': 4,
    'min_num_bins_for_missed_spks_est': 50,
}


def unit_stability(units_b, units=None, feat_names=['amps'], dist='norm', test='ks'):
    """
    Computes the probability that the empirical spike feature distribution(s), for specified
    feature(s), for all units, comes from a specific theoretical distribution, based on a specified
    statistical test. Also computes the coefficients of variation of the spike feature(s) for all
    units.

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : array-like (optional)
        A subset of all units for which to create the bar plot. (If `None`, all units are used)
    feat_names : list of strings (optional)
        A list of names of spike features that can be found in `spks` to specify which features to
        use for calculating unit stability.
    dist : string (optional)
        The type of hypothetical null distribution for which the empirical spike feature
        distributions are presumed to belong to.
    test : string (optional)
        The statistical test used to compute the probability that the empirical spike feature
        distributions come from `dist`.

    Returns
    -------
    p_vals_b : bunch
        A bunch with `feat_names` as keys, containing a ndarray with p-values (the probabilities
        that the empirical spike feature distribution for each unit comes from `dist` based on
        `test`) for each unit for all `feat_names`.
    cv_b : bunch
        A bunch with `feat_names` as keys, containing a ndarray with the coefficients of variation
        of each unit's empirical spike feature distribution for all features.

    See Also
    --------
    plot.feat_vars

    Examples
    --------
    1) Compute 1) the p-values obtained from running a one-sample ks test on the spike amplitudes
    for each unit, and 2) the variances of the empirical spike amplitudes distribution for each
    unit. Create a histogram of the variances of the spike amplitudes for each unit, color-coded by
    depth of channel of max amplitudes. Get cluster IDs of those units which have variances greater
    than 50.
        >>> p_vals_b, variances_b = bb.metrics.unit_stability(units_b)
        # Plot histograms of variances color-coded by depth of channel of max amplitudes
        >>> fig = bb.plot.feat_vars(units_b, feat_name='amps')
        # Get all unit IDs which have amps variance > 50
        >>> var_vals = np.array(tuple(variances_b['amps'].values()))
        >>> bad_units = np.where(var_vals > 50)
    """

    # Get units.
    if not (units is None):  # we're using a subset of all units
        unit_list = list(units_b[feat_names[0]].keys())
        # for each `feat` and unit in `unit_list`, remove unit from `units_b` if not in `units`
        for feat in feat_names:
            [units_b[feat].pop(unit) for unit in unit_list if not (int(unit) in units)]
    unit_list = list(units_b[feat_names[0]].keys())  # get new `unit_list` after removing units

    # Initialize `p_vals` and `variances`.
    p_vals_b = Bunch()
    cv_b = Bunch()

    # Set the test as a lambda function (in future, more tests can be added to this dict)
    tests = \
        {
            'ks': lambda x, y: stats.kstest(x, y)
        }
    test_fun = tests[test]

    # Compute the statistical tests and variances. For each feature, iteratively get each unit's
    # p-values and variances, and add them as keys to the respective bunches `p_vals_feat` and
    # `variances_feat`. After iterating through all units, add these bunches as keys to their
    # respective parent bunches, `p_vals` and `variances`.
    for feat in feat_names:
        p_vals_feat = Bunch((unit, 0) for unit in unit_list)
        cv_feat = Bunch((unit, 0) for unit in unit_list)
        for unit in unit_list:
            # If we're missing units/features, create a NaN placeholder and skip them:
            if len(units_b['times'][str(unit)]) == 0:
                p_val = np.nan
                cv = np.nan
            else:
                # compute p_val and var for current feature
                _, p_val = test_fun(units_b[feat][unit], dist)
                cv = np.var(units_b[feat][unit]) / np.mean(units_b[feat][unit])
            # Append current unit's values to list of units' values for current feature:
            p_vals_feat[str(unit)] = p_val
            cv_feat[str(unit)] = cv
        p_vals_b[feat] = p_vals_feat
        cv_b[feat] = cv_feat

    return p_vals_b, cv_b


def missed_spikes_est(feat, spks_per_bin=20, sigma=4, min_num_bins=50):
    """
    Computes the approximate fraction of spikes missing (i.e. a pseudo false negative estimate)
    from a spike feature distribution for a given unit, assuming the distribution is symmetric.

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705.

    Parameters
    ----------
    feat : ndarray
        The spikes' feature values (e.g. amplitudes)
    spks_per_bin : int (optional)
        The number of spikes per bin from which to compute the spike feature histogram.
    sigma : int (optional)
        The standard deviation for the gaussian kernel used to compute the pdf from the spike
        feature histogram.
    min_num_bins : int (optional)
        The minimum number of bins used to compute the spike feature histogram.

    Returns
    -------
    fraction_missing : float
        The fraction of missing spikes (0-0.5). *Note: If more than 50% of spikes are missing, an
        accurate estimate isn't possible.
    pdf : ndarray
        The computed pdf of the spike feature histogram.
    cutoff_idx : int
        The index for `pdf` at which point `pdf` is no longer symmetrical around the peak. (This
        is returned for plotting purposes).

    See Also
    --------
    plot.missed_spikes_est

    Examples
    --------
    1) Determine the fraction of spikes missing from unit 1 based on the recorded unit's spike
    amplitudes, assuming the distribution of the unit's spike amplitudes is symmetric.
        # Get unit 1 amplitudes from a unit bunch, and compute fraction spikes missing.
        >>> feat = units_b['amps']['1']
        >>> fraction_missing = bb.plot.missed_spikes_est(feat)
    """

    # Ensure minimum number of spikes requirement is met.
    error_str = 'The number of spikes in this unit is {0}, ' \
                'but it must be at least {1}'.format(feat.size, spks_per_bin * min_num_bins)
    assert (feat.size > (spks_per_bin * min_num_bins)), error_str

    # compute the spike feature histogram and pdf:
    num_bins = np.int(feat.size / spks_per_bin)
    hist, bins = np.histogram(feat, num_bins, density=True)
    pdf = filters.gaussian_filter1d(hist, sigma)

    # Find where the distribution stops being symmetric around the peak:
    peak_idx = np.argmax(pdf)
    max_idx_sym_around_peak = np.argmin(np.abs(pdf[peak_idx:] - pdf[0]))
    cutoff_idx = peak_idx + max_idx_sym_around_peak

    # compute fraction missing from the tail of the pdf (the area where pdf stops being
    # symmetric around peak).
    fraction_missing = np.sum(pdf[cutoff_idx:]) / np.sum(pdf)
    fraction_missing = 0.5 if (fraction_missing > 0.5) else fraction_missing

    return fraction_missing, pdf, cutoff_idx


def wf_similarity(wf1, wf2):
    """
    Computes a unit normalized spatiotemporal similarity score between two sets of waveforms.
    This score is based on how waveform shape correlates for each pair of spikes between the
    two sets of waveforms across space and time. The shapes of the arrays of the two sets of
    waveforms must be equal.

    Parameters
    ----------
    wf1 : ndarray
        An array of shape (#spikes, #samples, #channels).
    wf2 : ndarray
        An array of shape (#spikes, #samples, #channels).

    Returns
    -------
    s: float
        The unit normalized spatiotemporal similarity score.

    See Also
    --------
    io.extract_waveforms
    plot.single_unit_wf_comp

    Examples
    --------
    1) Compute the similarity between the first and last 100 waveforms for unit1, across the 20
    channels around the channel of max amplitude.
        # Get the channels around the max amp channel for the unit, two sets of timestamps for the
        # unit, and the two corresponding sets of waveforms for those two sets of timestamps.
        # Then compute `s`.
        >>> max_ch = clstrs_b['channels'][1]
        >>> if max_ch < 10:  # take only channels greater than `max_ch`.
        >>>     ch = np.arange(max_ch, max_ch + 20)
        >>> elif (max_ch + 10) > 385:  # take only channels less than `max_ch`.
        >>>    ch = np.arange(max_ch - 20, max_ch)
        >>> else:  # take `n_c_ch` around `max_ch`.
        >>>     ch = np.arange(max_ch - 10, max_ch + 10)
        >>> ts1 = units_b['times']['1'][:100]
        >>> ts2 = units_b['times']['1'][-100:]
        >>> wf1 = bb.io.extract_waveforms(path_to_ephys_file, ts1, ch)
        >>> wf2 = bb.io.extract_waveforms(path_to_ephys_file, ts2, ch)
        >>> s = bb.metrics.wf_similarity(wf1, wf2)

    TODO check `s` calculation:
    take median of waveforms
    xcorr all waveforms with median, and divide by autocorr of all waveforms
    profile
    for two sets of units: xcorr(cl1, cl2) / (sqrt autocorr(cl1) * autocorr(cl2))
    """

    # Remove warning for dividing by 0 when calculating `s` (this is resolved by using
    # `np.nan_to_num`)
    import warnings
    warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
    assert wf1.shape == wf2.shape, ('The shapes of the sets of waveforms are inconsistent ({})'
                                    '({})'.format(wf1.shape, wf2.shape))

    # Get number of spikes, samples, and channels of waveforms.
    n_spks = wf1.shape[0]
    n_samples = wf1.shape[1]
    n_ch = wf1.shape[2]

    # Create a matrix that will hold the similarity values of each spike in `wf1` to `wf2`.
    # Iterate over both sets of spikes, computing `s` for each pair.
    similarity_matrix = np.zeros((n_spks, n_spks))
    for spk1 in range(n_spks):
        for spk2 in range(n_spks):
            s_spk = \
                np.sum(np.nan_to_num(
                    wf1[spk1, :, :] * wf2[spk2, :, :] /
                    np.sqrt(wf1[spk1, :, :] ** 2 * wf2[spk2, :, :] ** 2))) / (n_samples * n_ch)
            similarity_matrix[spk1, spk2] = s_spk

    # Return mean of similarity matrix
    s = np.mean(similarity_matrix)
    return s


def firing_rate_cv(ts, hist_win=0.01, fr_win=0.5, n_bins=10):
    """
    Computes the coefficient of variation of the firing rate: the ratio of the standard
    deviation to the mean.

    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the firing rate.
    hist_win : float (optional)
        The time window (in s) to use for computing spike counts.
    fr_win : float (optional)
        The time window (in s) to use as a moving slider to compute the instantaneous firing rate.
    n_bins : int (optional)
        The number of bins in which to compute a coefficient of variation of the firing rate.

    Returns
    -------
    cv : float
        The mean coefficient of variation of the firing rate of the `n_bins` number of coefficients
        computed.
    cvs : ndarray
        The coefficients of variation of the firing for each bin of `n_bins`.
    fr : ndarray
        The instantaneous firing rate over time (in hz).

    See Also
    --------
    singlecell.firing_rate
    plot.firing_rate

    Examples
    --------
    1) Compute the coefficient of variation of the firing rate for unit 1 from the time of its
    first to last spike, and compute the coefficient of variation of the firing rate for unit 2
    from the first to second minute.
        >>> ts_1 = units_b['times']['1']
        >>> ts_2 = units_b['times']['2']
        >>> ts_2 = np.intersect1d(np.where(ts_2 > 60)[0], np.where(ts_2 < 120)[0])
        >>> cv, cvs, fr = bb.metrics.firing_rate_cv(ts_1)
        >>> cv_2, cvs_2, fr_2 = bb.metrics.firing_rate_cv(ts_2)
    """

    # Compute overall instantaneous firing rate and firing rate for each bin.
    fr = bb.singlecell.firing_rate(ts, hist_win=hist_win, fr_win=fr_win)
    bin_sz = np.int(fr.size / n_bins)
    fr_binned = np.array([fr[(b * bin_sz):(b * bin_sz + bin_sz)] for b in range(n_bins)])

    # Compute coefficient of variations of firing rate for each bin, and the mean c.v.
    cvs = np.std(fr_binned, axis=1) / np.mean(fr_binned, axis=1)
    # NaNs from zero spikes are turned into 0's
    # cvs[np.isnan(cvs)] = 0 nan's can happen if neuron doesn't spike in a bin
    cv = np.mean(cvs)

    return cv, cvs, fr


def firing_rate_fano_factor(ts, hist_win=0.01, fr_win=0.5, n_bins=10):
    '''
    Computes the fano factor of the firing rate: the ratio of the variance to the mean.
    (Almost identical to coeff. of variation)

    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the firing rate.
    hist_win : float
        The time window (in s) to use for computing spike counts.
    fr_win : float
        The time window (in s) to use as a moving slider to compute the instantaneous firing rate.
    n_bins : int (optional)
        The number of bins in which to compute a fano factor of the firing rate.

    Returns
    -------
    ff : float
        The mean fano factor of the firing rate of the `n_bins` number of factors
        computed.
    ffs : ndarray
        The fano factors of the firing for each bin of `n_bins`.
    fr : ndarray
        The instantaneous firing rate over time (in hz).

    See Also
    --------
    singlecell.firing_rate
    plot.firing_rate

    Examples
    --------
    1) Compute the fano factor of the firing rate for unit 1 from the time of its
    first to last spike, and compute the fano factor of the firing rate for unit 2
    from the first to second minute.
        >>> ts_1 = units_b['times']['1']
        >>> ts_2 = units_b['times']['2']
        >>> ts_2 = np.intersect1d(np.where(ts_2 > 60)[0], np.where(ts_2 < 120)[0])
        >>> ff, ffs, fr = bb.metrics.firing_rate_fano_factor(ts_1)
        >>> ff_2, ffs_2, fr_2 = bb.metrics.firing_rate_fano_factor(ts_2)
    '''

    # Compute overall instantaneous firing rate and firing rate for each bin.
    fr = bb.singlecell.firing_rate(ts, hist_win=hist_win, fr_win=fr_win)
    # this procedure can cut off data at the end, up to n_bins last timesteps
    bin_sz = np.int(fr.size / n_bins)
    fr_binned = np.array([fr[(b * bin_sz):(b * bin_sz + bin_sz)] for b in range(n_bins)])

    # Compute fano factor of firing rate for each bin, and the mean fano factor
    ffs = np.var(fr_binned, axis=1) / np.mean(fr_binned, axis=1)
    # ffs[np.isnan(ffs)] = 0 nan's can happen if neuron doesn't spike in a bin
    ff = np.mean(ffs)

    return ff, ffs, fr


def isi_viol(ts, rp=0.002):
    """
    Computes the fraction of isi violations for a unit.

    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the firing rate.
    rp : float (optional)
        The refractory period (in s).

    Returns
    -------
    frac_isi_viol : float
        The fraction of isi violations.
    n_isi_viol : int
        The number of isi violations.
    isis : ndarray
        The isis.

    See Also
    --------

    Examples
    --------
    1) Get the fraction of isi violations, the total number of isi violations, and the array of
    isis for unit 1.
        >>> unit_idxs = np.where(spks_b['clusters'] == 1)[0]
        >>> ts = spks_b['times'][unit_idxs]
        >>> frac_isi_viol, n_isi_viol, isi = bb.metrics.isi_viol(ts)
    """

    isis = np.diff(ts)
    v = np.where(isis < rp)[0]  # violations
    frac_isi_viol = len(v) / len(ts)
    return frac_isi_viol, len(v), isis


def max_drift(feat):
    """
    Computes the maximum drift (max - min) of a spike feature array.

    Parameters
    ----------
    feat : ndarray
        The spike feature values from which to compute the maximum drift.

    Returns
    -------
    md : float
        The maxmimum drift of the unit.

    See Also
    --------
    cum_drift

    Examples
    --------
    1) Get the maximum depth and amp drift for unit 1.
        >>> unit_idxs = np.where(spks_b['clusters'] == 1)[0]
        >>> depths = spks_b['depths'][unit_idxs]
        >>> amps = spks_b['amps'][unit_idxs]
        >>> depth_md = bb.metrics.max_drift(depths)
        >>> amp_md = bb.metrics.max_drift(amps)
    """

    md = np.max(feat) - np.min(feat)
    return md


def cum_drift(feat):
    """
    Computes the cumulative drift (normalized by the total number of spikes) of a spike feature
    array.

    Parameters
    ----------
    feat : ndarray
        The spike feature values from which to compute the maximum drift.

    Returns
    -------
    cd : float
        The cumulative drift of the unit.

    See Also
    --------
    max_drift

    Examples
    --------
    1) Get the cumulative depth drift for unit 1.
        >>> unit_idxs = np.where(spks_b['clusters'] == 1)[0]
        >>> depths = spks_b['depths'][unit_idxs]
        >>> amps = spks_b['amps'][unit_idxs]
        >>> depth_cd = bb.metrics.cum_drift(depths)
        >>> amp_cd = bb.metrics.cum_drift(amps)
    """

    cd = np.sum(np.abs(np.diff(feat))) / len(feat)
    return cd


def pres_ratio(ts, hist_win=10):
    """
    Computes the presence ratio of spike counts: the number of bins where there is at least one
    spike, over the total number of bins, given a specified bin width.

    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the presence ratio.
    hist_win : float (optional)
        The time window (in s) to use for computing the presence ratio.

    Returns
    -------
    pr : float
        The presence ratio.
    spks_bins : ndarray
        The number of spks in each bin.

    See Also
    --------
    plot.pres_ratio

    Examples
    --------
    1) Compute the presence ratio for unit 1, given a window of 10 s.
        >>> ts = units_b['times']['1']
        >>> pr, pr_bins = bb.metrics.pres_ratio(ts)
    """

    bins = np.arange(0, ts[-1] + hist_win, hist_win)
    spks_bins, _ = np.histogram(ts, bins)
    pr = len(np.where(spks_bins)[0]) / len(spks_bins)
    return pr, spks_bins


def ptp_over_noise(ephys_file, ts, ch, t=2.0, sr=30000, n_ch_probe=385, dtype='int16', offset=0,
                   car=True):
    """
    For specified channels, for specified timestamps, computes the mean (peak-to-peak amplitudes /
    the MADs of the background noise).

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    ts : ndarray_like
        The timestamps (in s) of the spikes.
    ch : ndarray_like
        The channels on which to extract the waveforms.
    t : numeric (optional)
        The time (in ms) of the waveforms to extract to compute the ptp.
    sr : int (optional)
        The sampling rate (in hz) that the ephys data was acquired at.
    n_ch_probe : int (optional)
        The number of channels of the recording.
    dtype: str (optional)
        The datatype represented by the bytes in `ephys_file`.
    offset: int (optional)
        The offset (in bytes) from the start of `ephys_file`.
    car: bool (optional)
        A flag to perform common-average-referencing before extracting waveforms.

    Returns
    -------
    ptp_sigma : ndarray
        An array containing the mean ptp_over_noise values for the specified `ts` and `ch`.

    Examples
    --------
    1) Compute ptp_over_noise for all spikes on 20 channels around the channel of max amplitude
    for unit 1.
        >>> ts = units_b['times']['1']
        >>> max_ch = max_ch = clstrs_b['channels'][1]
        >>> if max_ch < 10:  # take only channels greater than `max_ch`.
        >>>     ch = np.arange(max_ch, max_ch + 20)
        >>> elif (max_ch + 10) > 385:  # take only channels less than `max_ch`.
        >>>     ch = np.arange(max_ch - 20, max_ch)
        >>> else:  # take `n_c_ch` around `max_ch`.
        >>>     ch = np.arange(max_ch - 10, max_ch + 10)
        >>> p = bb.metrics.ptp_over_noise(ephys_file, ts, ch)
    """

    # Ensure `ch` is ndarray
    ch = np.asarray(ch)
    ch = ch.reshape((ch.size, 1)) if ch.size == 1 else ch

    # Get waveforms.
    wf = bb.io.extract_waveforms(ephys_file, ts, ch, t=t, sr=sr, n_ch_probe=n_ch_probe,
                                 dtype=dtype, offset=offset, car=car)

    # Initialize `mean_ptp` based on `ch`, and compute mean ptp of all spikes for each ch.
    mean_ptp = np.zeros((ch.size,))
    for cur_ch in range(ch.size, ):
        mean_ptp[cur_ch] = np.mean(np.max(wf[:, :, cur_ch], axis=1) -
                                   np.min(wf[:, :, cur_ch], axis=1))

    # Compute MAD for `ch` in chunks.
    s_reader = spikeglx.Reader(ephys_file)
    file_m = s_reader.data  # the memmapped array
    n_chunk_samples = 5e6  # number of samples per chunk
    n_chunks = np.ceil(file_m.shape[0] / n_chunk_samples).astype('int')
    # Get samples that make up each chunk. e.g. `chunk_sample[1] - chunk_sample[0]` are the
    # samples that make up the first chunk.
    chunk_sample = np.arange(0, file_m.shape[0], n_chunk_samples, dtype=int)
    chunk_sample = np.append(chunk_sample, file_m.shape[0])
    # Give time estimate for computing MAD.
    t0 = time.perf_counter()
    stats.median_absolute_deviation(file_m[chunk_sample[0]:chunk_sample[1], ch], axis=0)
    dt = time.perf_counter() - t0
    print('Performing MAD computation. Estimated time is {:.2f} mins.'
          ' ({})'.format(dt * n_chunks / 60, time.ctime()))
    # Compute MAD for each chunk, then take the median MAD of all chunks.
    mad_chunks = np.zeros((n_chunks, ch.size), dtype=np.int16)
    for chunk in range(n_chunks):
        mad_chunks[chunk, :] = stats.median_absolute_deviation(
            file_m[chunk_sample[chunk]:chunk_sample[chunk + 1], ch], axis=0, scale=1)
    print('Done. ({})'.format(time.ctime()))

    # Return `mean_ptp` over `mad`
    mad = np.median(mad_chunks, axis=0)
    ptp_sigma = mean_ptp / mad
    return ptp_sigma


def contamination_est(ts, rp=0.002):
    """
    An estimate of the contamination of the unit (i.e. a pseudo false positive measure) based on
    the number of spikes, number of isi violations, and time between the first and last spike.
    (see Hill et al. (2011) J Neurosci 31: 8699-8705).

    Parameters
    ----------
    ts : ndarray_like
        The timestamps (in s) of the spikes.
    rp : float (optional)
        The refractory period (in s).

    Returns
    -------
    ce : float
        An estimate of the fraction of contamination.

    See Also
    --------
    contamination_est2

    Examples
    --------
    1) Compute contamination estimate for unit 1.
        >>> ts = units_b['times']['1']
        >>> ce = bb.metrics.contamination_est(ts)
    """

    # Get number of spikes, number of isi violations, and time from first to final spike.
    n_spks = len(ts)
    n_isi_viol = np.sum(np.diff(ts) < rp)
    t = ts[-1] - ts[0]

    # `ce` is min of roots of solved quadratic equation.
    c = (t * n_isi_viol) / (2 * rp * n_spks ** 2)  # 3rd term in quadratic
    ce = np.min(np.abs(np.roots([-1, 1, c])))  # solve quadratic
    return ce


def contamination_est2(ts, min_time, max_time, rp=0.002, min_isi=0.0001):
    """
    An estimate of the contamination of the unit (i.e. a pseudo false positive measure) based on
    the number of spikes, number of isi violations, and time between the first and last spike.
    (see Hill et al. (2011) J Neurosci 31: 8699-8705).

    Modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz.

    Parameters
    ----------
    ts : ndarray_like
        The timestamps (in s) of the spikes.
    min_time : float
        The minimum time (in s) that a potential spike occurred.
    max_time : float
        The maximum time (in s) that a potential spike occurred.
    rp : float (optional)
        The refractory period (in s).
    min_isi : float (optional)
        The minimum interspike-interval (in s) for counting duplicate spikes.

    Returns
    -------
    ce : float
        An estimate of the contamination.
        A perfect unit has a ce = 0
        A unit with some contamination has a ce < 0.5
        A unit with lots of contamination has a ce > 1.0
    num_violations : int
        The total number of isi violations.

    See Also
    --------
    contamination_est

    Examples
    --------
    1) Compute contamination estimate for unit 1, with a minimum isi for counting duplicate
    spikes of 0.1 ms.
        >>> ts = units_b['times']['1']
        >>> ce = bb.metrics.contamination_est2(ts, min_isi=0.0001)
    """

    duplicate_spikes = np.where(np.diff(ts) <= min_isi)[0]

    ts = np.delete(ts, duplicate_spikes + 1)
    isis = np.diff(ts)

    num_spikes = ts.size
    num_violations = np.sum(isis < rp)
    violation_time = 2 * num_spikes * (rp - min_isi)
    total_rate = ts.size / (max_time - min_time)
    violation_rate = num_violations / violation_time
    ce = violation_rate / total_rate

    return ce, num_violations


def quick_unit_metrics(spike_clusters, spike_times, spike_amps, spike_depths,
                       params=METRICS_PARAMS):
    """
    Computes single unit metrics from only the spike times, amplitudes, and depths for a set of
    units.

    Metrics computed:
        num_spikes
        firing_rate
        presence_ratio
        presence_ratio_std
        frac_isi_viol (see `isi_viol`)
        contamination_est (see `contamination_est`)
        contamination_est2 (see `contamination_est2`)
        missed_spikes_est (see `missed_spikes_est`)
        cum_amp_drift (see `cum_drift`)
        max_amp_drift (see `max_drift`)
        cum_depth_drift (see `cum_drift`)
        max_depth_drift (see `max_drift`)

    Parameters
    ----------
    spike_clusters : ndarray_like
        A vector of the unit ids for a set of spikes.
    spike_times : ndarray_like
        A vector of the timestamps for a set of spikes.
    spike_amps : ndarray_like
        A vector of the amplitudes for a set of spikes.
    spike_depths : ndarray_like
        A vector of the depths for a set of spikes.
    params : dict (optional)
        Parameters used for computing some of the metrics in the function:
            'presence_window': float
                The time window (in s) used to look for spikes when computing the presence ratio.
            'refractory_period': float
                The refractory period used when computing isi violations and the contamination
                estimate.
            'min_isi': float
                The minimum interspike-interval (in s) for counting duplicate spikes when computing
                the contamination estimate.
            'spks_per_bin_for_missed_spks_est': int
                The number of spikes per bin used to compute the spike amplitude pdf for a unit,
                when computing the missed spikes estimate.
            'std_smoothing_kernel_for_missed_spks_est': float
                The standard deviation for the gaussian kernel used to compute the spike amplitude
                pdf for a unit, when computing the missed spikes estimate.
            'min_num_bins_for_missed_spks_est': int
                The minimum number of bins used to compute the spike amplitude pdf for a unit,
                when computing the missed spikes estimate.

    Returns
    -------
    r : bunch
        A bunch whose keys are the computed spike metrics.

    Notes
    -----
    This function is called by `ephysqc.unit_metrics_ks2` which is called by `spikes.ks2_to_alf`
    during alf extraction of an ephys dataset in the ibl ephys extraction pipeline.

    Examples
    --------
    1) Compute quick metrics from a ks2 output directory:
        >>> from ibllib.ephys.ephysqc import phy_model_from_ks2_path
        >>> m = phy_model_from_ks2_path(path_to_ks2_out)
        >>> cluster_ids = m.spike_clusters
        >>> ts = m.spike_times
        >>> amps = m.amplitudes
        >>> depths = m.depths
        >>> r = bb.metrics.quick_unit_metrics(cluster_ids, ts, amps, depths)
    """

    cluster_ids = np.arange(np.max(spike_clusters) + 1)
    nclust = cluster_ids.size
    r = Bunch({
        'cluster_id': cluster_ids,
        'num_spikes': np.full((nclust,), np.nan),
        'firing_rate': np.full((nclust,), np.nan),
        'presence_ratio': np.full((nclust,), np.nan),
        'presence_ratio_std': np.full((nclust,), np.nan),
        'frac_isi_viol': np.full((nclust,), np.nan),
        'contamination_est': np.full((nclust,), np.nan),
        'contamination_est2': np.full((nclust,), np.nan),
        'missed_spikes_est': np.full((nclust,), np.nan),
        'cum_amp_drift': np.full((nclust,), np.nan),
        'max_amp_drift': np.full((nclust,), np.nan),
        'cum_depth_drift': np.full((nclust,), np.nan),
        'max_depth_drift': np.full((nclust,), np.nan),
        # could add 'epoch_name' in future:
        # 'epoch_name': np.zeros(nclust, dtype='object'),
    })

    # vectorized computation of basic metrics such as presence ratio and firing rate
    tmin = spike_times[0]
    tmax = spike_times[-1]
    presence_ratio = bincount2D(spike_times, spike_clusters,
                                xbin=params['presence_window'],
                                ybin=cluster_ids, xlim=[tmin, tmax])[0]
    r.presence_ratio = np.sum(presence_ratio > 0, axis=1) / presence_ratio.shape[1]
    r.presence_ratio_std = np.std(presence_ratio, axis=1)
    r.num_spikes = np.sum(presence_ratio, axis=1)
    r.firing_rate = r.num_spikes / (tmax - tmin)

    # loop over each cluster to compute the rest of the metrics
    for ic in np.arange(nclust):
        # slice the spike_times array
        ispikes = spike_clusters == cluster_ids[ic]
        if np.all(~ispikes):  # if this cluster has no spikes, continue
            continue
        ts = spike_times[ispikes]
        amps = spike_amps[ispikes]
        depths = spike_depths[ispikes]

        # compute metrics
        r.frac_isi_viol[ic], _, _ = isi_viol(ts, rp=params['refractory_period'])
        r.contamination_est[ic] = contamination_est(ts, rp=params['refractory_period'])
        r.contamination_est2[ic], _ = contamination_est2(
            ts, tmin, tmax, rp=params['refractory_period'], min_isi=params['min_isi'])
        try:  # this may fail because `missed_spikes_est` requires a min number of spikes
            r.missed_spikes_est[ic], _, _ = missed_spikes_est(
                amps, spks_per_bin=params['spks_per_bin_for_missed_spks_est'],
                sigma=params['std_smoothing_kernel_for_missed_spks_est'],
                min_num_bins=params['min_num_bins_for_missed_spks_est'])
        except AssertionError:
            pass
        r.cum_amp_drift[ic] = cum_drift(amps)
        r.max_amp_drift[ic] = max_drift(amps)
        r.cum_depth_drift[ic] = cum_drift(depths)
        r.max_depth_drift[ic] = max_drift(depths)

    return r

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
"""
import brainbox as bb
import numpy as np
import scipy.stats as stats
import scipy.ndimage.filters as filters
# add spikemetrics as dependency?
# import spikemetrics as sm


def unit_stability(units_b, units=None, feat_names=['amps'], dist='norm', test='ks'):
    '''
    Computes the probability that the empirical spike feature distribution(s), for specified
    feature(s), for all units, comes from a specific theoretical distribution, based on a specified
    statistical test. Also calculates the variances of the spike feature(s) for all units.

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
        The statistical test used to calculate the probability that the empirical spike feature
        distributions come from `dist`.

    Returns
    -------
    p_vals_b : bunch
        A bunch with `feat_names` as keys, containing a ndarray with p-values (the probabilities
        that the empirical spike feature distribution for each unit comes from `dist` based on
        `test`) for each unit for all `feat_names`.
    variances_b : bunch
        A bunch with `feat_names` as keys, containing a ndarray with the variances of each unit's
        empirical spike feature distribution for all features.

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
    '''

    # Get units.
    if not(units is None):  # we're using a subset of all units
        unit_list = list(units_b[feat_names[0]].keys())
        # for each `feat` and unit in `unit_list`, remove unit from `units_b` if not in `units`
        for feat in feat_names:
            [units_b[feat].pop(unit) for unit in unit_list if not(int(unit) in units)]
    unit_list = list(units_b[feat_names[0]].keys())  # get new `unit_list` after removing units

    # Initialize `p_vals` and `variances`.
    p_vals_b = bb.core.Bunch()
    variances_b = bb.core.Bunch()

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
        p_vals_feat = bb.core.Bunch((unit, 0) for unit in unit_list)
        variances_feat = bb.core.Bunch((unit, 0) for unit in unit_list)
        for unit in unit_list:
            # If we're missing units/features, create a NaN placeholder and skip them:
            if len(units_b['times'][str(unit)]) == 0:
                p_val = np.nan
                var = np.nan
            else:
                # Calculate p_val and var for current feature
                _, p_val = test_fun(units_b[feat][unit], dist)
                var = np.var(units_b[feat][unit])
            # Append current unit's values to list of units' values for current feature:
            p_vals_feat[str(unit)] = p_val
            variances_feat[str(unit)] = var
        p_vals_b[feat] = p_vals_feat
        variances_b[feat] = variances_feat

    return p_vals_b, variances_b


def feat_cutoff(feat, spks_per_bin=20, sigma=5, min_num_bins=50):
    '''
    Computes the approximate fraction of spikes missing from a spike feature distribution for a
    given unit, assuming the distribution is symmetric.

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705.

    Parameters
    ----------
    feat : ndarray
        The spikes' feature values.
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
    plot.feat_cutoff

    Examples
    --------
    1) Determine the fraction of spikes missing from unit 1 based on the recorded unit's spike
    amplitudes, assuming the distribution of the unit's spike amplitudes is symmetric.
        # Get unit 1 amplitudes from a unit bunch, and calculate fraction spikes missing.
        >>> feat = units_b['amps']['1']
        >>> fraction_missing = bb.plot.feat_cutoff(feat)
    '''

    # Ensure minimum number of spikes requirement is met.
    error_str = 'The number of spikes in this unit is {0}, ' \
                'but it must be at least {1}'.format(feat.size, spks_per_bin * min_num_bins)
    assert (feat.size > (spks_per_bin * min_num_bins)), error_str

    # Calculate the spike feature histogram and pdf:
    num_bins = np.int(feat.size / spks_per_bin)
    hist, bins = np.histogram(feat, num_bins, density=True)
    pdf = filters.gaussian_filter1d(hist, sigma)

    # Find where the distribution stops being symmetric around the peak:
    peak_idx = np.argmax(pdf)
    max_idx_sym_around_peak = np.argmin(np.abs(pdf[peak_idx:] - pdf[0]))
    cutoff_idx = peak_idx + max_idx_sym_around_peak

    # Calculate fraction missing from the tail of the pdf (the area where pdf stops being
    # symmetric around peak).
    fraction_missing = np.sum(pdf[cutoff_idx:]) / np.sum(pdf)
    fraction_missing = 0.5 if (fraction_missing > 0.5) else fraction_missing

    return fraction_missing, pdf, cutoff_idx


def wf_similarity(wf1, wf2):
    '''
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

    TODO check `s` calculation
    '''

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
                    np.sqrt(wf1[spk1, :, :]**2 * wf2[spk2, :, :]**2))) / (n_samples * n_ch)
            similarity_matrix[spk1, spk2] = s_spk

    # Return mean of similarity matrix
    s = np.mean(similarity_matrix)
    return s


def firing_rate_coeff_var(ts, hist_win=0.01, fr_win=0.5, n_bins=10):
    '''
    Computes the coefficient of variation of the firing rate: the ratio of the standard
    deviation to the mean.

    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the firing rate.
    hist_win : float
        The time window (in s) to use for computing spike counts.
    fr_win : float
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
        >>> cv, cvs, fr = bb.metrics.firing_rate_coeff_var(ts_1)
        >>> cv_2, cvs_2, fr_2 = bb.metrics.firing_rate_coeff_var(ts_2)
    '''

    # Calculate overall instantaneous firing rate and firing rate for each bin.
    fr = bb.singlecell.firing_rate(ts, hist_win=hist_win, fr_win=fr_win)
    bin_sz = np.int(fr.size / n_bins)
    fr_binned = np.array([fr[(b * bin_sz):(b * bin_sz + bin_sz)] for b in range(n_bins)])

    # Calculate coefficient of variations of firing rate for each bin, and the mean c.v.
    cvs = np.std(fr_binned, axis=1) / np.mean(fr_binned, axis=1)
    cv = np.mean(cvs)

    return cv, cvs, fr


def isi_viol(ts, rp=0.002):
    '''
    Computes the fraction of isi violations for a unit.

    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the firing rate.
    rp : float
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
    '''

    isis = np.diff(ts)
    v = np.where(isis < rp)[0]  # violations
    frac_isi_viol = len(v) / len(ts)
    return frac_isi_viol, len(v), isis


def max_drift(depths):
    '''
    Computes the maximum drift of spikes in a unit.

    Parameters
    ----------
    depths : ndarray
        The spike depths from which to compute the maximum drift.

    Returns
    -------
    md : float
        The maxmimum drift of the unit (in mm).

    See Also
    --------

    Examples
    --------
    1) Get the maximum drift for unit 1.
        >>> unit_idxs = np.where(spks_b['clusters'] == 1)[0]
        >>> depths = spks_b['depths'][unit_idxs]
        >>> md = bb.metrics.max_drift(depths)
    '''

    md = np.max(depths) - np.min(depths)
    return md


def cum_drift(depths):
    '''
    Computes the cumulative drift (normalized by the total number of spikes) of spikes in a unit.

    Parameters
    ----------
    depths : ndarray
        The spike depths from which to compute the maximum drift.

    Returns
    -------
    md : float
        The maxmimum drift of the unit (in mm).

    See Also
    --------

    Examples
    --------
    1) Get the cumulative drift for unit 1.
        >>> unit_idxs = np.where(spks_b['clusters'] == 1)[0]
        >>> depths = spks_b['depths'][unit_idxs]
        >>> md = bb.metrics.cum_drift(depths)
    '''

    cd = np.sum(np.abs(np.diff(depths))) / len(depths)
    return cd

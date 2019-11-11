"""
Computes metrics for assessing quality of single units.
"""
import brainbox as bb
import numpy as np
import scipy.stats as stats
import scipy.ndimage.filters as filters
# add spikemetrics as dependency?
# import spikemetrics as sm


def unit_stability(spks, feat_names=['amps'], dist='norm', test='ks'):
    '''
    Computes the probability that the empirical spike feature distribution(s), for specified
    feature(s), for all units, comes from a specific theoretical distribution, based on a specified
    statistical test. Also calculates the variances of the spike feature(s) for all units.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    feat_names : list of strings (optional)
        A list of names of spike features that can be found in `spks` to specify which features to
        use for calculating unit stability.
    dist : string (optional)
        The type of hypothetical null distribution from which the empirical spike feature
        distributions are presumed to belong to.
    test : string (optional)
        The statistical test used to calculate the probability that the empirical spike feature
        distributions come from `dist`.

    Returns
    -------
    p_vals : bunch
        A bunch with `feat_names` as keys, containing a ndarray with p-values (the probabilities
        that the empirical spike feature distribution for each unit comes from `dist` based on
        `test`) for each unit for all `feat_names`.
    variances : bunch
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
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        # Load a spikes bunch and calculate unit stability:
        >>> spks = aio.load_object('path\\to\\ks_output', 'spikes')
        >>> p_vals, variances = bb.metrics.unit_stability(spks)
        # Plot histograms of variances color-coded by  depth of channel of max amplitudes
        >>> bb.plot.feat_vars(spks, feat_name='amps')
        # Get all unit IDs which have amps variance > 50
        >>> var_vals = np.array(tuple(variances['amps'].values()))
        >>> bad_units = np.where(var_vals > 50)
    '''

    # Get units bunch and number of units.
    units = bb.processing.get_units_bunch(spks, feat_names)
    num_units = np.max(spks['clusters']) + 1
    # Initialize `p_vals` and `variances`.
    p_vals = bb.core.Bunch()
    variances = bb.core.Bunch()
    # Set the test as a lambda function (in future, more tests can be added to this dict)
    tests = \
        {
            'ks': lambda x, y: stats.kstest(x, y)
        }
    test_fun = tests.get(test)
    # Compute the statistical tests and variances. For each feature, iteratively get each unit's
    # p-values and variances, and add them as keys to the respective bunches `p_vals_feat` and
    # `variances_feat`. After iterating through all units, add these bunches as keys to their
    # respective parent bunches, `p_vals` and `variances`.
    for feat in feat_names:
        p_vals_feat = bb.core.Bunch((str(unit), 0) for unit in np.arange(0, num_units))
        variances_feat = bb.core.Bunch((str(unit), 0) for unit in np.arange(0, num_units))
        unit = 0
        while unit < num_units:
            # If we're missing units/features, create a NaN placeholder and skip them:
            if units[feat][str(unit)].size == 0:
                p_val = np.nan
                var = np.nan
            else:
                # Calculate p_val and var for current feature
                _, p_val = test_fun(units[feat][str(unit)], dist)
                var = np.var(units[feat][str(unit)])
            # Append current unit's values to list of units' values for current feature:
            p_vals_feat[str(unit)] = p_val
            variances_feat[str(unit)] = var
            unit += 1
        p_vals[feat] = p_vals_feat
        variances[feat] = variances_feat
    return p_vals, variances


def feat_cutoff(spks, unit, feat_name='amps', spks_per_bin=20, sigma=5):
    '''
    Computes the approximate fraction of spikes missing from a spike feature distribution for a
    given unit, assuming the distribution is symmetric.

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    unit : int
        The unit number for the feature to plot.
    feat_name : string (optional)
        The spike feature to plot.
    spks_per_bin : int (optional)
        The number of spikes per bin from which to compute the spike feature histogram.
    sigma : int (optional)
        The standard deviation for the gaussian kernel used to compute the pdf from the spike
        feature histogram.

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
    1) Determine the fraction of spikes missing from a unit based on the recorded unit's spike
    amplitudes, assuming the distribution of the unit's spike amplitudes is symmetric.
        >>> import brainbox as bb
        >>> import alf.io as aio
        # Get a spikes bunch and calculate estimated fraction of missing spikes.
        >>> spks = aio.load_object('path\\to\\ks_output', 'spikes')
        >>> fraction_missing = bb.metrics.feat_cutoff(spks, 'amps', 1)
        # Plot histogram and pdf of the spike amplitude distribution.
        >>> bb.plot.feat_cutoff(spks, 1)
    '''

    min_num_bins = 50
    units = bb.processing.get_units_bunch(spks, [feat_name])
    feature = units[feat_name][str(unit)]
    error_str = 'The number of spikes in this unit is {0}, ' \
                'but it must be at least {1}'.format(feature.size, spks_per_bin * min_num_bins)
    assert (feature.size > (spks_per_bin * min_num_bins)), error_str

    # Calculate the spike feature histogram and pdf:
    num_bins = np.int(feature.size / spks_per_bin)
    hist, bins = np.histogram(feature, num_bins, density=True)
    pdf = filters.gaussian_filter1d(hist, sigma)

    # Find where the distribution stops being symmetric around the peak:
    peak_idx = np.argmax(pdf)
    max_idx_sym_around_peak = np.argmin(np.abs(pdf[peak_idx:] - pdf[0]))
    cutoff_idx = peak_idx + max_idx_sym_around_peak

    # Calculate fraction missing from the tail of the pdf (the area where pdf stops being
    # symmetric around peak)
    fraction_missing = np.sum(pdf[cutoff_idx:]) / np.sum(pdf)

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
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch, a clusters bunch, a units bunch, the channels around the max amp
        # channel for the unit, two sets of timestamps for the units, and the two corresponding
        # sets of waveforms for those two sets of timestamps. Then compute `s`.
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> clstrs = aio.load_object('path\\to\\alf_output', 'clusters')
        >>> max_ch = max_ch = clstrs['channels'][1]
        >>> ch = np.arange(max_ch - 10, max_ch + 10)
        >>> units = bb.processing.get_units_bunch(spks)
        >>> ts1 = units['times']['1'][:100]
        >>> ts2 = units['times']['1'][-100:]
        >>> wf1 = bb.io.extract_waveforms('path\\to\\ephys_bin_file', ts1, ch)
        >>> wf2 = bb.io.extract_waveforms('path\\to\\ephys_bin_file', ts2, ch)
        >>> s = bb.metrics.wf_similarity(wf1, wf2)
    '''

    # Remove warning for dividing by 0 when calculating `s` (this is resolved by using
    # `np.nan_to_num`)
    import warnings
    warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
    assert wf1.shape == wf2.shape, 'The shapes of the sets of waveforms are inconsistent'
    n_spks = wf1.shape[0]
    n_samples = wf1.shape[1]
    n_ch = wf1.shape[2]
    # Create a matrix which will hold the similarity values of each spike in `wf1` to `wf2`
    similarity_matrix = np.zeros((n_spks, n_spks))
    # Iterate over both sets of spikes, computing `s` for each pair
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


def firing_rate_coeff_var(spks, unit, t='all', hist_win=0.01, fr_win=0.5, n_bins=10):
    '''
    Computes the coefficient of variation of the firing rate: the ratio of the standard
    deviation to the mean.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    unit : int
        The unit number for which to calculate the firing rate.
    t : str or pair of floats (optional)
        The total time period for which the instantaneous firing rate is returned. Default: the
        time period from `unit`'s first to last spike.
    hist_win : float (optional)
        The time window (in s) to use for computing spike counts.
    fr_win : float (optional)
        The time window (in s) to use as a moving slider to compute the instantaneous firing rate.
    n_bins : int (optional)
        The number of bins in which to compute a coefficient of variation of the firing rate.

    Returns
    -------
    cv: float
        The mean coefficient of variation of the firing rate of the `n_bins` number of coefficients
        computed.
    cvs: ndarray
        The coefficients of variation of the firing for each bin of `n_bins`.
    fr: ndarray
        The instantaneous firing rate over time (in hz).

    See Also
    --------
    singlecell.firing_rate
    plot.firing_rate

    Examples
    --------
    1) Compute the coefficient of variation of the firing rate for unit1 from the time of its
    first to last spike, and compute the coefficient of variation of the firing rate for unit2 from
    the first to second minute.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch and calculate the firing rate.
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> cv, cvs, fr = metrics.firing_rate_coeff_var(spks, 1)
        >>> cv_2, cvs_2, fr_2 = metrics.firing_rate_coeff_var(spks, 2)
    '''

    fr = bb.singlecell.firing_rate(spks, unit, t=t, hist_win=hist_win, fr_win=fr_win)
    bin_sz = np.int(fr.size / n_bins)
    fr_binned = np.array([fr[(b * bin_sz):(b * bin_sz + bin_sz)] for b in range(n_bins)])
    cvs = np.std(fr_binned, axis=1) / np.mean(fr_binned, axis=1)
    cv = np.mean(cvs)
    return cv, cvs, fr

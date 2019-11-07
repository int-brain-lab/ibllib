"""
Contains metrics for assessing quality of single units. Currently requires that this single unit
data comes from kilosort (https://github.com/mouseLand/Kilosort2) output.

Todo: generalize to units that are obtained from any spike sorter.
"""

import brainbox as bb
import numpy as np
import scipy.stats as stats
import scipy.ndimage.filters as filters
## install via `pip install spikemetrics` or 
## `git clone https://github.com/SpikeInterface/spikemetrics`
#import spikemetrics as sm

def unit_stability(spks, features=['amps'], dist='norm', test='ks'):
    '''
    Determines the probability that the empirical spike feature distribution(s), for specified 
    feature(s), for all units, comes from a specific theoretical distribution, based on a specified 
    statistical test. Also calculates the variances of the spike feature(s) for all units.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for each unit.
    features : list of strings (optional)
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
        A bunch with `features` as keys, containing a ndarray with p-values (the probabilities that
        the empirical spike feature distribution for each unit comes from `dist` based on `test`)
        for each unit for all features.
    variances : bunch
        A bunch with `features` as keys, containing a ndarray with the variances of each unit's 
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
    units = bb.processing.get_units_bunch(spks, features)
    num_units = np.max(spks['clusters']) + 1
    # Initialize `p_vals` and `variances`.
    p_vals = bb.core.Bunch()
    variances = bb.core.Bunch()  
    # Set the test as a lambda function (in future, more tests can be added to this dict)
    tests = {
             'ks' : lambda x,y: stats.kstest(x,y)
             }
    test_fun = tests.get(test)
    # Compute the statistical tests and variances. For each feature, iteratively get each unit's
    # p-values and variances, and add them as keys to the respective bunches `p_vals_feat` and
    # `variances_feat`. After iterating through all units, add these bunches as keys to their
    # respective parent bunches, `p_vals` and `variances`.
    for feat in features:
        p_vals_feat = bb.core.Bunch((str(unit),0) for unit in np.arange(0,num_units))
        variances_feat = bb.core.Bunch((str(unit),0) for unit in np.arange(0,num_units))
        unit = 0
        while unit < num_units:
            # If we're missing units/features, create a NaN placeholder and skip them:
            if units[feat][str(unit)].size==0:
                p_val = np.nan
                var = np.nan
            else:
                # Calculate p_val and var for current feature
                _, p_val = test_fun(units[feat][str(unit)], dist)
                var = np.var(units[feat][str(unit)])
            # Append current unit's values to list of units' values for current feature:
            p_vals_feat[str(unit)] = p_val
            variances_feat[str(unit)] = var
            unit+=1
        p_vals[feat] = p_vals_feat
        variances[feat] = variances_feat
    return p_vals, variances


def feat_cutoff(spks, unit, feat_name='amps', **kwargs):
    ''' 
    Determines approximate fraction of spikes missing from a spike feature distribution for a 
    given unit, assuming the distribution is symmetric. 
    
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for each unit.
    unit : int
        The unit number for the feature to plot.
    feat_name : string (optional)
        The spike feature to plot.
    spks_per_bin : int (optional keyword arg)
        The number of spikes per bin from which to compute the spike feature histogram.
    sigma : int (optional keyword arg)
        The standard deviation for the gaussian kernel used to compute the pdf from the spike
        feature histogram.

    Returns
    -------
    fraction_missing : float
        The fraction of missing spikes (0-0.5). *Note: If more than 50% of spikes are missing, an
        accurate estimate isn't possible.

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
    import pdb
    pdb.set_trace()
    # Set keyword input args if given:
    default_args = {
                'spks_per_bin': 20,
                'sigma': 5
                }
    new_args = {**default_args, **kwargs}
    spks_per_bin = new_args['spks_per_bin']
    sigma = new_args['sigma']
    min_num_bins = 50
    units = bb.processing.get_units_bunch(spks, [feat_name])
    feature = units[feat_name][str(unit)]
    error_str = 'The number of spikes in this unit is {0}, ' \
                'but it must be at least {1}'.format(feature.size, spks_per_bin*min_num_bins)
    assert (feature.size>spks_per_bin*min_num_bins),error_str

    # Calculate the spike feature histogram and pdf:
    num_bins = np.int(feature.size / spks_per_bin)
    hist, bins = np.histogram(feature, num_bins, density=True)
    pdf = filters.gaussian_filter1d(hist, sigma)

    # Find where the distribution stops being symmetric around the peak:
    peak_idx = np.argmax(pdf)
    max_idx_sym_around_peak =  np.argmin(np.abs(pdf[peak_idx:] - pdf[0]))
    cutoff_idx = peak_idx + max_idx_sym_around_peak
    
    # Calculate fraction missing from the tail of the pdf (the area where pdf stops being 
    # symmetric around peak)
    fraction_missing = np.sum(pdf[cutoff_idx:]) / np.sum(pdf)

    return fraction_missing, cutoff_idx, pdf
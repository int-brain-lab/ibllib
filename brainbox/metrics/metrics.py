"""
Contains metrics for assessing quality of single units. Currently requires that this single unit
data comes from kilosort (https://github.com/mouseLand/Kilosort2) output.

Todo: generalize to units that are obtained from any spike sorter.
"""

import brainbox as bb
import numpy as np
import scipy.stats as stats
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
## install via `pip install spikemetrics` or 
## `git clone https://github.com/SpikeInterface/spikemetrics`
#import spikemetrics as sm

def unit_stability(spks, features=['amps'], dist='norm', test='ks'):
    '''
    Determines the probability that the empirical spike feature distribution(s) for specified 
    feature(s), for all the spikes in a unit, for all units, comes from a specific theoretical 
    distribution, based on a specificed statistical test. Also calculates the variances of this 
    same spike feature for all units.

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

    Examples
    --------
    1) Compute the p-values obtained from running a one-sample ks test on the spike
    amplitudes for each unit, and the variances of the empirical spike amplitudes distribution for
    each unit. Create a histogram of the variances of the spike amplitudes for each unit. Find the 
    cluster IDs of those units which have variances greater than 100.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        # Load a spikes bunch, get a units bunch, and get the depth of max channel for each unit:
        >>> spks = aio.load_object('path\\to\\ks_output', 'spikes')
        >>> units = bb.processing.get_units_bunch(spks)
        
        # Compute unit stability based on 'amps'.
        >>> p_vals, variances = bb.metrics.unit_stability(spks)
        # Plot histograms of variances in 20 evenly spaced bins:
        >>> var_vals = tuple(variances['amps'].values())
        >>> bins_variances = np.arange(0, np.nanmax(var_vals), (np.nanmax(var_vals) / 20))      
        >>> fig, ax = plt.subplots()
        >>> ax.hist(variances['amps'], bins_variances)
        >>> ax.set_title('Unit Amplitude Variances')
        >>> ax.set_ax.set_xlabel('Variances')
        >>> ax.set_ylabel('Counts')
        # Get all unit IDs which have amps variance > 100
        >>> variance_thresh = 100
        >>> bad_units = np.where(variances['amps'] > variance_thresh)
    '''

    # Compute units bunch.
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
        p_vals_feat = bb.core.Bunch((repr(unit),0) for unit in np.arange(0,num_units))
        variances_feat = bb.core.Bunch((repr(unit),0) for unit in np.arange(0,num_units))
        unit = 0
        while unit < num_units:
            # If we're missing units/features, create a NaN placeholder and skip them:
            if units[feat][repr(unit)].size==0:
                p_val = np.nan
                var = np.nan
                #continue
            else:
                # Calculate p_val and var for current feature
                _, p_val = test_fun(units[feat][repr(unit)], dist)
                var = np.var(units[feat][repr(unit)])
            # Append current unit's values to list of units' values for current feature:
            p_vals_feat[repr(unit)] = p_val
            variances_feat[repr(unit)] = var
            unit+=1

        p_vals[feat] = p_vals_feat
        variances[feat] = variances_feat
    return p_vals, variances


def feature_cutoff(feature, **kwargs): # num_histogram_bins=500, histogram_smoothing_value=3):
    ''' 
    Determines approximate fraction of spikes missing from a spike feature distribution, assuming
    the distribution is symmetric. 
    
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705.

    Parameters
    ----------
    feature : ndarray
        The spike feature distribution from which to calculate the fraction of missing spikes.
    num_bins : int (optional keyword arg)
        The number of bins from which to compute the spike feature histogram.
    sigma : int (optional keyword arg)
        The standard deviation for the gaussian kernel used to compute the pdf from the spike
        feature histogram.

    Returns
    -------
    fraction_missing : float
        The fraction of missing spikes (0-0.5). If more than 50% of spikes are missing, an accurate 
        estimate isn't possible.
    fig : figure
        A figure showing the pdf of the feature distribution, and the cutoff point (as a red 
        vertical line) where the distribution is no longer symmetric (the normalized sum of the 
        pdf beyond this line is what is considered as the fraction of missing spikes from the unit)
    
    Examples
    --------
    1) Determine the fraction of spikes missing from a unit based on the recorded unit's spike 
    amplitudes, assuming the distribution of the unit's spike amplitudes is symmetric.
        >>> import brainbox as bb
        >>> import alf.io as aio
        # Get a spikes bunch.
        >>> spks = aio.load_object('path\\to\\ks_output', 'spikes')
        # Get a units bunch
        >>> units = bb.processing.get_units_bunch(spks)
        # Compute feature cutoff for spike amplitudes for unit #4
        >>> fraction_missing, fig = bb.metrics.feature_cutoff(units['amps'][4])    
    '''
    # Set keyword input args if given:
    default_args = {
                    'num_bins': np.int(feature.size / 100),  # ~ 100 spikes/bin
                    'sigma': 5
                    }
    new_args = {**default_args, **kwargs}
    num_bins = new_args['num_bins']
    sigma = new_args['sigma']

    # Calculate the spike feature histogram and pdf:
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
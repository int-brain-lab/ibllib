"""
Plots metrics that assess quality of single units. Contains plotting functions for the output of
functions in the brainbox `metrics.py` module.
"""

import brainbox as bb
import numpy as np
import matplotlib.pyplot as plt

def plot_feat_vars(spks, feat_name='amps', cmap_name='coolwarm'):
    '''
    Plots the variances of a particular spike feature for all units as a bar plot, where each bar
    is color-coded corresponding to the depth of the max amplitude channel of the respective unit.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for each unit.
    feat_name : string (optional)
        The spike feature to plot.
    cmap_name : string (optional)
        The name of the colormap associated with the plot.

    Returns
    -------
    fig : figure
        A figure object containing the plot.

    See Also
    --------
    metrics.unit_stability

    Examples
    --------
    1) Create a bar plot of the variances of the spike amplitudes for each unit.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> bb.plot.plot_feat_vars(spks)
    '''
    
    # Get units bunch and calculate variances.
    units = bb.processing.get_units_bunch(spks)
    p_vals, variances = bb.metrics.unit_stability(spks, features=[feat_name])
    var_vals = tuple(variances['amps'].values())
    # Specify bad units (i.e. missing unit numbers from spike sorter output).
    num_units = np.max(spks['clusters']) + 1
    bad_units = np.where(np.isnan(var_vals))
    good_units = np.delete(np.arange(0,num_units), bad_units)
    # Get depth of max amplitude channel for each unit, and use 0 as a placeholder for `bad_units`.
    depths = [units['depths'][repr(unit)][0] for unit in good_units]
    depths = np.insert(depths, bad_units[0], 0)
    # Create unit normalized colormap based on `depths`.
    cmap = plt.cm.get_cmap(cmap_name)
    depths_norm = depths/np.max(depths)
    rgba = [cmap(depth) for depth in depths_norm]
    # Plot depth-color-coded bar plot of variances for `feature` for each unit.
    fig, ax = plt.subplots()
    ax.bar(x=np.arange(0,num_units), height=var_vals, color=rgba)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    max_depth = np.max(depths)
    cbar.set_ticks(cbar.get_ticks())  # must call `set_ticks` to call `set_ticklabels`
    cbar.set_ticklabels([0, max_depth*0.2, max_depth*0.4, max_depth*0.6, max_depth*0.8, max_depth])
    ax.set_title('{feat} variance'.format(feat=feat_name))
    ax.set_xlabel('unit number')
    ax.set_ylabel('variance')
    cbar.set_label('depth', rotation=0)
    return fig

def plot_feat_cutoff(spks, feat_name, unit, **kwargs):
    '''
    Plots the pdf of an estimated symmetric spike feature distribution, with a vertical cutoff line
    that indicates the approximate fraction of spikes missing from the distribution, assuming the
    true distribution is symmetric. 

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for each unit.
    feat_name : string (optional)
        The spike feature to plot.
    unit : int
        The unit number for the feature to plot.
    spks_per_bin : int (optional keyword arg)
        The number of spikes per bin from which to compute the spike feature histogram.
    sigma : int (optional keyword arg)
        The standard deviation for the gaussian kernel used to compute the pdf from the spike
        feature histogram.

    Returns
    -------
    fig : figure
        A figure object containing the plot.
    
    See Also
    --------
    metrics.feature_cutoff
    
    Examples
    --------
    1) Plot cutoff line indicating the fraction of spikes missing from a unit based on the recorded 
    unit's spike amplitudes, assuming the distribution of the unit's spike amplitudes is symmetric.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch, a units bunch, and plot feature cutoff for spike amplitudes for unit1
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> bb.plot.plot_feat_cutoff(spks, 'amps', 1)    
    '''
    
    # Set keyword input args if given:
    default_args = {
                'spks_per_bin': 20,
                'sigma': 5
                }
    new_args = {**default_args, **kwargs}
    spks_per_bin = new_args['spks_per_bin']
    sigma = new_args['sigma']
    # Calculate and plot the feature distribution histogram and pdf with symmetric cutoff:
    fraction_missing, cutoff_idx, pdf = \
        bb.metrics.feat_cutoff(spks, feat_name, unit, spks_per_bin=spks_per_bin, sigma=sigma)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    units = bb.processing.get_units_bunch(spks, [feat_name])
    feature = units[feat_name][str(unit)]
    num_bins = np.int(feature.size / spks_per_bin)
    ax[0].hist(feature, bins=num_bins)
    ax[0].set_xlabel('{0}'.format(feat_name))
    ax[0].set_ylabel('count')
    ax[0].set_title('histogram of {0} for unit{1}'.format(feat_name, str(unit)))
    ax[1].plot(pdf)
    ax[1].vlines(cutoff_idx, 0, np.max(pdf), colors='r')
    ax[1].set_xlabel('bin number')
    ax[1].set_ylabel('density')
    ax[1].set_title('cutoff of pdf at end of symmetry around peak\n' \
                    '(estimated {:.2f}% missing spikes)'.format(fraction_missing*100))
    return fig

def plot_feat_heatmap(spks, feat_name, ephys_data_path):
    '''
 
    '''
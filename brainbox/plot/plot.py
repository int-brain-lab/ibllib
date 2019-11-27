"""
Plots metrics that assess quality of single units. Some functions here generate plots for the
output of functions in the brainbox `metrics.py` module.
"""

import brainbox as bb
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from warnings import warn


def feat_vars(spks, feat_name='amps', dist='norm', test='ks', cmap_name='coolwarm'):
    '''
    Plots the variances of a particular spike feature for all units as a bar plot, where each bar
    is color-coded corresponding to the depth of the max amplitude channel of the respective unit.

    Parameters
    ----------
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    feat_name : string (optional)
        The spike feature to plot.
    dist : string (optional)
        The type of hypothetical null distribution from which the empirical spike feature
        distributions are presumed to belong to.
    test : string (optional)
        The statistical test used to calculate the probability that the empirical spike feature
        distributions come from `dist`.
    cmap_name : string (optional)
        The name of the colormap associated with the plot.

    Returns
    -------
    fig : figure
        A figure object containing the plot.
    var_vals : ndarray
        Contains the variances of `feat_name` for each unit.
    p_vals : ndarray
        Contains the probabilites that the distribution for `feat_name` for each unit comes from a
        `dist` distribution based on the `test` statistical test.

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
        >>> fig, var_vals, p_vals = bb.plot.feat_vars(spks)
    '''

    # Get units bunch and calculate variances.
    units = bb.processing.get_units_bunch(spks)
    p_vals, variances = bb.metrics.unit_stability(spks, feat_names=[feat_name],
                                                  dist=dist, test=test)
    var_vals = np.array(tuple(variances[feat_name].values()))
    p_vals = np.array(tuple(p_vals[feat_name].values()))
    # Specify bad units (i.e. missing unit numbers from spike sorter output).
    num_units = np.max(spks['clusters']) + 1
    bad_units = np.where(np.isnan(var_vals))
    good_units = np.delete(np.arange(0, num_units), bad_units)
    # Get depth of max amplitude channel for each unit, and use 0 as a placeholder for `bad_units`.
    depths = [units['depths'][repr(unit)][0] for unit in good_units]
    depths = np.insert(depths, bad_units[0], 0)
    # Create unit normalized colormap based on `depths`.
    cmap = plt.cm.get_cmap(cmap_name)
    depths_norm = depths / np.max(depths)
    rgba = [cmap(depth) for depth in depths_norm]
    # Plot depth-color-coded bar plot of variances for `feature` for each unit.
    fig, ax = plt.subplots()
    ax.bar(x=np.arange(0, num_units), height=var_vals, color=rgba)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    max_d = np.max(depths)
    cbar.set_ticks(cbar.get_ticks())  # must call `set_ticks` to call `set_ticklabels`
    cbar.set_ticklabels([0, max_d * 0.2, max_d * 0.4, max_d * 0.6, max_d * 0.8, max_d])
    ax.set_title('{feat} variance'.format(feat=feat_name))
    ax.set_xlabel('unit number')
    ax.set_ylabel('variance')
    cbar.set_label('depth', rotation=0)
    return fig, var_vals, p_vals


def feat_cutoff(spks, unit, feat_name='amps', spks_per_bin=20, sigma=5):
    '''
    Plots the pdf of an estimated symmetric spike feature distribution, with a vertical cutoff line
    that indicates the approximate fraction of spikes missing from the distribution, assuming the
    true distribution is symmetric.

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
    fig : figure
        A figure object containing the plot.
    fraction_missing : float
        The fraction of missing spikes (0-0.5). *Note: If more than 50% of spikes are missing, an
        accurate estimate isn't possible.

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
        >>> fig, fraction_missing = bb.plot.feat_cutoff(spks, 1)
    '''

    # Calculate and plot the feature distribution histogram and pdf with symmetric cutoff:
    fraction_missing, pdf, cutoff_idx = \
        bb.metrics.feat_cutoff(spks, unit, feat_name=feat_name,
                               spks_per_bin=spks_per_bin, sigma=sigma)
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
    ax[1].set_title('cutoff of pdf at end of symmetry around peak\n'
                    '(estimated {:.2f}% missing spikes)'.format(fraction_missing * 100))
    return fig, fraction_missing


def single_unit_wf_comp(ephys_file, spks, clstrs, unit, n_ch=20, ts1='start', ts2='end',
                        n_spks=100, sr=30000, n_ch_probe=385, dtype='int16', car=True,
                        col=['b', 'r']):
    '''
    Plots waveforms from a single unit across a specified number of channels between two separate
    time periods, after (optionally) common-average-referencing. In this way, waveforms can be
    compared to see if there is, e.g. drift during the recording.

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    clstrs : bunch
        A clusters bunch containing fields with unit information (e.g. mean amps, channel of max
        amplitude, etc...), used here to extract the channel of max amplitude for the given unit.
    unit : int
        The unit number for the waveforms to plot.
    n_ch : int (optional)
        The number of channels around the channel of max amplitude to plot.
    ts1 : array_like (optional)
        A set of timestamps for which to compare waveforms with `ts2`.
    ts2: array_like (optional)
        A set of timestamps for which to compare waveforms with `ts1`.
    n_spks: int (optional)
        The number of spikes to plot for each channel if `ts1` and `ts2` are kept as their defaults
    sr : int (optional)
        The sampling rate (in hz) that the ephys data was acquired at.
    n_ch_probe : int (optional)
        The number of channels of the recording.
    dtype: str (optional)
        The datatype represented by the bytes in `ephys_file`.
    car: bool (optional)
        A flag for whether or not to perform common-average-referencing before extracting waveforms
    col: list of strings or float arrays (optional)
        Two elements in the list, where each specifies the color the `ts1` and `ts2` waveforms
        will be plotted in, respectively.

    Returns
    -------
    fig : figure
        A figure object containing the plot.
    wf1 : ndarray
        The waveforms for the spikes in `ts1`: an array of shape (#spikes, #samples, #channels).
    wf2 : ndarray
        The waveforms for the spikes in `ts2`: an array of shape (#spikes, #samples, #channels).
    s : float
        The similarity score between the two sets of waveforms, calculated by
        `metrics.wf_similarity`

    See Also
    --------
    io.extract_waveforms

    Examples
    --------
    1) Compare first and last 100 spike waveforms for unit1, across 20 channels around the channel
    of max amplitude, and compare the first and last 50 spike waveforms for unit2, across 15
    channels around the mean.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch, a clusters bunch, and plot waveforms for unit1 across 20 channels.
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> clstrs = aio.load_object('path\\to\\alf_output', 'clusters')
        >>> fig, wf1, wf2 = bb.plot.single_unit_wf_comp('path\\to\\ephys_file', spks, clstrs, \
                                                        unit=1)
        # Get a units bunch, and plot waveforms for unit2 from the first to second minute
        # across 15 channels.
        >>> units = bb.processing.get_units_bunch(spks, ['times'])
        >>> ts1 = units['times']['2'][:50]
        >>> ts2 = units['times']['2'][-50:]
        >>> fig2, wf1_2, wf2_2 = bb.plot.single_unit_wf_comp('path\\to\\ephys_file', spks, \
                                                             clstrs, unit=1, ts1=ts1, ts2=ts2)
    '''

    # Take the first and last 200 timestamps by default.
    units = bb.processing.get_units_bunch(spks)
    ts1 = units['times'][str(unit)][0:n_spks] if ts1 == 'start' else ts1
    ts2 = units['times'][str(unit)][-(n_spks + 1):-1] if ts2 == 'end' else ts2
    # Get the channel of max amplitude and `n_ch` around it.
    max_ch = clstrs['channels'][unit]
    n_c_ch = n_ch // 2
    if max_ch < n_c_ch:  # take only channels greater than `max_ch`.
        ch = np.arange(max_ch, max_ch + n_ch)
    elif (max_ch + n_c_ch) > n_ch_probe:  # take only channels less than `max_ch`.
        ch = np.arange(max_ch - n_ch, max_ch)
    else:  # take `n_c_ch` around `max_ch`.
        ch = np.arange(max_ch - n_c_ch, max_ch + n_c_ch)
    # Extract the waveforms for these timestamps and compute similarity score.
    wf1 = bb.io.extract_waveforms(ephys_file, ts1, ch, sr=sr, n_ch_probe=n_ch_probe, dtype=dtype,
                                  car=True)
    wf2 = bb.io.extract_waveforms(ephys_file, ts2, ch, sr=sr, n_ch_probe=n_ch_probe, dtype=dtype,
                                  car=True)
    s = bb.metrics.wf_similarity(wf1, wf2)
    # Plot these waveforms against each other.
    fig, ax = plt.subplots(nrows=n_ch, ncols=2)  # left col is all waveforms, right col is mean
    for cur_ax, cur_ch in enumerate(ch):
        ax[cur_ax][0].plot(wf1[:, :, cur_ax].T, c=col[0])
        ax[cur_ax][0].plot(wf2[:, :, cur_ax].T, c=col[1])
        ax[cur_ax][1].plot(np.mean(wf1[:, :, cur_ax], axis=0), c=col[0])
        ax[cur_ax][1].plot(np.mean(wf2[:, :, cur_ax], axis=0), c=col[1])
        ax[cur_ax][0].set_ylabel('Ch {0}'.format(cur_ch))
    plt.legend(['1st spike set', '2nd spike set'])
    fig.suptitle('comparison of waveforms from two sets of spikes for unit {0} \
                 \n s = {1:.2f}'.format(unit, s))
    return fig, wf1, wf2, s


def amp_heatmap(ephys_file, spks, clstrs, unit, t='all', n_ch=20, sr=30000, n_ch_probe=385,
                dtype='int16', cmap_name='RdBu', car=True):
    '''
    Plots a heatmap of the normalized voltage values over space and time at the timestamps of a
    particular unit over a specified number of channels, after (optionally)
    common-average-referencing.

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    spks : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    clstrs : bunch
        A clusters bunch containing fields with unit information (e.g. mean amps, channel of max
        amplitude, etc...), used here to extract the channel of max amplitude for the given unit.
    unit : int
        The unit number for which to plot the amp heatmap.
    t : str or pair of floats (optional)
        The time period from which to get the spike amplitudes. Default: all spike amplitudes.
    n_ch: int (optional)
        The number of channels for which to plot the amp heatmap.
    sr : int (optional)
        The sampling rate (in hz) that the ephys data was acquired at.
    n_ch_probe : int (optional)
        The number of channels of the recording.
    dtype: str (optional)
        The datatype represented by the bytes in `ephys_file`.
    cmap_name : string (optional)
        The name of the colormap associated with the plot.
    car: bool (optional)
        A flag for whether or not to perform common-average-referencing before extracting waveforms

    Returns
    -------
    fig : figure
        A figure object containing the plot.
    v_vals_norm : ndarray
        The unit-normalized voltage values displayed in `fig`.

    Examples
    --------
    1) Plot a heatmap of the spike amplitudes across 20 channels around the channel of max
    amplitude for unit1.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch, a clusters bunch, and plot heatmap for unit1 across 20 channels.
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> clstrs = aio.load_object('path\\to\\alf_output', 'clusters')
        >>> bb.plot.amp_heatmap('path\\to\\ephys_file', spks, clstrs, unit=1)
    '''

    # Get memmapped array of `ephys_file`
    item_bytes = np.dtype(dtype).itemsize
    n_samples = op.getsize(ephys_file) // (item_bytes * n_ch_probe)
    file_m = np.memmap(ephys_file, shape=(n_samples, n_ch_probe), dtype=dtype, mode='r')
    # Get voltage values for each peak amplitude sample for `n_ch` around `max_ch`:
    # Get the channel of max amplitude and `n_ch` around it.
    max_ch = clstrs['channels'][unit]
    n_c_ch = n_ch // 2
    if max_ch < n_c_ch:  # take only channels greater than `max_ch`.
        ch = np.arange(max_ch, max_ch + n_ch)
    elif (max_ch + n_c_ch) > n_ch_probe:  # take only channels less than `max_ch`.
        ch = np.arange(max_ch - n_ch, max_ch)
    else:  # take `n_c_ch` around `max_ch`.
        ch = np.arange(max_ch - n_c_ch, max_ch + n_c_ch)
    unit_idxs = np.where(spks['clusters'] == unit)
    max_amp_samples = spks['samples'][unit_idxs]
    ts = spks['times'][unit_idxs]
    v_vals = file_m[np.ix_(max_amp_samples, ch)]
    if car:  # Compute and subtract temporal and spatial noise from `v_vals`.
        # Get subset of time (from first to last max amp sample)
        t_subset = np.arange(max_amp_samples[0], max_amp_samples[-1] + 1, dtype='int16')
        # Specify output arrays as `dtype='int16'`
        out_noise_t = np.zeros((len(t_subset),), dtype='int16')
        out_noise_s = np.zeros((len(ch),), dtype='int16')
        noise_t = np.median(file_m[np.ix_(t_subset, ch)], axis=1, out=out_noise_t)
        noise_s = np.median(file_m[np.ix_(t_subset, ch)], axis=0, out=out_noise_s)
        v_vals -= noise_t[max_amp_samples - max_amp_samples[0], None]
        v_vals -= noise_s[None, :]
    # Plot heatmap.
    v_vals_norm = (v_vals / np.max(abs(v_vals))).T
    fig, ax = plt.subplots()
    cbar_map = ax.imshow(v_vals_norm, cmap=cmap_name, aspect='auto',
                         extent=[ts[0], ts[-1], ch[0], ch[-1]], origin='lower')
    ax.set_yticks(np.arange(ch[0], ch[-1], 5))
    ax.set_ylabel('channel numbers')
    ax.set_xlabel('time (s)')
    ax.set_title('heatmap of voltage at unit{0} timestamps'.format(unit))
    cbar = fig.colorbar(cbar_map, ax=ax)
    cbar.set_label('V', rotation=90)
    return fig, v_vals_norm


def firing_rate(spks, unit, t='all', hist_win=0.01, fr_win=0.5, n_bins=10, show_fr_cv=True):
    '''
    Plots the instantaneous firing rate of a unit over time, and optionally overlays the value of
    the coefficient of variation of the firing rate for a specified number of bins.

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
        The number of bins in which to compute coefficients of variation of the firing rate.
    show_fr_cv : bool (optional)
        A flag for whether or not to compute and show the coefficients of variation of the firing
        rate for `n_bins`.

    Returns
    -------
    fig : figure
        A figure object containing the plot.
    fr: ndarray
        The instantaneous firing rate over time (in hz).
    cv: float
        The mean coefficient of variation of the firing rate of the `n_bins` number of coefficients
        computed. Can only be returned if `show_fr_cv` is True.
    cvs: ndarray
        The coefficients of variation of the firing for each bin of `n_bins`. Can only be returned
        if `show_fr_cv` is True.

    See Also
    --------
    metrics.firing_rate_coeff_var
    singecell.firing_rate

    Examples
    --------
    1) Plot the firing rate for unit1 from the time of its first to last spike, showing the cv
    of the firing rate for 10 evenly spaced bins, and plot the firing rate for unit2 from the first
    to second minute, without showing the cv.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch and calculate the firing rate.
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> fig, fr_1, cv_1, cvs_1 = bb.plot.firing_rate(spks, unit=1)
        >>> fig2, fr_2 = bb.plot.firing_rate(spks, unit=2, t=[60,120], show_fr_cv=False)
    '''
    fig, ax = plt.subplots()
    if not(show_fr_cv):  # compute just the firing rate
        fr = bb.singlecell.firing_rate(spks, unit, t=t, hist_win=hist_win, fr_win=fr_win)
    else:  # compute firing rate and coefficients of variation
        cv, cvs, fr = bb.metrics.firing_rate_coeff_var(spks, unit, t=t, hist_win=hist_win,
                                                       fr_win=fr_win, n_bins=n_bins)
    x = np.arange(fr.size) * hist_win
    ax.plot(x, fr)
    ax.set_title('Firing Rate for Unit {0}'.format(unit))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (s$^-1$)')

    if not(show_fr_cv):
        return fig, fr
    else:  # show coefficients of variation
        y_max = np.max(fr) * 1.05
        x_l = x[np.int(x.size / n_bins)]
        # Plot vertical lines separating plots into `n_bins`.
        [ax.vlines((x_l * i), 0, y_max, linestyles='dashed', linewidth=2)
         for i in range(1, n_bins)]
        # Plot text with cv of firing rate for each bin.
        [ax.text(x_l * (i + 1), y_max, 'cv={0:.2f}'.format(cvs[i]), fontsize=9, ha='right')
         for i in range(n_bins)]
        return fig, fr, cv, cvs


def peri_event_time_histogram(
        spike_times, spike_clusters, events, cluster_id,  # Everything you need for a basic plot
        t_before=0.2, t_after=0.5, bin_size=0.025, smoothing=0.025, as_rate=True,
        include_raster=False, n_rasters=None, error_bars='std', ax=None,
        pethline_kwargs={'color': 'blue', 'lw': 2},
        errbar_kwargs={'color': 'blue', 'alpha': 0.5},
        eventline_kwargs={'color': 'black', 'alpha': 0.5},
        raster_kwargs={'color': 'black', 'lw': 0.5}, **kwargs):
    """
    Plot peri-event time histograms, with the meaning firing rate of units centered on a given
    series of events. Can optionally add a raster underneath the PETH plot of individual spike
    trains about the events.

    Arguments:
        spike_times {array-like} -- Spike times (in seconds)
        spike_clusters {array-like} -- Cluster identities for each element of spikes
        events {array-like} -- Times to align the histogram(s) to
        cluster_id {int} -- Identity of the cluster for which to plot a PETH

    Keyword Arguments:
        t_before {float} -- Time before event to plot (default: {0.2})
        t_after {float} -- Time after event to plot (default: {0.5})
        bin_size {float} -- Width of bin for histograms (default: {0.025})
        smoothing {float} -- sigma of gaussian smoothing to use in histograms. (default: {0.025})
        as_rate {bool} -- Whether to use spike counts or rates in the plot (default: {False})
        include_raster {bool} -- Whether to put a raster below the PETH of individual spike trains
            (default: {False})
        n_rasters {int} -- If include_raster is True, the number of rasters to include. If None
            will default to plotting rasters around all provided events. (default: {None})
        error_bars {str} -- Defines which type of error bars to plot. Options are:
            -- 'std' for 1 standard deviation
            -- 'sem' for standard error of the mean
            -- 'none' for only plotting the mean value
            (default: {'std'})
        ax {matplotlib axes} -- If passed, the function will plot on the passed axes. Note: current
            behavior causes whatever was on the axes to be cleared before plotting!
            (default: {None})
        pethline_kwargs {dict} -- Dict containing line properties to define PETH plot line. Default
            is a blue line with weight of 2. Needs to have color. See matplotlib plot documentation
            for more options.
            (default: {'color': 'blue', 'lw': 2})
        errbar_kwargs {dict} -- Dict containing fill-between properties to define PETH error bars.
            Default is a blue fill with 50 percent opacity.. Needs to have color. See matplotlib
            fill_between documentation for more options.
            (default: {'color': 'blue', 'alpha': 0.5})
        eventline_kwargs {dict} -- Dict containing fill-between properties to define line at event.
            Default is a black line with 50 percent opacity.. Needs to have color. See matplotlib
            vlines documentation for more options.
            (default: {'color': 'black', 'alpha': 0.5})
        raster_kwargs {dict} -- Dict containing properties defining lines in the raster plot.
            Default is black lines with line width of 0.5. See matplotlib vlines for more options.
            (default: {'color': 'black', 'lw': 0.5})
    """

    # Check to make sure if we fail, we fail in an informative way
    if not len(spike_times) == len(spike_clusters):
        raise ValueError('Spike times and clusters are not of the same shape')
    if cluster_id not in np.unique(spike_clusters):
        raise ValueError('Cluster to plot was not found in spike clusters')
    if len(events) == 1:
        raise ValueError('Cannot make a PETH with only one event.')
    if error_bars not in ('std', 'sem', 'none'):
        raise ValueError('Invalid error bar type was passed.')

    # Compute peths
    peths, binned_spikes = bb.singlecell.calculate_peths(spike_times, spike_clusters, [cluster_id],
                                                         events, t_before, t_after, bin_size,
                                                         smoothing, as_rate)
    # Construct an axis object if none passed
    if ax is None:
        plt.figure()
        ax = plt.gca()
    # Plot the curve and add error bars
    mean = peths.means[0, :]
    ax.plot(peths.tscale, mean, **pethline_kwargs)
    if error_bars == 'std':
        bars = peths.stds[0, :]
    elif error_bars == 'sem':
        bars = peths.stds[0, :] / np.sqrt(len(events))
    else:
        bars = np.zeros_like(mean)
    if error_bars != 'none':
        ax.fill_between(peths.tscale, mean - bars, mean + bars, **errbar_kwargs)

    # Plot the event marker line. Extends to 5% higher than max value of means plus any error bar.
    plot_edge = (mean.max() + bars[mean.argmax()]) * 1.05
    ax.vlines(0., 0., plot_edge, **eventline_kwargs)
    # Set the limits on the axes to t_before and t_after. Either set the ylim to the 0 and max
    # values of the PETH, or if we want to plot a spike raster below, create an equal amount of
    # blank space below the zero where the raster will go.
    ax.set_xlim([-t_before, t_after])
    ax.set_ylim([-plot_edge if include_raster else 0., plot_edge])
    # Put y ticks only at min, max, and zero
    if mean.min() != 0:
        ax.set_yticks([0, mean.min(), mean.max()])
    else:
        ax.set_yticks([0., mean.max()])
    # Move the x axis line from the bottom of the plotting space to zero if including a raster,
    # Then plot the raster
    if include_raster:
        if n_rasters is None:
            n_rasters = len(events)
        if n_rasters > 60:
            warn("Number of raster traces is greater than 60. This might look bad on the plot.")
        ax.axhline(0., color='black')
        tickheight = plot_edge / len(events[:n_rasters])  # How much space per trace
        tickedges = np.arange(0., -plot_edge - 1e-5, -tickheight)
        clu_spks = spike_times[spike_clusters == cluster_id]
        for i, t in enumerate(events[:n_rasters]):
            idx = np.bitwise_and(clu_spks >= t - t_before, clu_spks <= t + t_after)
            event_spks = clu_spks[idx]
            ax.vlines(event_spks - t, tickedges[i + 1], tickedges[i], **raster_kwargs)
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes', y=0.75)
    else:
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s) after event')
    return ax

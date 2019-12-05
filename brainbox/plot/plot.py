"""
Plots metrics that assess quality of single units. Some functions here generate plots for the
output of functions in the brainbox `metrics.py` module.

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

import os.path as op
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import brainbox as bb


def feat_vars(units_b, units=None, feat_name='amps', dist='norm', test='ks', cmap_name='coolwarm',
              ax=None):
    '''
    Plots the variances of a particular spike feature for all units as a bar plot, where each bar
    is color-coded corresponding to the depth of the max amplitude channel of the respective unit.

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : array-like (optional)
        A subset of all units for which to create the bar plot. (If `None`, all units are used)
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
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
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
    1) Create a bar plot of the variances of the spike amplitudes for all units.
        >>> fig, var_vals, p_vals = bb.plot.feat_vars(units_b)
    '''

    # Get units.
    if not(units is None):  # we're using a subset of all units
        unit_list = list(units_b['depths'].keys())
        # For each unit in `unit_list`, remove unit from `units_b` if not in `units`
        [units_b['depths'].pop(unit) for unit in unit_list if not(int(unit) in units)]
    unit_list = list(units_b['depths'].keys())  # get new `unit_list` after removing units

    # Calculate variances for all units
    p_vals_b, variances_b = bb.metrics.unit_stability(
            units_b, units=units, feat_names=[feat_name], dist=dist, test=test)
    var_vals = np.array(tuple(variances_b[feat_name].values()))
    p_vals = np.array(tuple(p_vals_b[feat_name].values()))

    # Specify and remove bad units (i.e. missing unit numbers from spike sorter output).
    bad_units = np.where(np.isnan(var_vals))[0]
    if len(bad_units) > 0:
        [unit_list.pop(bad_unit) for bad_unit in bad_units]
        good_units = unit_list
    else:
        good_units = unit_list

    # Get depth of max amplitude channel for good units
    depths = np.asarray([np.mean(units_b['depths'][str(unit)]) for unit in good_units])

    # Create unit normalized colormap based on `depths`, sorted by depth.
    cmap = plt.cm.get_cmap(cmap_name)
    depths_norm = depths / np.max(depths)
    rgba = np.asarray([cmap(depth) for depth in np.sort(np.flip(depths_norm))])

    # Plot depth-color-coded h bar plot of variances for `feature` for each unit, where units are
    # sorted descendingly by depth along y-axis.
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    ax.barh(y=[int(unit) for unit in good_units], width=var_vals[np.argsort(depths)], color=rgba)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    max_d = np.max(depths)
    tick_labels = [int(max_d) * tick for tick in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    cbar.set_ticks(cbar.get_ticks())  # must call `set_ticks` to call `set_ticklabels`
    cbar.set_ticklabels(tick_labels)
    ax.set_title('{feat} variance'.format(feat=feat_name))
    ax.set_ylabel('unit number (sorted by depth)')
    ax.set_xlabel('variance')
    cbar.set_label('depth', rotation=0)

    return var_vals, p_vals


def feat_cutoff(feat, feat_name, unit, spks_per_bin=20, sigma=5, min_num_bins=50, ax=None):
    '''
    Plots the pdf of an estimated symmetric spike feature distribution, with a vertical cutoff line
    that indicates the approximate fraction of spikes missing from the distribution, assuming the
    true distribution is symmetric.

    Parameters
    ----------
    feat : ndarray
        The spikes' feature values.
    feat_name : string
        The spike feature to plot.
    unit : int
        The unit from which the spike feature distribution comes from.
    spks_per_bin : int (optional)
        The number of spikes per bin from which to compute the spike feature histogram.
    sigma : int (optional)
        The standard deviation for the gaussian kernel used to compute the pdf from the spike
        feature histogram.
    min_num_bins : int (optional)
        The minimum number of bins used to compute the spike feature histogram.
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
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
        >>> feat = units_b['amps']['1']
        >>> fraction_missing = bb.plot.feat_cutoff(feat, feat_name='amps', unit=1)
    '''

    # Calculate the feature distribution histogram and fraction of spikes missing.
    fraction_missing, pdf, cutoff_idx = \
        bb.metrics.feat_cutoff(feat, spks_per_bin, sigma, min_num_bins)
    
    # Plot.
    if ax is None:  # create two axes
        fig, ax = plt.subplots(nrows=1, ncols=2)
    if ax is None or len(ax) == 2:  # plot histogram and pdf on two separate axes
        num_bins = np.int(feat.size / spks_per_bin)
        ax[0].hist(feat, bins=num_bins)
        ax[0].set_xlabel('{0}'.format(feat_name))
        ax[0].set_ylabel('count')
        ax[0].set_title('histogram of {0} for unit{1}'.format(feat_name, str(unit)))
        ax[1].plot(pdf)
        ax[1].vlines(cutoff_idx, 0, np.max(pdf), colors='r')
        ax[1].set_xlabel('bin number')
        ax[1].set_ylabel('density')
        ax[1].set_title('cutoff of pdf at end of symmetry around peak\n'
                        '(estimated {:.2f}% missing spikes)'.format(fraction_missing * 100))
    else:  # just plot pdf
        ax = ax[0]
        ax.plot(pdf)
        ax.vlines(cutoff_idx, 0, np.max(pdf), colors='r')
        ax.set_xlabel('bin number')
        ax.set_ylabel('density')
        ax.set_title('cutoff of pdf at end of symmetry around peak\n'
                     '(estimated {:.2f}% missing spikes)'.format(fraction_missing * 100))

    return fraction_missing


def wf_comp(ephys_file, ts1, ts2, ch, sr=30000, n_ch_probe=385, dtype='int16', car=True,
            col=['b', 'r'], ax=None):
    '''
    Plots two different sets of waveforms across specified channels after (optionally)
    common-average-referencing. In this way, waveforms can be compared to see if there is, 
    e.g. drift during the recording, or if two units should be merged, or one unit should be split.

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    ts1 : array_like
        A set of timestamps for which to compare waveforms with `ts2`. 
    ts2: array_like
        A set of timestamps for which to compare waveforms with `ts1`.
    ch : array-like
        The channels to use for extracting and plotting the waveforms.
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
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
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
    metrics.wf_similarity

    Examples
    --------
    1) Compare first and last 100 spike waveforms for unit1, across 20 channels around the channel
    of max amplitude, and compare the waveforms in the first minute to the waveforms in the fourth
    minutes for unit2, across 10 channels around the mean.
        # Get first and last 100 spikes, and 20 channels around channel of max amp for unit 1:
        >>> ts1 = units_b['times']['1'][:100]
        >>> ts2 = units_b['times']['1'][-100:]
        >>> max_ch = clstrs_b['channels'][1]
        >>> if max_ch < n_c_ch:  # take only channels greater than `max_ch`.
        >>>     ch = np.arange(max_ch, max_ch + 20)
        >>> elif (max_ch + n_c_ch) > n_ch_probe:  # take only channels less than `max_ch`.
        >>>     ch = np.arange(max_ch - 20, max_ch)
        >>> else:  # take `n_c_ch` around `max_ch`.
        >>>     ch = np.arange(max_ch - 10, max_ch + 10)
        >>> wf1, wf2, s = bb.plot.wf_comp(path_to_ephys_file, ts1, ts2, ch)
        # Plot waveforms for unit2 from the first and fourth minutes across 10 channels.
        >>> ts = units_b['times']['2']
        >>> ts1_2 = ts[np.where(ts<60)[0]]
        >>> ts2_2 = ts[np.where(ts>180)[0][:len(ts1)]]
        >>> max_ch = clstrs_b['channels'][2]
        >>> if max_ch < n_c_ch:  # take only channels greater than `max_ch`.
        >>>     ch = np.arange(max_ch, max_ch + 10)
        >>> elif (max_ch + n_c_ch) > n_ch_probe:  # take only channels less than `max_ch`.
        >>>     ch = np.arange(max_ch - 10, max_ch)
        >>> else:  # take `n_c_ch` around `max_ch`.
        >>>     ch = np.arange(max_ch - 5, max_ch + 5)
        >>> wf1_2, wf2_2, s_2 = bb.plot.wf_comp(path_to_ephys_file, ts1_2, ts2_2, ch)
    '''

    # Extract the waveforms for these timestamps and compute similarity score.
    wf1 = bb.io.extract_waveforms(ephys_file, ts1, ch, sr=sr, n_ch_probe=n_ch_probe, dtype=dtype,
                                  car=car)
    wf2 = bb.io.extract_waveforms(ephys_file, ts2, ch, sr=sr, n_ch_probe=n_ch_probe, dtype=dtype,
                                  car=car)
    s = bb.metrics.wf_similarity(wf1, wf2)

    # Plot these waveforms against each other.
    n_ch = len(ch)
    if ax is None:
        fig, ax = plt.subplots(nrows=n_ch, ncols=2)  # left col is all waveforms, right col is mean
    for cur_ax, cur_ch in enumerate(ch):
        ax[cur_ax][0].plot(wf1[:, :, cur_ax].T, c=col[0])
        ax[cur_ax][0].plot(wf2[:, :, cur_ax].T, c=col[1])
        ax[cur_ax][1].plot(np.mean(wf1[:, :, cur_ax], axis=0), c=col[0])
        ax[cur_ax][1].plot(np.mean(wf2[:, :, cur_ax], axis=0), c=col[1])
        ax[cur_ax][0].set_ylabel('Ch {0}'.format(cur_ch))
    ax[0][0].set_title('All Waveforms. S = {:.2f}'.format(s))
    ax[0][1].set_title('Mean Waveforms')
    plt.legend(['1st spike set', '2nd spike set'])

    return wf1, wf2, s


def amp_heatmap(ephys_file, ts, ch, sr=30000, n_ch_probe=385, dtype='int16', cmap_name='RdBu',
                car=True, ax=None):
    '''
    Plots a heatmap of the normalized voltage values over time and space for given timestamps and
    channels, after (optionally) common-average-referencing.

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    ts: array_like
        A set of timestamps for which to get the voltage values.
    ch : array-like
        The channels to use for extracting the voltage values.
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
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
    v_vals : ndarray
        The voltage values.

    Examples
    --------
    1) Plot a heatmap of the spike amplitudes across 20 channels around the channel of max
    amplitude for all spikes in unit 1.
        >>> ts = units_b['times']['1']
        >>> max_ch = clstrs_b['channels'][1]
        >>> if max_ch < n_c_ch:  # take only channels greater than `max_ch`.
        >>>     ch = np.arange(max_ch, max_ch + 20)
        >>> elif (max_ch + n_c_ch) > n_ch_probe:  # take only channels less than `max_ch`.
        >>>     ch = np.arange(max_ch - 20, max_ch)
        >>> else:  # take `n_c_ch` around `max_ch`.
        >>>     ch = np.arange(max_ch - 10, max_ch + 10)
        >>> bb.plot.amp_heatmap(path_to_ephys_file, ts, ch)
    '''

    # Get memmapped array of `ephys_file`
    item_bytes = np.dtype(dtype).itemsize
    n_samples = op.getsize(ephys_file) // (item_bytes * n_ch_probe)
    file_m = np.memmap(ephys_file, shape=(n_samples, n_ch_probe), dtype=dtype, mode='r')

    # Get voltage values for each peak amplitude sample for `ch`.
    max_amp_samples = (ts * sr).astype(int)
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
    if ax is None:
        ax = plt.gca()
    v_vals_norm = (v_vals / np.max(abs(v_vals))).T
    cbar_map = ax.imshow(v_vals_norm, cmap=cmap_name, aspect='auto',
                         extent=[ts[0], ts[-1], ch[0], ch[-1]], origin='lower')
    ax.set_yticks(np.arange(ch[0], ch[-1], 5))
    ax.set_ylabel('Channel Numbers')
    ax.set_xlabel('Time (s)')
    ax.set_title('Voltage Heatmap')
    fig = ax.figure
    cbar = fig.colorbar(cbar_map, ax=ax)
    cbar.set_label('V', rotation=90)

    return v_vals


def firing_rate(ts, hist_win=0.01, fr_win=0.5, n_bins=10, show_fr_cv=True, ax=None):
    '''
    Plots the instantaneous firing rate of for given spike timestamps over time, and optionally
    overlays the value of the coefficient of variation of the firing rate for a specified number
    of bins.

    Parameters
    ----------
    ts : ndarray
        The spike timestamps from which to compute the firing rate.
    hist_win : float (optional)
        The time window (in s) to use for computing spike counts.
    fr_win : float (optional)
        The time window (in s) to use as a moving slider to compute the instantaneous firing rate.
    n_bins : int (optional)
        The number of bins in which to compute coefficients of variation of the firing rate.
    show_fr_cv : bool (optional)
        A flag for whether or not to compute and show the coefficients of variation of the firing
        rate for `n_bins`.
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
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
    1) Plot the firing rate for unit 1 from the time of its first to last spike, showing the cv
    of the firing rate for 10 evenly spaced bins.
        >>> ts = units_b['times']['1']
        >>> fr, cv, cvs = bb.plot.firing_rate(ts)
    '''
    
    if ax is None:
        ax = plt.gca()
    if not(show_fr_cv):  # compute just the firing rate
        fr = bb.singlecell.firing_rate(ts, hist_win=hist_win, fr_win=fr_win)
    else:  # compute firing rate and coefficients of variation
        cv, cvs, fr = bb.metrics.firing_rate_coeff_var(ts, hist_win=hist_win, fr_win=fr_win,
                                                       n_bins=n_bins)
    x = np.arange(fr.size) * hist_win
    ax.plot(x, fr)
    ax.set_title('Firing Rate')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (s$^-1$)')

    if not(show_fr_cv):
        return fr
    else:  # show coefficients of variation
        y_max = np.max(fr) * 1.05
        x_l = x[np.int(x.size / n_bins)]
        # Plot vertical lines separating plots into `n_bins`.
        [ax.vlines((x_l * i), 0, y_max, linestyles='dashed', linewidth=2)
         for i in range(1, n_bins)]
        # Plot text with cv of firing rate for each bin.
        [ax.text(x_l * (i + 1), y_max, 'cv={0:.2f}'.format(cvs[i]), fontsize=9, ha='right')
         for i in range(n_bins)]
        return fr, cv, cvs


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

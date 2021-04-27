import numpy as np
from matplotlib import cm

from brainbox.plot_base import (ImagePlot, ScatterPlot, ProbePlot, LinePlot, plot_line,
                                plot_image, plot_probe, plot_scatter, arrange_channels2banks)
from brainbox.processing import bincount2D, compute_cluster_average


def image_lfp_spectrum_plot(lfp_power, lfp_freq, chn_coords, chn_inds, freq_range=(0, 300),
                            avg_across_depth=False, cmap='viridis', display=False):
    """
    Prepare data for 2D image plot of LFP power spectrum along depth of probe

    :param lfp_power:
    :param lfp_freq:
    :param chn_coords:
    :param chn_inds:
    :param freq_range:
    :param avg_across_depth: Whether to average across channels at same depth
    :param cmap:
    :param display: generate figure
    :return: ImagePlot object, if display=True also returns matplotlib fig and ax objects
    """

    freq_idx = np.where((lfp_freq >= freq_range[0]) & (lfp_freq < freq_range[1]))[0]
    freqs = lfp_freq[freq_idx]
    lfp = np.take(lfp_power[freq_idx], chn_inds, axis=1)
    lfp_db = 10 * np.log10(lfp)
    lfp_db[np.isinf(lfp_db)] = np.nan
    x = freqs
    y = chn_coords[:, 1]

    # Average across channels that are at the same depth
    if avg_across_depth:
        chn_depth, chn_idx, chn_count = np.unique(chn_coords[:, 1], return_index=True,
                                                  return_counts=True)
        chn_idx_eq = np.copy(chn_idx)
        chn_idx_eq[np.where(chn_count == 2)] += 1

        lfp_db = np.apply_along_axis(lambda a: np.mean([a[chn_idx], a[chn_idx_eq]], axis=0), 1,
                                     lfp_db)

        x = freqs
        y = chn_depth

    data = ImagePlot(lfp_db, x=x, y=y, cmap=cmap)
    data.set_labels(title='LFP Power Spectrum', xlabel='Frequency (Hz)',
                    ylabel='Distance from probe tip (um)', clabel='LFP Power (dB)')
    data.set_clim(clim=np.quantile(lfp_db, [0.1, 0.9]))

    if display:
        fig, ax = plot_image(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def image_rms_plot(rms_amps, rms_times, chn_coords, chn_inds, avg_across_depth=False,
                   median_subtract=True, cmap='plasma', band='AP', display=False):
    """
    Prepare data for 2D image plot of RMS data along depth of probe

    :param rms_amps:
    :param rms_times:
    :param chn_coords:
    :param chn_inds:
    :param avg_across_depth: Whether to average across channels at same depth
    :param median_subtract: Whether to apply median subtraction correction
    :param cmap:
    :param band: Frequency band of rms data, can be either 'LF' or 'AP'
    :param display: generate figure
    :return: ImagePlot object, if display=True also returns matplotlib fig and ax objects
    """

    rms = rms_amps[:, chn_inds] * 1e6
    x = rms_times
    y = chn_coords[:, 1]

    if avg_across_depth:
        chn_depth, chn_idx, chn_count = np.unique(chn_coords[:, 1], return_index=True,
                                                  return_counts=True)
        chn_idx_eq = np.copy(chn_idx)
        chn_idx_eq[np.where(chn_count == 2)] += 1
        rms = np.apply_along_axis(lambda a: np.mean([a[chn_idx], a[chn_idx_eq]], axis=0), 1, rms)
        y = chn_depth

    if median_subtract:
        median = np.mean(np.apply_along_axis(lambda a: np.median(a), 1, rms))
        rms = np.apply_along_axis(lambda a: a - np.median(a), 1, rms) + median

    data = ImagePlot(rms, x=x, y=y, cmap=cmap)
    data.set_labels(title=f'{band} RMS', xlabel='Time (s)',
                    ylabel='Distance from probe tip (um)', clabel=f'{band} RMS (uV)')
    data.set_clim(clim=np.quantile(rms, [0.1, 0.9]))

    if display:
        fig, ax = plot_image(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def scatter_raster_plot(spike_amps, spike_depths, spike_times, n_amp_bins=10, cmap='BuPu',
                        subsample_factor=100, display=False):
    """
    Prepare data for 2D raster plot of spikes with colour and size indicative of spike amplitude

    :param spike_amps:
    :param spike_depths:
    :param spike_times:
    :param n_amp_bins: no. of colour and size bins into which to split amplitude data
    :param cmap:
    :param subsample_factor: factor by which to subsample data when too many points for efficient
    display
    :param display: generate figure
    :return: ScatterPlot object, if display=True also returns matplotlib fig and ax objects
    """

    amp_range = np.quantile(spike_amps, [0, 0.9])
    amp_bins = np.linspace(amp_range[0], amp_range[1], n_amp_bins)
    color_bin = np.linspace(0.0, 1.0, n_amp_bins + 1)
    colors = (cm.get_cmap(cmap)(color_bin)[np.newaxis, :, :3][0])

    spike_amps = spike_amps[0:-1:subsample_factor]
    spike_colors = np.zeros((spike_amps.size, 3))
    spike_size = np.zeros(spike_amps.size)
    for iA in range(amp_bins.size):
        if iA == (amp_bins.size - 1):
            idx = np.where(spike_amps > amp_bins[iA])[0]
            # Make saturated spikes the darkest colour
            spike_colors[idx] = colors[-1]
        else:
            idx = np.where((spike_amps > amp_bins[iA]) & (spike_amps <= amp_bins[iA + 1]))[0]
            spike_colors[idx] = [*colors[iA]]

        spike_size[idx] = iA / (n_amp_bins / 8)

    data = ScatterPlot(x=spike_times[0:-1:subsample_factor], y=spike_depths[0:-1:subsample_factor],
                       c=spike_amps * 1e6, cmap='BuPu')
    data.set_ylim((0, 3840))
    data.set_color(color=spike_colors)
    data.set_clim(clim=amp_range * 1e6)
    data.set_marker_size(marker_size=spike_size)
    data.set_labels(title='Spike times vs Spike depths', xlabel='Time (s)',
                    ylabel='Distance from probe tip (um)', clabel='Spike amplitude (uV)')

    if display:
        fig, ax = plot_scatter(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def image_fr_plot(spike_depths, spike_times, chn_coords, t_bin=0.05, d_bin=5, cmap='binary',
                  display=False):
    """
    Prepare data 2D raster plot of firing rate across recording

    :param spike_depths:
    :param spike_times:
    :param chn_coords:
    :param t_bin: time bin to average across (see also brainbox.processing.bincount2D)
    :param d_bin: depth bin to average across (see also brainbox.processing.bincount2D)
    :param cmap:
    :param display: generate figure
    :return: ImagePlot object, if display=True also returns matplotlib fig and ax objects
    """

    n, x, y = bincount2D(spike_times, spike_depths, t_bin, d_bin,
                         ylim=[0, np.max(chn_coords[:, 1])])
    fr = n.T / t_bin

    data = ImagePlot(fr, x=x, y=y, cmap=cmap)
    data.set_labels(title='Firing Rate', xlabel='Time (s)',
                    ylabel='Distance from probe tip (um)', clabel='Firing Rate (Hz)')
    data.set_clim(clim=(np.min(np.mean(fr, axis=0)), np.max(np.mean(fr, axis=0))))
    if display:
        fig, ax = plot_image(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def image_crosscorr_plot(spike_depths, spike_times, chn_coords, t_bin=0.05, d_bin=40,
                         cmap='viridis', display=False):
    """
    Prepare data for 2D cross correlation plot of data across depth

    :param spike_depths:
    :param spike_times:
    :param chn_coords:
    :param t_bin: t_bin: time bin to average across (see also brainbox.processing.bincount2D)
    :param d_bin: depth bin to average across (see also brainbox.processing.bincount2D)
    :param cmap:
    :param display: generate figure
    :return: ImagePlot object, if display=True also returns matploltlib fig and ax objects
    """

    n, x, y = bincount2D(spike_times, spike_depths, t_bin, d_bin,
                         ylim=[0, np.max(chn_coords[:, 1])])
    corr = np.corrcoef(n)
    corr[np.isnan(corr)] = 0

    data = ImagePlot(corr, x=y, y=y, cmap=cmap)
    data.set_labels(title='Correlation', xlabel='Distance from probe tip (um)',
                    ylabel='Distance from probe tip (um)', clabel='Correlation')

    if display:
        fig, ax = plot_image(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def scatter_amp_depth_fr_plot(spike_amps, spike_clusters, spike_depths, spike_times, cmap='hot',
                              display=False):
    """
    Prepare data for 2D scatter plot of cluster depth vs cluster amp with colour indicating cluster
    firing rate

    :param spike_amps:
    :param spike_clusters:
    :param spike_depths:
    :param spike_times:
    :param cmap:
    :param display: generate figure
    :return: ScatterPlot object, if display=True also returns matplotlib fig and ax objects
    """

    cluster, cluster_depth, n_cluster = compute_cluster_average(spike_clusters, spike_depths)
    _, cluster_amp, _ = compute_cluster_average(spike_clusters, spike_amps)
    cluster_amp = cluster_amp * 1e6
    cluster_fr = n_cluster / np.max(spike_times)

    data = ScatterPlot(x=cluster_amp, y=cluster_depth, c=cluster_fr, cmap=cmap)
    data.set_xlim((0.9 * np.min(cluster_amp), 1.1 * np.max(cluster_amp)))

    if display:
        fig, ax = plot_scatter(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def probe_lfp_spectrum_plot(lfp_power, lfp_freq, chn_coords, chn_inds, freq_range=(0, 4),
                            display=False, pad=True, x_offset=1):
    """
    Prepare data for 2D probe plot of LFP power spectrum along depth of probe

    :param lfp_power:
    :param lfp_freq:
    :param chn_coords:
    :param chn_inds:
    :param freq_range:
    :param display:
    :param pad: whether to add nans around the individual image plots. For matplotlib use pad=True,
    for pyqtgraph use pad=False
    :param x_offset: Distance between the channel banks in x direction
    :return: ProbePlot object, if display=True also returns matplotlib fig and ax objects
    """

    freq_idx = np.where((lfp_freq >= freq_range[0]) & (lfp_freq < freq_range[1]))[0]
    lfp = np.take(lfp_power[freq_idx], chn_inds, axis=1)
    lfp_db = 10 * np.log10(lfp)
    lfp_db[np.isinf(lfp_db)] = np.nan
    lfp_db = np.mean(lfp_db, axis=0)

    data_bank, x_bank, y_bank = arrange_channels2banks(lfp_db, chn_coords, depth=None,
                                                       pad=pad, x_offset=x_offset)
    data = ProbePlot(data_bank, x=x_bank, y=y_bank)
    data.set_labels(ylabel='Distance from probe tip (um)', clabel='PSD 0-4 Hz (dB)')
    clim = np.nanquantile(np.concatenate([np.squeeze(np.ravel(d)) for d in data_bank]).ravel(),
                          [0.1, 0.9])
    data.set_clim(clim)

    if display:
        fig, ax = plot_probe(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def probe_rms_plot(rms_amps, chn_coords, chn_inds, cmap='plasma', band='AP',
                   display=False, pad=True, x_offset=1):
    """
    Prepare data for 2D probe plot of RMS along depth of probe

    :param rms_amps:
    :param chn_coords:
    :param chn_inds:
    :param cmap:
    :param band:
    :param display:
    :param pad: whether to add nans around the individual image plots. For matplotlib use pad=True,
    for pyqtgraph use pad=False
    :param x_offset: Distance between the channel banks in x direction
    :return: ProbePlot object, if display=True also returns matplotlib fig and ax objects
    """

    rms = (np.mean(rms_amps, axis=0)[chn_inds]) * 1e6

    data_bank, x_bank, y_bank = arrange_channels2banks(rms, chn_coords, depth=None,
                                                       pad=pad, x_offset=x_offset)
    data = ProbePlot(data_bank, x=x_bank, y=y_bank, cmap=cmap)
    data.set_labels(ylabel='Distance from probe tip (um)', clabel=f'{band} RMS (uV)')
    clim = np.nanquantile(np.concatenate([np.squeeze(np.ravel(d)) for d in data_bank]).ravel(),
                          [0.1, 0.9])
    data.set_clim(clim)

    if display:
        fig, ax = plot_probe(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def line_fr_plot(spike_depths, spike_times, chn_coords, d_bin=10, display=False):
    """
    Prepare data for 1D line plot of average firing rate across depth

    :param spike_depths:
    :param spike_times:
    :param chn_coords:
    :param d_bin: depth bin to average across (see also brainbox.processing.bincount2D)
    :param display:
    :return:
    """
    t_bin = np.max(spike_times)
    n, x, y = bincount2D(spike_times, spike_depths, t_bin, d_bin,
                         ylim=[0, np.max(chn_coords[:, 1])])
    mean_fr = n[:, 0] / t_bin

    data = LinePlot(x=mean_fr, y=y)
    data.set_xlim((0, np.max(mean_fr)))
    data.set_labels(title='Avg Firing Rate', xlabel='Firing Rate (Hz)',
                    ylabel='Distance from probe tip (um)')

    if display:
        fig, ax = plot_line(data.convert2dict())
        return data.convert2dict(), fig, ax

    return data


def line_amp_plot(spike_amps, spike_depths, spike_times, chn_coords, d_bin=10, display=False):
    """
    Prepare data for 1D line plot of average firing rate across depth
    :param spike_amps:
    :param spike_depths:
    :param spike_times:
    :param chn_coords:
    :param d_bin: depth bin to average across (see also brainbox.processing.bincount2D)
    :param display:
    :return:
    """
    t_bin = np.max(spike_times)
    n, _, _ = bincount2D(spike_times, spike_depths, t_bin, d_bin,
                         ylim=[0, np.max(chn_coords[:, 1])])
    amp, x, y = bincount2D(spike_times, spike_depths, t_bin, d_bin,
                           ylim=[0, np.max(chn_coords[:, 1])], weights=spike_amps)

    mean_amp = np.divide(amp[:, 0], n[:, 0]) * 1e6
    mean_amp[np.isnan(mean_amp)] = 0
    remove_bins = np.where(n[:, 0] < 50)[0]
    mean_amp[remove_bins] = 0

    data = LinePlot(x=mean_amp, y=y)
    data.set_xlim((0, np.max(mean_amp)))
    data.set_labels(title='Avg Amplitude', xlabel='Amplitude (uV)',
                    ylabel='Distance from probe tip (um)')
    if display:
        fig, ax = plot_line(data.convert2dict())
        return data.convert2dict(), fig, ax
    return data

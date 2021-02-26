import numpy as np
from matplotlib import cm
import matplotlib
import scipy
from brainbox.plot.plot_base import ImagePlot, ScatterPlot, LinePlot
from brainbox.processing import bincount2D


def image_lfp_spectrum_plot(lfp_power, lfp_freq, chn_coords, chn_inds, freq_range=(0, 300),
                            avg_across_depth=False):
    """

    :param lfp_power:
    :param lfp_freq:
    :param chn_coords:
    :param chn_inds:
    :param freq_range:
    :param avg_across_depth:
    :return:
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
        def avg_chn_depth(a):
            return np.mean([a[chn_idx], a[chn_idx_eq]], axis=0)

        lfp_db = np.apply_along_axis(avg_chn_depth, 1, lfp_db)

        lfp_db = np.apply_along_axis(lambda a: np.mean([a[chn_idx], a[chn_idx_eq]], axis=0), 1,
                                     lfp_db)

        x = freqs
        y = chn_depth


    data = ImagePlot(lfp_db, x=x, y=y, cmap='viridis')
    data.set_labels(title='LFP Power Spectrum', xlabel='Frequency (Hz)',
                    ylabel='Distance from probe tip (um)', clabel='LFP Power (dB)')
    data.set_clim(clim=np.quantile(lfp_db, [0.1, 0.9]))

    return data



def image_rms_plot(rms_amps, rms_times, chn_coords, chn_inds, avg_across_depth=False,
                   median_subtract=True, **kwargs):

    rms = rms_amps[:, chn_inds]
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

    data = ImagePlot(rms, x=x, y=y, **kwargs)
    data.set_labels(xlabel='Time (s)', ylabel='Distance from probe tip (um)', **kwargs)
    data.set_clim(clim=np.quantile(rms, [0.1, 0.9]))

    return data


def scatter_raster_plot(spike_amps, spike_depths, spike_times, n_amp_bins=10, amp_range=None,
                    cmap='BuPu', saturated_color=None, subsample_factor=100):

    amp_range = np.quantile(spike_amps, [0, 0.9])
    amp_bins = np.linspace(amp_range[0], amp_range[1], n_amp_bins)
    color_bin = np.linspace(0.0, 1.0, n_amp_bins + 1)
    colors = (cm.get_cmap(cmap)(color_bin)[np.newaxis, :, :3][0]) * 255

    spike_amps = spike_amps[0:-1:subsample_factor]
    spike_colors = np.empty((spike_amps.size, 3))
    spike_size = np.empty(spike_amps.size)
    for iA in range(amp_bins.size):
        if iA == (amp_bins.size - 1):
            idx = np.where(spike_amps > amp_bins[iA])[0]
            # Make saturated spikes a very dark purple
            spike_colors[idx] = colors[-1]
        else:
            idx = np.where((spike_amps > amp_bins[iA]) & (spike_amps <= amp_bins[iA + 1]))[0]
            spike_colors[idx] = [*colors[iA]]

        spike_size[idx] = iA / (n_amp_bins / 6)

    data = ScatterPlot(x=spike_times[0:-1:subsample_factor], y=spike_depths[0:-1:subsample_factor],
                       c=spike_colors, cmap='BuPu')
    data.set_color(color=spike_colors)
    data.set_clim(clim=amp_range * 1e6)
    data.set_marker_size(marker_size=spike_size)
    data.set_labels(title='Spike times vs Spike depths', xlabel='Time (s)',
                    ylabel='Distance from probe tip (um)', clabel='Spike amplitude (uV)')

    return data


def image_fr(spike_depths, spike_times, chn_coords, t_bin=0.05, d_bin=5, cmap='binary'):

    n, x, y = bincount2D(spike_times, spike_depths, t_bin, d_bin,
                                  ylim=[0, np.max(chn_coords[:, 1])])
    fr = n.T / t_bin
    data = ImagePlot(fr, x=x, y=y, cmap=cmap)
    data.set_labels(title='Firing Rate', xlabel='Time (s)',
                    ylabel='Distance from probe tip (um)', clabel='Firing Rate (Hz)')
    data.set_clim(clim=(np.min(np.mean(fr, axis=0)), np.max(np.mean(fr,axis=0))))

    return data


def image_crosscorr(spike_depths, spike_times, chn_coords, t_bin=0.05, d_bin=40):
    n, x, y = bincount2D(spike_times, spike_depths, t_bin, d_bin,
                                  ylim=[0, np.max(chn_coords[:, 1])])
    corr = np.corrcoef(n)
    corr[np.isnan(corr)] = 0

    data = ImagePlot(corr, x=y, y=y)
    data.set_labels(title='Correlation', xlabel='Distance from probe tip (um)',
                    ylabel='Distance from probe tip (um)', clabel='Correlation')


def scatter_amp_depth_fr_plot(spike_clusters, spike_amps, spike_depths, spike_times, cmap='hot'):
    cluster, n_cluster = np.unique(spike_clusters, return_counts=True)
    _, cluster_depth, _ = compute_cluster_average(spike_clusters, spike_depths)
    _, cluster_amp, _ = compute_cluster_average(spike_clusters, spike_amps)

    cluster_fr = n_cluster / np.max(spike_times)

    data = scatter_x_y_c_plot(cluster_amp, cluster_depth, cluster_fr, cmap=cmap)

    return data


def scatter_x_y_c_plot(x, y, c, cmap=None, clim=None):
    # convenience function for pyqtgraph implementation

    data = ScatterPlot(x=x, y=y, c=c, cmap=cmap)
    data.set_clim(clim=clim)

    norm = matplotlib.colors.Normalize(vmin=data.clim[0], vmax=data.clim[1], clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))
    cluster_color = np.array([mapper.to_rgba(col) for col in c])
    data.set_color(color=cluster_color)

    return data


def compute_cluster_average(spike_clusters, spike_var):
    clust, inverse, counts = np.unique(spike_clusters, return_inverse=True, return_counts=True)
    _spike_var = scipy.sparse.csr_matrix((spike_var, (inverse, np.zeros(inverse.size, dtype=int))))
    spike_var_avg = np.ravel(_spike_var.toarray()) / counts

    return clust, spike_var_avg, counts


def probe_lfp_spectrum_plot(lfp_power, lfp_freq, chn_coords, chn_inds, freq_range=(0, 4)):

    freq_idx = np.where((lfp_freq >= freq_range[0]) & (lfp_freq < freq_range[1]))[0]
    lfp = np.take(lfp_power[freq_idx], chn_inds, axis=1)
    lfp_db = 10 * np.log10(lfp)
    lfp_db[np.isinf(lfp_db)] = np.nan
    lfp_db = np.mean(lfp_db, axis=0)

    data_bank, x_bank, y_bank = arrange_channels2banks(lfp_db, chn_coords, y_coords=None,
                                                       pad=True, x_offset=1)
    data = ProbePlot(data_bank, x=x_bank, y=y_bank)
    data.set_labels(ylabel='Distance from probe tip (um)', clabel='PSD 0-4 Hz (dB)')

    return data


def arrange_channels2banks(data, chn_coords, y_coords=None, pad=True, x_offset=1):
    data_bank = []
    x_bank = []
    y_bank = []

    if y_coords is None:
        y_coords = chn_coords[:, 1]

    for iX, x in enumerate(np.unique(chn_coords[:, 0])):
        bnk_idx = np.where(chn_coords[:, 0] == x)[0]
        bnk_data = data[bnk_idx, np.newaxis].T
        # This is a hack! Although data is 1D we give it two x coords so we can correctly set
        # scale and extent (compatible with pyqtgraph and matplotlib.imshow)
        # For matplotlib.image.Nonuniformimage must use pad=True option
        bnk_x = np.array((iX * x_offset, (iX + 1)* x_offset))
        bnk_y = y_coords[bnk_idx]
        if pad:
            # pad data in y direction
            bnk_data = np.insert(bnk_data, 0, np.nan)
            bnk_data = np.append(bnk_data, np.nan)
            # pad data in x direction
            bnk_data = bnk_data.T
            bnk_data = np.insert(bnk_data, 0, np.full(bnk_data.shape[1], np.nan), axis=0)
            bnk_data = np.append(bnk_data, np.full((1, bnk_data.shape[1]), np.nan), axis=0)

            # pad the x values
            bnk_x = np.arange(iX * x_offset, (iX + 3) * x_offset, x_offset)

            # pad the y values
            bnk_y = np.insert(bnk_y, 0, bnk_y[0] - np.abs(bnk_y[2] - bnk_y[0]))
            bnk_y = np.append(bnk_y, bnk_y[-1] + np.abs(bnk_y[-3] - bnk_y[-1]))


        data_bank.append(bnk_data)
        x_bank.append(bnk_x)
        y_bank.append(bnk_y)

    return data_bank, x_bank, y_bank

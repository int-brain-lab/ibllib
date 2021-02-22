import numpy as np

# This is base dict to store plot data
data_dict = {'x': None,
             'y': None,
             'xrange': None,
             'yrange': None,
             'img': None,
             'xscale': None,
             'yscale': None,
             'xoffset': None,
             'yoffset': None,
             'cmap': None,
             'cmap_levels': None,
             'xlabel': None,
             'ylabel': None,
             'title': None,
             'size': None,
             'symbol': None,
             'color': None,
             'pen': None}



def prepare_lfp_spectrum_plot(lfp_power, lfp_freq, chn_coords, chn_inds=None,
                              freq_range = (0, 300), avg_across_depth=False,
                              avg_across_freq=False, probe_layout=False):

    if not chn_inds:
        chn_inds = np.arange(len(chn_coords))

    freq_idx = np.where((lfp_freq >= freq_range[0]) & (lfp_freq < freq_range[1]))[0]
    lfp = lfp_power[freq_idx, chn_inds]
    lfp_db = 10 * np.log10(lfp)
    lfp_db[np.isinf(lfp_db)] = np.nan

    # Average across channels that are at the same depth
    if avg_across_depth:
        _, chn_depth, chn_count = np.unique(chn_coords[:, 1], return_index=True,
                                            return_counts=True)
        chn_depth_eq = np.copy(chn_depth)
        chn_depth_eq[np.where(chn_count == 2)] += 1
        def avg_chn_depth(a):
            return np.mean([a[chn_depth], a[chn_depth_eq]], axis=0)

        lfp_dB = np.apply_along_axis(avg_chn_depth, 1, lfp_db)

    # Average across frequency
    if avg_across_freq:
        lfp_dB = np.mean(lfp_db)

    # Arrange into probe layout
    if probe_layout:
        # Get to this later
        #lfp_dB, xscale, yscale = arrange2probe(lfp_db)
    else:
        xscale = (freq_range[1] - freq_range[0]) / lfp_db.shape[0]
        yscale = (np.max(chn_coords[:, 1]) - np.min(chn_coords[:, 1])) / lfp_db.shape[1]

    data = data_dict
    data['img'] = lfp_dB
    data[]

    data_img = {
        'data': lfp_dB,
        'scale': np.array([xscale, yscale]),
        'levels': levels,
        'offset': np.array([0, 0]),
        'cmap': 'viridis',
        'xrange': np.array([freq_range[0], freq_range[-1]]),
        'xaxis': 'Frequency (Hz)',
        'title': 'PSD (dB)'
    }


def get_lfp_spectrum_data(self):
    freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
    data_probe = {}
    if not self.lfp_data_status:
        data_img = None
        for freq in freq_bands:
            lfp_band_data = {f"{freq[0]} - {freq[1]} Hz": None}
            data_probe.update(lfp_band_data)

        return data_img, data_probe
    else:
        # Power spectrum image
        freq_range = [0, 300]
        freq_idx = np.where((self.lfp_freq >= freq_range[0]) &
                            (self.lfp_freq < freq_range[1]))[0]
        _lfp = np.take(self.lfp_power[freq_idx], self.chn_ind, axis=1)
        _lfp_dB = 10 * np.log10(_lfp)
        _, self.chn_depth, chn_count = np.unique(self.chn_coords[:, 1], return_index=True,
                                                 return_counts=True)
        self.chn_depth_eq = np.copy(self.chn_depth)
        self.chn_depth_eq[np.where(chn_count == 2)] += 1

        def avg_chn_depth(a):
            return (np.mean([a[self.chn_depth], a[self.chn_depth_eq]], axis=0))

        img = np.apply_along_axis(avg_chn_depth, 1, _lfp_dB)
        levels = np.quantile(img, [0.1, 0.9])
        xscale = (freq_range[-1] - freq_range[0]) / img.shape[0]
        yscale = (np.max(self.chn_coords[:, 1]) - np.min(self.chn_coords[:, 1])) / img.shape[1]

        data_img = {
            'img': img,
            'scale': np.array([xscale, yscale]),
            'levels': levels,
            'offset': np.array([0, 0]),
            'cmap': 'viridis',
            'xrange': np.array([freq_range[0], freq_range[-1]]),
            'xaxis': 'Frequency (Hz)',
            'title': 'PSD (dB)'
        }

        # Power spectrum in bands on probe
        for freq in freq_bands:
            freq_idx = np.where((self.lfp_freq >= freq[0]) & (self.lfp_freq < freq[1]))[0]
            lfp_avg = np.mean(self.lfp_power[freq_idx], axis=0)[self.chn_ind]
            lfp_avg_dB = 10 * np.log10(lfp_avg)
            probe_img, probe_scale, probe_offset = self.arrange_channels2banks(lfp_avg_dB)
            probe_levels = np.quantile(lfp_avg_dB, [0.1, 0.9])

            lfp_band_data = {f"{freq[0]} - {freq[1]} Hz": {
                'img': probe_img,
                'scale': probe_scale,
                'offset': probe_offset,
                'levels': probe_levels,
                'cmap': 'viridis',
                'xaxis': 'Time (s)',
                'xrange': np.array([0 * BNK_SIZE, (N_BNK) * BNK_SIZE]),
                'title': f"{freq[0]} - {freq[1]} Hz (dB)"}
            }
            data_probe.update(lfp_band_data)

        return data_img, data_probe


def arrange_channels2banks_warped(data, y):
    bnk_data = []
    bnk_y = []
    bnk_x = []
    for iX, x in enumerate(np.unique(SITES_COORDINATES[:, 0])):
        bnk_idx = np.where(SITES_COORDINATES[:, 0] == x)[0]
        bnk_vals = data[bnk_idx]
        bnk_vals = np.insert(bnk_vals, 0, np.nan)
        bnk_vals = np.append(bnk_vals, np.nan)
        bnk_vals = bnk_vals[:, np.newaxis].T
        bnk_vals = np.insert(bnk_vals, 0, np.full(bnk_vals.shape[1], np.nan), axis=0)
        bnk_vals = np.append(bnk_vals, np.full((1, bnk_vals.shape[1]), np.nan), axis=0)
        bnk_data.append(bnk_vals)

        y_pos = y[bnk_idx]
        y_pos = np.insert(y_pos, 0, y_pos[0] - np.abs(y_pos[2] - y_pos[0]))
        y_pos = np.append(y_pos, y_pos[-1] + np.abs(y_pos[-3] - y_pos[-1]))
        bnk_y.append(y_pos)

        x = np.arange(iX, iX + 3)
        bnk_x.append(x)

    return bnk_x, bnk_y, bnk_data


def arrange_channels2banks2(self, data):
    Y_OFFSET = 20
    bnk_data = []
    bnk_scale = np.empty((N_BNK, 2))
    bnk_offset = np.empty((N_BNK, 2))
    for iX, x in enumerate(np.unique(self.chn_coords[:, 0])):
        bnk_idx = np.where(self.chn_coords[:, 0] == x)[0]
        bnk_vals = data[bnk_idx]
        _bnk_data = np.reshape(bnk_vals, (bnk_vals.size, 1)).T
        _bnk_yscale = ((np.max(self.chn_coords[bnk_idx, 1]) -
                        np.min(self.chn_coords[bnk_idx, 1])) / _bnk_data.shape[1])
        _bnk_xscale = BNK_SIZE / _bnk_data.shape[0]
        _bnk_yoffset = np.min(self.chn_coords[bnk_idx, 1]) - Y_OFFSET
        _bnk_xoffset = BNK_SIZE * iX
        bnk_data.append(_bnk_data)
        bnk_scale[iX, :] = np.array([_bnk_xscale, _bnk_yscale])
        bnk_offset[iX, :] = np.array([_bnk_xoffset, _bnk_yoffset])
    return bnk_data, bnk_scale, bnk_offset
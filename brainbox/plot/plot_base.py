import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import matplotlib
from matplotlib import cm


axis_dict = {'x': 0, 'y': 1, 'z': 2}


class DefaultPlot(object):
    """
    Base class for organising data into a structure that can be easily used to create plots.
    The idea is that the dictionary is independent of plotting method and so can be fed into
    matplotlib, pyqtgraph, datoviz (or any plotting method of choice).
    """
    def __init__(self, plot_type, data):
        """

        :param plot_type: type of plot (just for reference)
        :type plot_type: string
        :param data: dict of data containing at least 'x', 'y', and may additionally contain 'z'
        for 3D plots and 'c' 2D (image or scatter plots) with third variable represented by colour
        :type data: dict
        """
        self.plot_type = plot_type
        self.data = data
        self.hlines = []
        self.vlines = []
        self.set_labels()

    def add_lines(self, pos, orientation, lim=None, style='--', width=3, color='k'):
        """
        Method to specify position and style of horizontal or vertical reference lines
        :param pos: position of line
        :param orientation: either 'v' for vertical line or 'h' for horizontal line
        :param lim:
        :param style: line style
        :param width: line width
        :param color: line colour
        :return:
        """
        if orientation == 'v':
            lim = self._set_default(lim, self.ylim)
            self.vlines.append({'pos': pos, 'lim': lim, 'style': style, 'width': width,
                                'color': color})
        if orientation == 'h':
            lim = self._set_default(lim, self.xlim)
            self.hlines.append({'pos': pos, 'lim': lim, 'style': style, 'width': width,
                                'color': color})

    def set_labels(self, title=None, xlabel=None, ylabel=None, zlabel=None, clabel=None):
        self.labels = {'title': title, 'xlabel': xlabel, 'ylabel': ylabel, 'zlabel': zlabel,
                       'clabel': clabel}

    def set_xlim(self, xlim=None):
        self.xlim = self._set_lim('x', lim=xlim)

    def set_ylim(self, ylim=None):
        self.ylim = self._set_lim('y', lim=ylim)

    def set_zlim(self, zlim=None):
        self.zlim = self._set_lim('z', lim=zlim)

    def set_clim(self, clim=None):
        self.clim = self._set_lim('c', lim=clim)

    def _set_lim(self, axis, lim=None):
        if lim is not None:
            assert(len(lim) == 2)
        else:
            lim = (np.nanmin(self.data[axis]), np.nanmax(self.data[axis]))
        return lim

    def _set_default(self, val, default):
        if val is None:
            return default
        else:
            return val

    def convert2dict(self):
        return vars(self)


class ImagePlot(DefaultPlot):
    def __init__(self, img, x=None, y=None, cmap=None):

        data = {'x': self._set_default(x, np.arange(img.shape[0])),
                'y': self._set_default(y, np.arange(img.shape[1])), 'c': img}
        # Initialise default plot class with data
        super().__init__('image', data)
        self.scale = None
        self.offset = None
        self.cmap = self._set_default(cmap, 'viridis')

        self.set_xlim()
        self.set_ylim()
        self.set_clim()

    def set_scale(self, scale=None):
        # For pyqtgraph implementation
        if scale is not None:
            assert(len(scale) == 2)
        self.scale = self._set_default(scale, (self._get_scale('x'), self._get_scale('y')))

    def _get_scale(self, axis):
        lim = self._set_lim(axis)
        scale = (lim[1] - lim[0]) / self.data['c'].shape[axis_dict[axis]]
        return scale

    def set_offset(self, offset=None):
        # For pyqtgraph implementation
        if offset is not None:
            assert(len(offset) == 2)
        self.offset = self._set_default(offset, (0, 0))


class ProbePlot(DefaultPlot):
    def __init__(self, img, x, y, cmap=None):
        data = {'x': x, 'y': y, 'c': img}
        super().__init__('probe', data)

        self.cmap = self._set_default(cmap, 'viridis')

        self.set_xlim()
        self.set_ylim()
        self.set_clim()
        self.set_scale()
        self.set_offset()

    def set_scale(self, idx=None, scale=None):
        if scale is not None:
            self.scale[idx] = scale
        else:
            self.scale = [(self._get_scale(i, 'x'), self._get_scale(i, 'y'))
                          for i in range(len(self.data['x']))]

    def _get_scale(self, idx, axis):
        lim = self._set_lim_list(axis, idx)
        scale = (lim[1] - lim[0]) / self.data['c'][idx].shape[axis_dict[axis]]
        return scale

    def _set_lim_list(self, axis, idx, lim=None):
        if lim is not None:
            assert(len(lim) == 2)
        else:
            lim = (np.nanmin(self.data[axis][idx]), np.nanmax(self.data[axis][idx]))
        return lim

    def set_offset(self, idx=None, offset=None):
        if offset is not None:
            self.offset[idx] = offset
        else:
            self.offset = [(np.min(self.data['x'][i]), np.min(self.data['y'][i]))
                           for i in range(len(self.data['x']))]

    def _set_lim(self, axis, lim=None):
        if lim is not None:
            assert (len(lim) == 2)
        else:
            data = np.concatenate([np.squeeze(np.ravel(d)) for d in self.data[axis]]).ravel()
            lim = (np.nanmin(data), np.nanmax(data))
        return lim


class ScatterPlot(DefaultPlot):
    # z for 3D scatter or for using colorbar with scatterplot
    # currently not supported
    def __init__(self, x, y, z=None, c=None, cmap=None, plot_type='scatter'):
        data = {'x': x, 'y': y, 'z': z, 'c': c}
        super().__init__(plot_type, data)

        self._set_init_style()
        self.set_xlim()
        self.set_ylim()
        # If we have 3D data
        if data['z'] is not None:
            self.set_zlim()
        # If we want colorbar associated with scatter plot
        self.set_clim()
        self.cmap = self._set_default(cmap, 'viridis')

    def _set_init_style(self):
        self.set_color()
        self.set_marker_size()
        self.set_marker_type('o')
        self.set_opacity()
        self.set_line_color()
        self.set_line_width()

    def set_color(self, color=None):
        self.color = self._set_default(color, 'b')

    def set_marker_size(self, marker_size=None):
        self.marker_size = self._set_default(marker_size, None)

    def set_marker_type(self, marker_type=None):
        self.marker_type = self._set_default(marker_type, None)

    def set_opacity(self, opacity=None):
        self.opacity  = self._set_default(opacity, 1)

    def set_line_color(self, line_color=None):
        self.line_color = self._set_default(line_color, None)

    def set_line_width(self, line_width=None):
        self.line_width = self._set_default(line_width, None)

    def set_line_style(self, line_style=None):
        self.line_style = self._set_default(line_style, '-')


class LinePlot(ScatterPlot):
    def __init__(self, x, y):
        super().__init__(x, y, plot_type='line')

        self._set_init_style()
        self.set_xlim()
        self.set_ylim()

    def _set_init_style(self):
        self.set_line_color('k')
        self.set_line_width(2)
        self.set_line_style()
        self.set_marker_size()
        self.set_marker_type()


def add_lines(ax, data, **kwargs):

    for vline in data['vlines']:
        ax.vlines(vline['pos'], ymin=vline['lim'][0], ymax=vline['lim'][1],
                  linestyles=vline['style'], linewidth=vline['width'], colors=vline['color'],
                  **kwargs)

    for hline in data['hlines']:
        ax.hlines(hline['pos'], xmin=hline['lim'][0], xmax=hline['lim'][1],
                  linestyles=hline['style'], linewidth=hline['width'], colors=hline['color'],
                  **kwargs)

    return ax


def plot_image(data, ax=None, show_cbar=True, **kwargs):
    if not isinstance(data, dict):
        data = data.convert2dict()

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    img = ax.imshow(data['data']['c'].T, extent=np.r_[data['xlim'], data['ylim']], cmap=data['cmap'],
                    vmin=data['clim'][0], vmax=data['clim'][1], origin='lower', aspect='auto',
                    **kwargs)

    ax.set_xlim(data['xlim'][0], data['xlim'][1])
    ax.set_ylim(data['ylim'][0], data['ylim'][1])
    ax.set_xlabel(data['labels']['xlabel'])
    ax.set_ylabel(data['labels']['ylabel'])
    ax.set_title(data['labels']['title'])

    if show_cbar:
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(data['labels']['clabel'])

    ax = add_lines(ax, data, **kwargs)

    plt.show()

    return ax, fig


def plot_scatter(data, ax=None, show_cbar=True, **kwargs):
    if not isinstance(data, dict):
        data = data.convert2dict()

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    # Single color for all points
    if data['data']['c'] is None:
        scat = ax.scatter(x=data['data']['x'], y=data['data']['y'], c=data['color'],
                          s=data['marker_size'], marker=data['marker_type'],
                          edgecolors=data['line_color'], linewidths=data['line_width'])
    else:
        # Colour for each point specified
        if len(data['color']) == len(data['data']['x']):
            if np.max(data['color']) > 1:
                data['color'] = data['color'] / 255

            scat = ax.scatter(x=data['data']['x'], y=data['data']['y'], c=data['color'],
                              s=data['marker_size'], marker=data['marker_type'],
                              edgecolors=data['line_color'], linewidths=data['line_width'])
            if show_cbar:
                norm = matplotlib.colors.Normalize(vmin=data['clim'][0], vmax=data['clim'][1],
                                                   clip=True)
                cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=data['cmap']), ax=ax)
                cbar.set_label(data['labels']['clabel'])
        # Automatically generate from c data
        else:
            scat = ax.scatter(x=data['data']['x'], y=data['data']['y'], c=data['data']['c'],
                              s=data['marker_size'], marker=data['marker_type'], cmap=data['cmap'],
                              vmin=data['clim'][0], vmax=data['clim'][1],
                              edgecolors=data['line_color'], linewidths=data['line_width'])
            if show_cbar:
                cbar = fig.colorbar(scat, ax=ax)
                cbar.set_label(data['labels']['clabel'])

    ax = add_lines(ax, data, **kwargs)

    ax.set_xlim(data['xlim'][0], data['xlim'][1])
    ax.set_ylim(data['ylim'][0], data['ylim'][1])
    ax.set_xlabel(data['labels']['xlabel'])
    ax.set_ylabel(data['labels']['ylabel'])
    ax.set_title(data['labels']['title'])

    plt.show()

    return ax, fig


def plot_probe(data, ax=None, show_cbar=True, make_pretty=True, **kwargs):

    if not isinstance(data, dict):
        data = data.convert2dict()

    if not ax:
        fig, ax = plt.subplots(figsize=(2,8), **kwargs)
    else:
        fig = plt.gcf()

    for (x, y, dat) in zip(data['data']['x'], data['data']['y'], data['data']['c']):
        im = NonUniformImage(ax, interpolation='nearest', cmap=data['cmap'])
        im.set_clim(data['clim'][0], data['clim'][1])
        im.set_data(x, y, dat.T)
        ax.images.append(im)

    ax.set_xlim(data['xlim'][0], data['xlim'][1])
    ax.set_ylim(data['ylim'][0], data['ylim'][1])
    ax.set_xlabel(data['labels']['xlabel'])
    ax.set_ylabel(data['labels']['ylabel'])
    ax.set_title(data['labels']['title'])

    if make_pretty:
        ax.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    if show_cbar:
        cbar = fig.colorbar(im, orientation="horizontal", pad=0.02, ax=ax)
        cbar.set_label(data['labels']['clabel'])

    ax = add_lines(ax, data, **kwargs)

    plt.show()

    return ax, fig


def plot_line(data, ax=None, **kwargs):
    if not isinstance(data, dict):
        data = data.convert2dict()

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    line = ax.plot(data['data']['x'], data['data']['y'], color=data['line_color'],
                   linestyle=data['line_style'], linewidth=data['line_width'],
                   marker=data['marker_type'], markersize=data['marker_size'])
    ax = add_lines(ax, data, **kwargs)

    ax.set_xlim(data['xlim'][0], data['xlim'][1])
    ax.set_ylim(data['ylim'][0], data['ylim'][1])
    ax.set_xlabel(data['labels']['xlabel'])
    ax.set_ylabel(data['labels']['ylabel'])
    ax.set_title(data['labels']['title'])

    plt.show()

    return ax, fig


def scatter_xyc_plot(x, y, c, cmap=None, clim=None, rgb=False):
    """
    General function for preparing x y scatter plot with third variable encoded by colour of points
    :param x:
    :param y:
    :param c:
    :param cmap:
    :param clim:
    :param rgb: Whether to compute rgb (set True when preparing pyqtgraph data)
    :return:
    """

    data = ScatterPlot(x=x, y=y, c=c, cmap=cmap)
    data.set_clim(clim=clim)
    if rgb:
        norm = matplotlib.colors.Normalize(vmin=data.clim[0], vmax=data.clim[1], clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))
        cluster_color = np.array([mapper.to_rgba(col) for col in c])
        data.set_color(color=cluster_color)

    return data


def arrange_channels2banks(data, chn_coords, depth=None, pad=True, x_offset=1):
    """
    Rearranges data on channels so it matches geometry of probe. e.g For Neuropixel 2.0 rearranges
    channels into 4 banks with checkerboard pattern

    :param data: data on channels
    :param chn_coords: local coordinates of channels on probe
    :param depth: depth location of electrode (for example could be relative to bregma). If none
    given will stay in probe local coordinates
    :param pad: for matplotlib implementation with NonUniformImage we need to surround our data
    with nans so that it shows as finite display
    :param x_offset: spacing between banks in x direction
    :return: list, data, x position and y position for each bank
    """
    data_bank = []
    x_bank = []
    y_bank = []

    if depth is None:
        depth = chn_coords[:, 1]

    for iX, x in enumerate(np.unique(chn_coords[:, 0])):
        bnk_idx = np.where(chn_coords[:, 0] == x)[0]
        bnk_data = data[bnk_idx, np.newaxis].T
        # This is a hack! Although data is 1D we give it two x coords so we can correctly set
        # scale and extent (compatible with pyqtgraph and matplotlib.imshow)
        # For matplotlib.image.Nonuniformimage must use pad=True option
        bnk_x = np.array((iX * x_offset, (iX + 1)* x_offset))
        bnk_y = depth[bnk_idx]
        if pad:
            # pad data in y direction
            bnk_data = np.insert(bnk_data, 0, np.nan)
            bnk_data = np.append(bnk_data, np.nan)
            # pad data in x direction
            bnk_data = bnk_data[:, np.newaxis].T
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

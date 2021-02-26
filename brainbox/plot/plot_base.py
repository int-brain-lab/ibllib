import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage

axis_dict = {'x': 0, 'y': 1, 'z': 2}


class DefaultPlot(object):
    def __init__(self, plot_type, data):
        self.plot_type = plot_type
        self.data = data
        self.hlines = []
        self.vlines = []
        self.set_labels()

    def add_lines(self, pos, orientation, lim=None, style='--', width=6, color='k'):
        if orientation == 'v':
            lim = self._set_default(lim, self.set_ylim())
            print(lim)
            self.vlines.append({'pos': pos, 'lim': lim, 'style': style, 'width': int(width),
                                'color': color})
        if orientation == 'h':
            lim = self._set_default(lim, self.set_xlim())
            self.hlines.append({'pos': pos, 'lim': lim, 'style': style, 'width': int(width),
                                'color': color})

    def set_labels(self, title=None, xlabel=None, ylabel=None, zlabel=None, clabel=None):
        self.labels = {'title': title, 'xlabel': xlabel, 'ylabel': ylabel, 'zlabel': zlabel,
                       'clabel': clabel}

    def set_xlim(self, xlim=None):
        self.xlim = self._set_lim('x', lim=xlim)
        return self.xlim

    def set_ylim(self, ylim=None):
        self.ylim = self._set_lim('y', lim=ylim)
        return self.ylim

    def set_zlim(self, zlim=None):
        self.zlim = self._set_lim('z', lim=zlim)
        return self.zlim

    def set_clim(self, clim=None):
        self.clim = self._set_lim('c', lim=clim)
        return self.clim

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
            concat_data = np.concatenate([np.squeeze(d) for d in self.data[axis]]).ravel()
            lim = (np.nanmin(concat_data), np.nanmax(concat_data))
        return lim


class ScatterPlot(DefaultPlot):
    # z for 3D scatter or for using colorbar with scatterplot
    # currently not supported
    def __init__(self, x, y, z=None, c=None, cmap=None):
        data = {'x': x, 'y': y, 'z': z, 'c': c}
        super().__init__('scatter', data)

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
        self.set_marker_type()
        self.set_opacity()
        self.set_edge_color()
        self.set_edge_width()

    def set_color(self, color=None):
        # Only use for single colour for all point. If want to specifiy colour for each point use
        # pass in this data as data['c']
        self.color = self._set_default(color, 'b')

    def set_marker_size(self, marker_size=None):
        self.marker_size = self._set_default(marker_size, None)

    def set_marker_type(self, marker_type=None):
        self.marker_type = self._set_default(marker_type, 'o')

    def set_opacity(self, opacity=None):
        self.opacity  = self._set_default(opacity, 1)

    def set_edge_color(self, edge_color=None):
        self.edge_color = self._set_default(edge_color, None)

    def set_edge_width(self, line_width=None):
        self.edge_width = self._set_default(line_width, None)


class LinePlot(DefaultPlot):
    def __init__(self, x, y):
        data = {'x': x, 'y': y}
        super().__init__('line', data)

        self._set_init_style()
        self.set_xlim()
        self.set_ylim()

    def _set_init_style(self):
        self.set_line_color()
        self.set_line_width()
        self.set_line_style()

    def set_line_color(self, edge_color=None):
        self.line_color = self._set_default(edge_color, 'k')

    def set_line_width(self, line_width=None):
        self.line_width = self._set_default(line_width, 8)

    def set_line_style(self, line_style=None):
        self.line_style = self._set_default(line_style, 'solid')


def add_lines(ax, data, **kwargs):

    for vline in data['vlines']:
        ax.vlines(x=vline['pos'], ymin=vline['lim'][0], ymax=vline['lim'][0],
                 linestyles=vline['style'], linewidth=vline['width'], colors=vline['color'],
                 **kwargs)

    for hline in data['hlines']:
        ax.hlines(y=hline['pos'], xmin=hline['lim'][0], xmax=hline['lim'][0],
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

    img = ax.imshow(data['data']['c'], extent=np.r_[data['xlim'], data['ylim']], cmap=data['cmap'],
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
                          edgecolors=data['edge_color'], linewidths=data['edge_width'])
        show_cbar = False
    else:
        # Colour for each point specified
        # need to check if we want to divide by 255
        if len(data['color']) == len(data['data']['x']):
            scat = ax.scatter(x=data['data']['x'], y=data['data']['y'], c=data['color']/255,
                              s=data['marker_size'], marker=data['marker_type'], cmap=data['cmap'],
                              vmin=data['clim'][0], vmax=data['clim'][1],
                              edgecolors=data['edge_color'], linewidths=data['edge_width'])
        # Automatically generate from c data
        else:
            scat = ax.scatter(x=data['data']['x'], y=data['data']['y'], c=data['data']['c'],
                              s=data['marker_size'], marker=data['marker_type'], cmap=data['cmap'],
                              vmin=data['clim'][0], vmax=data['clim'][1],
                              edgecolors=data['edge_color'], linewidths=data['edge_width'])

    ax.set_xlim(data['xlim'][0], data['xlim'][1])
    ax.set_ylim(data['ylim'][0], data['ylim'][1])
    ax.set_xlabel(data['labels']['xlabel'])
    ax.set_ylabel(data['labels']['ylabel'])
    ax.set_title(data['labels']['title'])

    if show_cbar:
        cbar = fig.colorbar(scat, ax=ax)
        cbar.set_label(data['labels']['clabel'])

    ax = add_lines(ax, data, **kwargs)

    plt.show()

    return ax, fig


def plot_probe(data, ax=None, show_cbar=True, **kwargs):
    # This still isn't working
    if not isinstance(data, dict):
        data = data.convert2dict()

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    for (x, y, dat) in zip(data['data']['x'], data['data']['y'], data['data']['c']):
        im = NonUniformImage(ax, interpolation='nearest', cmap=data['cmap'])
        im.set_clim(data['clim'][0], data['clim'][1])
        im.set_data(np.array((x[0])), y, dat.T)
        ax.images.append(im)


    ax.set_xlim(data['xlim'][0], data['xlim'][1])
    ax.set_ylim(data['ylim'][0], data['ylim'][1])
    ax.set_xlabel(data['labels']['xlabel'])
    ax.set_ylabel(data['labels']['ylabel'])
    ax.set_title(data['labels']['title'])

    if show_cbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(data['labels']['clabel'])

    ax = add_lines(ax, data, **kwargs)

    plt.show()

    return ax, fig



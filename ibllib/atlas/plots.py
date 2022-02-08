"""
Module that has convenience plotting functions for 2D atlas slices
"""

from ibllib.atlas import AllenAtlas
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_scalar_on_slice(regions, values, coord=-1000, slice='coronal', mapping='Allen', hemisphere='left',
                         cmap='viridis', background='image', clevels=None, brain_atlas=None, ax=None):
    """
    Function to plot scalar value per allen region on histology slice
    :param regions: array of acronyms of Allen regions
    :param values: array of scalar value per acronym. If hemisphere is 'both' and different values want to be shown on each
    hemispheres, values should contain 2 columns, 1st column for LH values, 2nd column for RH values
    :param coord: coordinate of slice in um (not needed when slice='top')
    :param slice: orientation of slice, options are 'coronal', 'sagittal', 'horizontal', 'top' (top view of brain)
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cim, cmax]
    :param brain_atlas: AllenAtlas object
    :param ax: optional axis object to plot on
    :return:
    """

    if clevels is None:
        clevels = (np.min(values), np.max(values))

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    # Find the mapping to use
    map_ext = '-lr'
    map = mapping + map_ext

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][0]] = vR
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][1]] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v

        lr_divide = int((br.id.shape[0] - 1) / 2)
        if hemisphere == 'left':
            region_values[0:lr_divide] = np.nan
        elif hemisphere == 'right':
            region_values[lr_divide:] = np.nan
            region_values[0] = np.nan

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)

    if slice == 'coronal':

        if background == 'image':
            ba.plot_cslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_cslice(coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_cslice(coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_sslice(coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_sslice(coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_hslice(coord / 1e6, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_hslice(coord / 1e6, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
        else:
            ba.plot_top(volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
            ba.plot_top(volume='boundary', mapping=map, ax=ax,
                        cmap=cmap_bound, vmin=0.01, vmax=0.8)

    return fig, ax

"""
Module that has convenience plotting functions for 2D atlas slices
"""

from ibllib.atlas import AllenAtlas, FlatMap
from ibllib.atlas.regions import BrainRegions
from iblutil.numerical import ismember
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def prepare_lr_data(acronyms_lh, values_lh, acronyms_rh, values_rh):
    """
    Prepare data in format needed for plotting when providing different region values per hemisphere

    :param acronyms_lh: array of acronyms on left hemisphere
    :param values_lh: values for each acronym on left hemisphere
    :param acronyms_rh: array of acronyms on right hemisphere
    :param values_rh: values for each acronym on left hemisphere
    :return: combined acronyms and two column array of values
    """

    acronyms = np.unique(np.r_[acronyms_lh, acronyms_rh])
    values = np.nan * np.ones((acronyms.shape[0], 2))
    _, l_idx = ismember(acronyms_lh, acronyms)
    _, r_idx = ismember(acronyms_rh, acronyms)
    values[l_idx, 0] = values_lh
    values[r_idx, 1] = values_rh

    return acronyms, values


def reorder_data(acronyms, values, brain_regions=None):
    """
    Reorder list of acronyms and values to match the Allen ordering
    :param acronyms: array of acronyms
    :param values: array of values
    :param brain_regions: BrainRegions object
    :return: ordered array of acronyms and values
    """

    br = brain_regions or BrainRegions()
    atlas_id = br.acronym2id(acronyms, hemisphere='right')
    all_ids = br.id[br.order][:br.n_lr + 1]
    ordered_ids = np.zeros_like(all_ids) * np.nan
    ordered_values = np.zeros_like(all_ids) * np.nan
    _, idx = ismember(atlas_id, all_ids)
    ordered_ids[idx] = atlas_id
    ordered_values[idx] = values

    ordered_ids = ordered_ids[~np.isnan(ordered_ids)]
    ordered_values = ordered_values[~np.isnan(ordered_values)]
    ordered_acronyms = br.id2acronym(ordered_ids)

    return ordered_acronyms, ordered_values


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
        clevels = (np.nanmin(values), np.nanmax(values))

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    # Find the mapping to use
    if '-lr' in mapping:
        map = mapping
    else:
        map = mapping + '-lr'

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            idx = np.where(br.acronym[br.mappings[map]] == r)[0]
            idx_lh = idx[idx > br.n_lr]
            idx_rh = idx[idx <= br.n_lr]
            region_values[idx_rh] = vR
            region_values[idx_lh] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v
            if hemisphere == 'left':
                region_values[0:(br.n_lr + 1)] = np.nan
            elif hemisphere == 'right':
                region_values[br.n_lr:] = np.nan
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


def plot_scalar_on_flatmap(regions, values, depth=0, flatmap='dorsal_cortex', mapping='Allen', hemisphere='left',
                           cmap='viridis', background='boundary', clevels=None, flmap_atlas=None, ax=None):
    """
    Function to plot scalar value per allen region on histology slice
    :param regions: array of acronyms of Allen regions
    :param values: array of scalar value per acronym. If hemisphere is 'both' and different values want to be shown on each
    hemispheres, values should contain 2 columns, 1st column for LH values, 2nd column for RH values
    :param depth: depth in flatmap in um
    :param flatmap: name of flatmap (currently only option is 'dorsal_cortex')
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cim, cmax]
    :param brain_atlas: FlatMap object
    :param ax: optional axis object to plot on
    :return:
    """

    if clevels is None:
        clevels = (np.min(values), np.max(values))

    ba = flmap_atlas or FlatMap(flatmap=flatmap)
    br = ba.regions

    # Find the mapping to use
    if '-lr' in mapping:
        map = mapping
    else:
        map = mapping + '-lr'

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            idx = np.where(br.acronym[br.mappings[map]] == r)[0]
            idx_lh = idx[idx > br.n_lr]
            idx_rh = idx[idx <= br.n_lr]
            region_values[idx_rh] = vR
            region_values[idx_lh] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v
            if hemisphere == 'left':
                region_values[0:(br.n_lr + 1)] = np.nan
            elif hemisphere == 'right':
                region_values[br.n_lr:] = np.nan
                region_values[0] = np.nan

    d_idx = int(np.round(depth / ba.res_um))  # need to find nearest to 25

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if background == 'image':
        ba.plot_flatmap(d_idx, volume='image', mapping=map, ax=ax)
        ba.plot_flatmap(d_idx, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
    else:
        ba.plot_flatmap(d_idx, volume='value', region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
        ba.plot_flatmap(d_idx, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    # For circle flatmap we don't want to cut the axis
    if ba.name != 'circles':
        if hemisphere == 'left':
            ax.set_xlim(0, np.ceil(ba.flatmap.shape[1] / 2))
        elif hemisphere == 'right':
            ax.set_xlim(np.ceil(ba.flatmap.shape[1] / 2), ba.flatmap.shape[1])

    return fig, ax


def plot_scalar_on_barplot(acronyms, values, errors=None, order=True, ylim=None, ax=None, brain_regions=None):
    br = brain_regions or BrainRegions()

    if order:
        acronyms, values = reorder_data(acronyms, values, brain_regions)

    _, idx = ismember(acronyms, br.acronym)
    colours = br.rgb[idx]

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    ax.bar(np.arange(acronyms.size), values, color=colours)

    return fig, ax

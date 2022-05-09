"""
Module that has convenience plotting functions for 2D atlas slices
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic

from ibllib.atlas import AllenAtlas, FlatMap
from ibllib.atlas.regions import BrainRegions
from iblutil.numerical import ismember


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
                         background='image', cmap='viridis', clevels=None, show_cbar=False, brain_atlas=None, ax=None):
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
    :param clevels: min max color levels [cmin, cmax]
    :param show_cbar: whether or not to add colorbar to axis
    :param brain_atlas: AllenAtlas object
    :param ax: optional axis object to plot on
    :return:
    """

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    if clevels is None:
        clevels = (np.nanmin(values), np.nanmax(values))
        print(clevels)

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

    if show_cbar:
        fig, ax, cbar = _plot_slice(coord / 1e6, slice, region_values, 'value', background=background, map=map, clevels=clevels,
                                    cmap=cmap, ba=ba, ax=ax, show_cbar=show_cbar)
        return fig, ax, cbar
    else:
        fig, ax = _plot_slice(coord / 1e6, slice, region_values, 'value', background=background, map=map, clevels=clevels,
                              cmap=cmap, ba=ba, ax=ax, show_cbar=show_cbar)
        return fig, ax


def plot_scalar_on_flatmap(regions, values, depth=0, flatmap='dorsal_cortex', mapping='Allen', hemisphere='left',
                           background='boundary', cmap='viridis', clevels=None, show_cbar=False, flmap_atlas=None, ax=None):
    """
    Function to plot scalar value per allen region on flatmap slice

    :param regions: array of acronyms of Allen regions
    :param values: array of scalar value per acronym. If hemisphere is 'both' and different values want to be shown on each
    hemispheres, values should contain 2 columns, 1st column for LH values, 2nd column for RH values
    :param depth: depth in flatmap in um
    :param flatmap: name of flatmap (currently only option is 'dorsal_cortex')
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cmin, cmax]
    :param show_cbar: whether to add colorbar to axis
    :param flmap_atlas: FlatMap object
    :param ax: optional axis object to plot on
    :return:
    """

    if clevels is None:
        clevels = (np.nanmin(values), np.nanmax(values))

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

    if show_cbar:
        norm = matplotlib.colors.Normalize(vmin=clevels[0], vmax=clevels[1], clip=False)
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        return fig, ax, cbar
    else:
        return fig, ax


def plot_volume_on_slice(volume, coord=-1000, slice='coronal', mapping='Allen', background='boundary', cmap='Reds',
                         clevels=None, show_cbar=False, brain_atlas=None, ax=None):
    """
    Plot slice at through volume

    :param volume: 3D array of volume (must be same shape as brain_atlas object)
    :param coord: coordinate of slice in um
    :param slice:  orientation of slice, options are 'coronal', 'sagittal', 'horizontal'
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cmin, cmax]
    :param show_cbar: whether or not to add colorbar to axis
    :param brain_atlas: AllenAtlas object
    :param ax: optional axis object to plot on
    :return:
    """

    ba = brain_atlas or AllenAtlas()
    assert volume.shape == ba.image.shape, 'Volume must have same shape as ba'

    # Find the mapping to use
    if '-lr' in mapping:
        map = mapping
    else:
        map = mapping + '-lr'

    if show_cbar:
        fig, ax, cbar = _plot_slice(coord / 1e6, slice, volume, 'volume', background=background, map=map, clevels=clevels,
                                    cmap=cmap, ba=ba, ax=ax, show_cbar=show_cbar)
        return fig, ax, cbar
    else:
        fig, ax = _plot_slice(coord / 1e6, slice, volume, 'volume', background=background, map=map, clevels=clevels,
                              cmap=cmap, ba=ba, ax=ax, show_cbar=show_cbar)
        return fig, ax


def plot_points_on_slice(xyz, values=None, coord=-1000, slice='coronal', mapping='Allen', background='boundary', cmap='Reds',
                         clevels=None, show_cbar=False, aggr='mean', fwhm=100, brain_atlas=None, ax=None):
    """
    Plot xyz points on slice. Points that lie in the same voxel within slice are aggregated according to method specified.
    A 3D Gaussian smoothing kernel with distance specified by fwhm is applied to images.

    :param xyz: 3 column array of xyz coordinates of points in metres
    :param values: array of values per xyz coordinates, if no values are given the sum of xyz points in each voxel is
    returned
    :param coord: coordinate of slice in um (not needed when slice='top')
    :param slice: orientation of slice, options are 'coronal', 'sagittal', 'horizontal', 'top' (top view of brain)
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cmin, cmax]
    :param show_cbar: whether or not to add colorbar to axis
    :param aggr: aggregation method. Options are sum, count, mean, std, median, min and max.
    Can also give in custom function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html)
    :param fwhm: fwhm distance of gaussian kernel in um
    :param brain_atlas: AllenAtlas object
    :param ax: optional axis object to plot on

    :return:
    """

    ba = brain_atlas or AllenAtlas()

    # Find the mapping to use
    if '-lr' in mapping:
        map = mapping
    else:
        map = mapping + '-lr'

    region_values = compute_volume_from_points(xyz, values, aggr=aggr, fwhm=fwhm, ba=ba)

    if show_cbar:
        fig, ax, cbar = _plot_slice(coord / 1e6, slice, region_values, 'volume', background=background, map=map, clevels=clevels,
                                    cmap=cmap, ba=ba, ax=ax, show_cbar=show_cbar)
        return fig, ax, cbar
    else:
        fig, ax = _plot_slice(coord / 1e6, slice, region_values, 'volume', background=background, map=map, clevels=clevels,
                              cmap=cmap, ba=ba, ax=ax, show_cbar=show_cbar)
        return fig, ax


def compute_volume_from_points(xyz, values=None, aggr='sum', fwhm=100, ba=None):
    """
    Creates a 3D volume with xyz points placed in corresponding voxel in volume. Points that fall into the same voxel within the
    volume are aggregated according to the method specified in aggr. Gaussian smoothing with a 3D kernel with distance specified
    by fwhm (full width half max) argument is applied. If fwhm = 0, no gaussian smoothing is applied.

    :param xyz: 3 column array of xyz coordinates of points in metres
    :param values: 1 column array of values per xyz coordinates, if no values are given the sum of xyz points in each voxel is
    returned
    :param aggr: aggregation method. Options are sum, count, mean, std, median, min and max. Can also give in custom function
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html)
    :param fwhm: full width at half maximum of gaussian kernel in um
    :param ba: AllenAtlas object
    :return:
    """

    ba = ba or AllenAtlas()

    idx = ba._lookup(xyz)
    ba_shape = ba.image.shape[0] * ba.image.shape[1] * ba.image.shape[2]

    if values is not None:
        volume = binned_statistic(idx, values, range=[0, ba_shape], statistic=aggr, bins=ba_shape).statistic
        volume[np.isnan(volume)] = 0
    else:
        volume = np.bincount(idx, minlength=ba_shape, weights=values)

    volume = volume.reshape(ba.image.shape[0], ba.image.shape[1], ba.image.shape[2]).astype(np.float32)

    if fwhm > 0:
        # Compute sigma used for gaussian kernel
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))
        sigma = fwhm / (fwhm_over_sigma_ratio * ba.res_um)
        # TODO to speed up only apply gaussian filter on slices within distance of chosen coordinate
        volume = gaussian_filter(volume, sigma=sigma)

    # Mask so that outside of the brain is set to nan
    volume[ba.label == 0] = np.nan

    return volume


def _plot_slice(coord, slice, region_values, vol_type, background='boundary', map='Allen', clevels=None, cmap='viridis',
                show_cbar=False, ba=None, ax=None):

    ba = ba or AllenAtlas()

    if clevels is None:
        clevels = (np.nanmin(region_values), np.nanmax(region_values))

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)
    else:
        cmap_bound = None

    if slice == 'coronal':
        if background == 'image':
            ba.plot_cslice(coord, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_cslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_cslice(coord, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_sslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_sslice(coord, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_hslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_hslice(coord, volume='boundary', mapping=map, ax=ax, cmap=cmap_bound, vmin=0.01, vmax=0.8)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
        else:
            ba.plot_top(volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
            ba.plot_top(volume='boundary', mapping=map, ax=ax,
                        cmap=cmap_bound, vmin=0.01, vmax=0.8)

    if show_cbar:
        norm = matplotlib.colors.Normalize(vmin=clevels[0], vmax=clevels[1], clip=False)
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        return fig, ax, cbar
    else:
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

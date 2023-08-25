"""
Module that has convenience plotting functions for 2D atlas slices and flatmaps.
"""
import copy
import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Polygon, PathPatch
import matplotlib.path as mpath
from iblutil.io.hashfile import md5
import one.remote.aws as aws

from ibllib.atlas import AllenAtlas
from ibllib.atlas.flatmaps import FlatMap, _swanson_labels_positions, swanson, swanson_json
from ibllib.atlas.regions import BrainRegions
from iblutil.numerical import ismember
from ibllib.atlas.atlas import BrainCoordinates, ALLEN_CCF_LANDMARKS_MLAPDV_UM

_logger = logging.getLogger(__name__)


def get_bc_10():
    """
    Get BrainCoordinates object for 10um Allen Atlas

    Returns
    -------
    BrainCoordinates object
    """
    dims2xyz = np.array([1, 0, 2])
    res_um = 10
    scaling = np.array([1, 1, 1])
    image_10 = np.array([1320, 1140, 800])

    iorigin = (ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / res_um)
    dxyz = res_um * 1e-6 * np.array([1, -1, -1]) * scaling
    nxyz = np.array(image_10)[dims2xyz]
    bc = BrainCoordinates(nxyz=nxyz, xyz0=(0, 0, 0), dxyz=dxyz)
    bc = BrainCoordinates(nxyz=nxyz, xyz0=-bc.i2xyz(iorigin), dxyz=dxyz)

    return bc


def plot_polygon(ax, xy, color, reg_id, edgecolor='k', linewidth=0.3, alpha=1):
    """
    Function to plot matplotlib polygon on an axis

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    xy: numpy.array
        2D array of x and y coordinates of vertices of polygon
    color: str, tuple of int
        The color to fill the polygon
    reg_id: str, int
        An id to assign to the polygon
    edgecolor: str, tuple of int
        The color of the edge of the polgon
    linewidth: int
        The width of the edges of the polygon
    alpha: float between 0 and 1
        The opacitiy of the polygon

    Returns
    -------

    """
    p = Polygon(xy, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, gid=f'region_{reg_id}')
    ax.add_patch(p)


def plot_polygon_with_hole(ax, vertices, codes, color, reg_id, edgecolor='k', linewidth=0.3, alpha=1):
    """
    Function to plot matplotlib polygon that contains a hole on an axis

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    vertices: numpy.array
        2D array of x and y coordinates of vertices of polygon
    codes: numpy.array
        1D array of path codes used to link the vertices
        (https://matplotlib.org/stable/tutorials/advanced/path_tutorial.html)
    color: str, tuple of int
        The color to fill the polygon
    reg_id: str, int
        An id to assign to the polygon
    edgecolor: str, tuple of int
        The color of the edge of the polgon
    linewidth: int
        The width of the edges of the polygon
    alpha: float between 0 and 1
        The opacitiy of the polygon

    Returns
    -------

    """

    path = mpath.Path(vertices, codes)
    patch = PathPatch(path, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, gid=f'region_{reg_id}')
    ax.add_patch(patch)


def coords_for_poly_hole(coords):
    """
    Function to convert

    Parameters
    ----------
    coords : dict
        Dictionary containing keys x, y and invert. x and y contain numpy.array of x coordinates, y coordinates
        for the vertices of the polgyon. The invert key is either 1 or -1 and deterimine how to assign the paths.
        The value for invert for each polygon was assigned manually after looking at the result

    Returns
    -------
    all_coords: numpy.array
        2D array of x and y coordinates of vertices of polygon
    all_codes: numpy.array
        1D array of path codes used to link the vertices
        (https://matplotlib.org/stable/tutorials/advanced/path_tutorial.html)

    """
    for i, c in enumerate(coords):
        xy = np.c_[c['x'], c['y']]
        codes = np.ones(len(xy), dtype=mpath.Path.code_type) * mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO
        if i == 0:
            val = c.get('invert', 1)
            all_coords = xy[::val]
            all_codes = codes
        else:
            codes[-1] = mpath.Path.CLOSEPOLY
            val = c.get('invert', -1)
            all_coords = np.concatenate((all_coords, xy[::val]))
            all_codes = np.concatenate((all_codes, codes))

    return all_coords, all_codes


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
    Reorder list of acronyms and values to match the Allen ordering.

    TODO Document more

    Parameters
    ----------
    acronyms : array_like of str
        The acronyms to match the Allen ordering, whatever that means.
    values : array_like
        An array of some sort of values I guess...
    brain_regions : ibllib.atlas.regions.BrainRegions
        A brain regions object.

    Returns
    -------
    numpy.array of str
        An ordered array of acronyms
    numpy.array
        An ordered array of values. I don't know what those values are, not IDs, so maybe indices?
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


def load_slice_files(slice, mapping):
    """
    Function to load in set of vectorised atlas slices for a given atlas axis and mapping.

    If the data does not exist locally, it will download the files automatically stored in a AWS S3
    bucket.

    Parameters
    ----------
    slice : {'coronal', 'sagittal', 'horizontal', 'top'}
        The axis of the atlas to load.
    mapping : {'Allen', 'Beryl', 'Cosmos'}
        The mapping to load.

    Returns
    -------
    slice_data : numpy.array
        A json containing the vertices to draw each region for each slice in the Allen annotation volume.

    """
    OLD_MD5 = {
        'coronal': [],
        'sagittal': [],
        'horizontal': [],
        'top': []
    }

    slice_file = AllenAtlas._get_cache_dir().parent.joinpath('svg', f'{slice}_{mapping}_paths.npy')
    if not slice_file.exists() or md5(slice_file) in OLD_MD5[slice]:
        slice_file.parent.mkdir(exist_ok=True, parents=True)
        _logger.info(f'downloading swanson paths from {aws.S3_BUCKET_IBL} s3 bucket...')
        aws.s3_download_file(f'atlas/{slice_file.name}', slice_file)

    slice_data = np.load(slice_file, allow_pickle=True)

    return slice_data


def _plot_slice_vector(coords, slice, values, mapping, empty_color='silver', clevels=None, cmap='viridis', show_cbar=False,
                       ba=None, ax=None, slice_json=None, **kwargs):
    """
    Function to plot scalar value per allen region on vectorised version of histology slice. Do not use directly but use
    through plot_scalar_on_slice function with vector=True.

    Parameters
    ----------
    coords: float
        Coordinate of slice in um (not needed when slice='top').
    slice: {'coronal', 'sagittal', 'horizontal', 'top'}
        The axis through the atlas volume to display.
    values: numpy.array
        Array of values for each of the lateralised Allen regions found using BrainRegions().acronym. If no
        value is assigned to the acronym, the value at corresponding to that index should be NaN.
    mapping: {'Allen', 'Beryl', 'Cosmos'}
        The mapping to use.
    empty_color: str, tuple of int, default='silver'
        The color used to fill the regions that do not have any values assigned (regions with NaN).
    clevels: numpy.array, list or tuple
        The min and max values to use for the colormap.
    cmap: string
        Colormap to use.
    show_cbar: bool, default=False
        Whether to display a colorbar.
    ba : ibllib.atlas.AllenAtlas
        A brain atlas object.
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    slice_json: numpy.array
        The set of vectorised slices for this slice, obtained using load_slice_files(slice, mapping).
    **kwargs
        Set of kwargs passed into matplotlib.patches.Polygon.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The plotted figure.
    ax: matplotlib.pyplot.Axes
        The plotted axes.
    cbar: matplotlib.pyplot.colorbar, optional
        matplotlib colorbar object, only returned if show_cbar=True

    """
    ba = ba or AllenAtlas()
    mapping = mapping.split('-')[0].lower()
    if clevels is None:
        clevels = (np.nanmin(values), np.nanmax(values))

    if ba.res_um == 10:
        bc10 = ba.bc
    else:
        bc10 = get_bc_10()

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_axis_off()
    else:
        fig = ax.get_figure()

    colormap = cm.get_cmap(cmap)
    norm = colors.Normalize(vmin=clevels[0], vmax=clevels[1])
    nan_vals = np.isnan(values)
    rgba_color = np.full((values.size, 4), fill_value=np.nan)
    rgba_color[~nan_vals] = colormap(norm(values[~nan_vals]), bytes=True)

    if slice_json is None:
        slice_json = load_slice_files(slice, mapping)

    if slice == 'coronal':
        idx = bc10.y2i(coords)
        xlim = np.array([0, bc10.nx])
        ylim = np.array([0, bc10.nz])
    elif slice == 'sagittal':
        idx = bc10.x2i(coords)
        xlim = np.array([0, bc10.ny])
        ylim = np.array([0, bc10.nz])
    elif slice == 'horizontal':
        idx = bc10.z2i(coords)
        xlim = np.array([0, bc10.nx])
        ylim = np.array([0, bc10.ny])
    else:
        # top case
        xlim = np.array([0, bc10.nx])
        ylim = np.array([0, bc10.ny])

    if slice != 'top':
        slice_json = slice_json.item().get(str(int(idx)))

    for i, reg in enumerate(slice_json):
        color = rgba_color[reg['thisID']]
        if any(np.isnan(color)):
            color = empty_color
        else:
            color = color / 255
        coords = reg['coordsReg']

        if len(coords) == 0:
            continue

        if isinstance(coords, (list, tuple)):
            vertices, codes = coords_for_poly_hole(coords)
            plot_polygon_with_hole(ax, vertices, codes, color, **kwargs)
        else:
            xy = np.c_[coords['x'], coords['y']]
            plot_polygon(ax, xy, color, **kwargs)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()

    if show_cbar:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        return fig, ax, cbar
    else:
        return fig, ax


def plot_scalar_on_slice(regions, values, coord=-1000, slice='coronal', mapping=None, hemisphere='left',
                         background='image', cmap='viridis', clevels=None, show_cbar=False, empty_color='silver',
                         brain_atlas=None, ax=None, vector=False, slice_files=None, **kwargs):
    """
    Function to plot scalar value per region on histology slice.

    Parameters
    ----------
    regions : array_like
        An array of brain region acronyms.
    values : numpy.array
        An array of scalar value per acronym. If hemisphere is 'both' and different values want to
        be shown on each hemisphere, values should contain 2 columns, 1st column for LH values, 2nd
        column for RH values.
    coord : float
        Coordinate of slice in um (not needed when slice='top').
    slice : {'coronal', 'sagittal', 'horizontal', 'top'}, default='coronal'
        Orientation of slice.
    mapping : str, optional
        Atlas mapping to use, options are depend on atlas used (see `ibllib.atlas.BrainRegions`).
        If None, the atlas default mapping is used.
    hemisphere : {'left', 'right', 'both'}, default='left'
        The hemisphere to display.
    background : {image', 'boundary'}, default='image'
        Background slice to overlay onto, options are 'image' or 'boundary'. If `vector` is false,
        this argument is ignored.
    cmap: str, default='viridis'
        Colormap to use.
    clevels : array_like
        The min and max color levels to use.
    show_cbar: bool, default=False
        Whether to display a colorbar.
    empty_color : str, default='silver'
        Color to use for regions without any values (only used when `vector` is true).
    brain_atlas : ibllib.atlas.AllenAtlas
        A brain atlas object.
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    vector : bool, default=False
        Whether to show as bitmap or vector graphic.
    slice_files: numpy.array
        The set of vectorised slices for this slice, obtained using `load_slice_files(slice, mapping)`.
    **kwargs
        Set of kwargs passed into matplotlib.patches.Polygon, e.g. linewidth=2, edgecolor='None'
        (only used when vector = True).

    Returns
    -------
    fig: matplotlib.figure.Figure
        The plotted figure.
    ax: matplotlib.pyplot.Axes
        The plotted axes.
    cbar: matplotlib.pyplot.colorbar, optional
        matplotlib colorbar object, only returned if show_cbar=True.
    """

    ba = brain_atlas or AllenAtlas()
    br = ba.regions
    mapping = mapping or br.default_mapping

    if clevels is None:
        clevels = (np.nanmin(values), np.nanmax(values))

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
        if vector:
            fig, ax, cbar = _plot_slice_vector(coord / 1e6, slice, region_values, map, clevels=clevels, cmap=cmap, ba=ba,
                                               ax=ax, empty_color=empty_color, show_cbar=show_cbar, slice_json=slice_files,
                                               **kwargs)
        else:
            fig, ax, cbar = _plot_slice(coord / 1e6, slice, region_values, 'value', background=background, map=map,
                                        clevels=clevels, cmap=cmap, ba=ba, ax=ax, show_cbar=show_cbar)
        return fig, ax, cbar
    else:
        if vector:
            fig, ax = _plot_slice_vector(coord / 1e6, slice, region_values, map, clevels=clevels, cmap=cmap, ba=ba,
                                         ax=ax, empty_color=empty_color, show_cbar=show_cbar, slice_json=slice_files, **kwargs)
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
        cmap_bound = cm.get_cmap("bone_r").copy()
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
        norm = colors.Normalize(vmin=clevels[0], vmax=clevels[1], clip=False)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        return fig, ax, cbar
    else:
        return fig, ax


def plot_volume_on_slice(volume, coord=-1000, slice='coronal', mapping='Allen', background='boundary', cmap='Reds',
                         clevels=None, show_cbar=False, brain_atlas=None, ax=None):
    """
    Plot slice through a volume

    :param volume: 3D array of volume (must be same shape as brain_atlas object)
    :param coord: coordinate of slice in um
    :param slice:  orientation of slice, options are 'coronal', 'sagittal', 'horizontal'
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cmin, cmax]
    :param show_cbar: whether to add colorbar to axis
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
    :param show_cbar: whether to add colorbar to axis
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
    """
    Function to plot scalar value per allen region on histology slice.

    Do not use directly but use through plot_scalar_on_slice function.

    Parameters
    ----------
    coord: float
        coordinate of slice in um (not needed when slice='top').
    slice: {'coronal', 'sagittal', 'horizontal', 'top'}
        the axis through the atlas volume to display.
    region_values: numpy.array
        Array of values for each of the lateralised Allen regions found using BrainRegions().acronym. If no
        value is assigned to the acronym, the value at corresponding to that index should be nan.
    vol_type: 'value'
        The type of volume to be displayed, should always be 'value' if values want to be displayed.
    background: {'image', 'boundary'}
        The background slice to overlay the values onto. When 'image' it uses the Allen dwi image, when
        'boundary' it displays the boundaries between regions.
    map: {'Allen', 'Beryl', 'Cosmos'}
        the mapping to use.
    clevels: numpy.array, list or tuple
        The min and max values to use for the colormap.
    cmap: str, default='viridis'
        Colormap to use.
    show_cbar: bool, default=False
        Whether to display a colorbar.
    ba : ibllib.atlas.AllenAtlas
        A brain atlas object.
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The plotted figure
    ax: matplotlib.pyplot.Axes
        The plotted axes.
    cbar: matplotlib.pyplot.colorbar
        matplotlib colorbar object, only returned if show_cbar=True.

    """
    ba = ba or AllenAtlas()

    if clevels is None:
        clevels = (np.nanmin(region_values), np.nanmax(region_values))

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if slice == 'coronal':
        if background == 'image':
            ba.plot_cslice(coord, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_cslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_cslice(coord, volume='boundary', mapping=map, ax=ax)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_sslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_sslice(coord, volume='boundary', mapping=map, ax=ax)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
        else:
            ba.plot_hslice(coord, volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                           vmax=clevels[1], ax=ax)
            ba.plot_hslice(coord, volume='boundary', mapping=map, ax=ax)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
        else:
            ba.plot_top(volume=vol_type, region_values=region_values, mapping=map, cmap=cmap, vmin=clevels[0],
                        vmax=clevels[1], ax=ax)
            ba.plot_top(volume='boundary', mapping=map, ax=ax)

    if show_cbar:
        norm = colors.Normalize(vmin=clevels[0], vmax=clevels[1], clip=False)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        return fig, ax, cbar
    else:
        return fig, ax


def plot_scalar_on_barplot(acronyms, values, errors=None, order=True, ax=None, brain_regions=None):
    """
    Function to plot scalar value per allen region on a bar plot. If order=True, the acronyms and values are reordered
    according to the order defined in the Allen structure tree

    Parameters
    ----------
    acronyms: numpy.array
        A 1D array of acronyms
    values: numpy.array
        A 1D array of values corresponding to each acronym in the acronyms array
    errors: numpy.array
        A 1D array of error values corresponding to each acronym in the acronyms array
    order: bool, default=True
        Whether to order the acronyms according to the order defined by the Allen structure tree
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    brain_regions : ibllib.atlas.regions.BrainRegions
        A brain regions object

    Returns
    -------
    fig: matplotlib.figure.Figure
        The plotted figure
    ax: matplotlib.pyplot.Axes
        The plotted axes.

    """
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


def plot_swanson_vector(acronyms=None, values=None, ax=None, hemisphere=None, br=None, orientation='landscape',
                        empty_color='silver', vmin=None, vmax=None, cmap='viridis', annotate=False, annotate_n=10,
                        annotate_order='top', annotate_list=None, mask=None, mask_color='w', fontsize=10, **kwargs):
    """
    Function to plot scalar value per allen region on the swanson projection. Plots on a vecortised version of the
    swanson projection

    Parameters
    ----------
    acronyms: numpy.array
        A 1D array of acronyms or atlas ids
    values: numpy.array
        A 1D array of values corresponding to each acronym in the acronyms array
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    hemisphere : {'left', 'right', 'both', 'mirror'}
        The hemisphere to display.
    br : ibllib.atlas.BrainRegions
        A brain regions object.
    orientation : {landscape', 'portrait'}, default='landscape'
        The plot orientation.
    empty_color : str, tuple of int, default='silver'
        The greyscale matplotlib color code or an RGBA int8 tuple defining the filling of brain
        regions not provided.
    vmin: float
        Minimum value to restrict the colormap
    vmax: float
        Maximum value to restrict the colormap
    cmap: string
        matplotlib named colormap to use
    annotate : bool, default=False
        If true, labels the regions with acronyms.
    annotate_n: int
        The number of regions to annotate
    annotate_order: {'top', 'bottom'}
        If annotate_n is specified, whether to annotate the n regions with the highest (top) or lowest (bottom) values
    annotate_list: numpy.array of list
        List of regions to annotate, if this is provided, if overwrites annotate_n and annotate_order
    mask: numpy.array or list
        List of regions to apply a mask to (fill them with a specific color)
    mask_color: string, tuple or list
        Color for the mask
    fontsize : int
        The annotation font size in points.
    **kwargs
        See plot_polygon and plot_polygon_with_hole.

    Returns
    -------
    matplotlib.pyplot.Axes
        The plotted axes.

    """
    br = BrainRegions() if br is None else br
    br.compute_hierarchy()
    sw_shape = (2968, 6820)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_axis_off()

    if hemisphere != 'both' and acronyms is not None and not isinstance(acronyms[0], str):
        # If negative atlas ids are passed in and we are not going to lateralise (e.g hemisphere='both')
        # transfer them over to one hemisphere
        acronyms = np.abs(acronyms)

    if acronyms is not None:
        ibr, vals = br.propagate_down(acronyms, values)
        colormap = cm.get_cmap(cmap)
        vmin = vmin or np.nanmin(vals)
        vmax = vmax or np.nanmax(vals)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        rgba_color = colormap(norm(vals), bytes=True)

    if mask is not None:
        imr, _ = br.propagate_down(mask, np.ones_like(mask))
    else:
        imr = []

    sw_json = swanson_json()
    if hemisphere == 'both':
        sw_rev = copy.deepcopy(sw_json)
        for sw in sw_rev:
            sw['thisID'] = sw['thisID'] + br.n_lr
        sw_json = sw_json + sw_rev

    plot_idx = []
    plot_val = []
    for i, reg in enumerate(sw_json):

        coords = reg['coordsReg']
        reg_id = reg['thisID']

        if acronyms is None:
            color = br.rgba[br.mappings['Swanson'][reg['thisID']]] / 255
            if hemisphere is None:
                col_l = None
                col_r = color
            elif hemisphere == 'left':
                col_l = empty_color if orientation == 'portrait' else color
                col_r = color if orientation == 'portrait' else empty_color
            elif hemisphere == 'right':
                col_l = color if orientation == 'portrait' else empty_color
                col_r = empty_color if orientation == 'portrait' else color
            elif hemisphere in ['both', 'mirror']:
                col_l = color
                col_r = color
        else:
            idx = np.where(ibr == reg['thisID'])[0]
            idxm = np.where(imr == reg['thisID'])[0]
            if len(idx) > 0:
                plot_idx.append(ibr[idx[0]])
                plot_val.append(vals[idx[0]])
                color = rgba_color[idx[0]] / 255
            elif len(idxm) > 0:
                color = mask_color
            else:
                color = empty_color

            if hemisphere is None:
                col_l = None
                col_r = color
            elif hemisphere == 'left':
                col_l = empty_color if orientation == 'portrait' else color
                col_r = color if orientation == 'portrait' else empty_color
            elif hemisphere == 'right':
                col_l = color if orientation == 'portrait' else empty_color
                col_r = empty_color if orientation == 'portrait' else color
            elif hemisphere == 'mirror':
                col_l = color
                col_r = color
            elif hemisphere == 'both':
                if reg_id <= br.n_lr:
                    col_l = color if orientation == 'portrait' else None
                    col_r = None if orientation == 'portrait' else color
                else:
                    col_l = None if orientation == 'portrait' else color
                    col_r = color if orientation == 'portrait' else None

        if reg['hole']:
            vertices, codes = coords_for_poly_hole(coords)
            if orientation == 'portrait':
                vertices[:, [0, 1]] = vertices[:, [1, 0]]
                if col_r is not None:
                    plot_polygon_with_hole(ax, vertices, codes, col_r, reg_id, **kwargs)
                if col_l is not None:
                    vertices_inv = np.copy(vertices)
                    vertices_inv[:, 0] = -1 * vertices_inv[:, 0] + (sw_shape[0] * 2)
                    plot_polygon_with_hole(ax, vertices_inv, codes, col_l, reg_id, **kwargs)
            else:
                if col_r is not None:
                    plot_polygon_with_hole(ax, vertices, codes, col_r, reg_id, **kwargs)
                if col_l is not None:
                    vertices_inv = np.copy(vertices)
                    vertices_inv[:, 1] = -1 * vertices_inv[:, 1] + (sw_shape[0] * 2)
                    plot_polygon_with_hole(ax, vertices_inv, codes, col_l, reg_id, **kwargs)
        else:
            coords = [coords] if isinstance(coords, dict) else coords
            for c in coords:
                if orientation == 'portrait':
                    xy = np.c_[c['y'], c['x']]
                    if col_r is not None:
                        plot_polygon(ax, xy, col_r, reg_id, **kwargs)
                    if col_l is not None:
                        xy_inv = np.copy(xy)
                        xy_inv[:, 0] = -1 * xy_inv[:, 0] + (sw_shape[0] * 2)
                        plot_polygon(ax, xy_inv, col_l, reg_id, **kwargs)
                else:
                    xy = np.c_[c['x'], c['y']]
                    if col_r is not None:
                        plot_polygon(ax, xy, col_r, reg_id, **kwargs)
                    if col_l is not None:
                        xy_inv = np.copy(xy)
                        xy_inv[:, 1] = -1 * xy_inv[:, 1] + (sw_shape[0] * 2)
                        plot_polygon(ax, xy_inv, col_l, reg_id, **kwargs)

    if orientation == 'portrait':
        ax.set_ylim(0, sw_shape[1])
        if hemisphere is None:
            ax.set_xlim(0, sw_shape[0])
        else:
            ax.set_xlim(0, 2 * sw_shape[0])
    else:
        ax.set_xlim(0, sw_shape[1])
        if hemisphere is None:
            ax.set_ylim(0, sw_shape[0])
        else:
            ax.set_ylim(0, 2 * sw_shape[0])

    if annotate:
        if annotate_list is not None:
            annotate_swanson(ax=ax, acronyms=annotate_list, orientation=orientation, br=br, thres=10, fontsize=fontsize)
        elif acronyms is not None:
            ids = br.index2id(np.array(plot_idx))
            _, indices, _ = np.intersect1d(br.id, br.remap(ids, 'Swanson-lr'), return_indices=True)
            a, b = ismember(ids, br.id[indices])
            sorted_id = ids[a]
            vals = np.array(plot_val)[a]
            sort_vals = np.argsort(vals) if annotate_order == 'bottom' else np.argsort(vals)[::-1]
            annotate_swanson(ax=ax, acronyms=sorted_id[sort_vals[:annotate_n]], orientation=orientation, br=br,
                             thres=10, fontsize=fontsize)
        else:
            annotate_swanson(ax=ax, orientation=orientation, br=br, fontsize=fontsize)

    def format_coord(x, y):
        patch = next((p for p in ax.patches if p.contains_point(p.get_transform().transform(np.r_[x, y]))), None)
        if patch is not None:
            ind = int(patch.get_gid().split('_')[1])
            ancestors = br.ancestors(br.id[ind])['acronym']
            return f'sw-{ind}, {ancestors}, aid={br.id[ind]}-{br.acronym[ind]} \n {br.name[ind]}'
        else:
            return ''

    ax.format_coord = format_coord

    ax.invert_yaxis()
    ax.set_aspect('equal')
    return ax


def plot_swanson(acronyms=None, values=None, ax=None, hemisphere=None, br=None,
                 orientation='landscape', annotate=False, empty_color='silver', **kwargs):
    """
    Displays the 2D image corresponding to the swanson flatmap.

    This case is different from the others in the sense that only a region maps to another regions,
    there is no correspondence to the spatial 3D coordinates.

    Parameters
    ----------
    acronyms: numpy.array
        A 1D array of acronyms or atlas ids
    values: numpy.array
        A 1D array of values corresponding to each acronym in the acronyms array
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    hemisphere : {'left', 'right', 'both', 'mirror'}
        The hemisphere to display.
    br : ibllib.atlas.BrainRegions
        A brain regions object.
    orientation : {landscape', 'portrait'}, default='landscape'
        The plot orientation.
    empty_color : str, tuple of int, default='silver'
        The greyscale matplotlib color code or an RGBA int8 tuple defining the filling of brain
        regions not provided.
    vmin: float
        Minimum value to restrict the colormap
    vmax: float
        Maximum value to restrict the colormap
    cmap: string
        matplotlib named colormap to use
    annotate : bool, default=False
        If true, labels the regions with acronyms.
    **kwargs
        See matplotlib.pyplot.imshow.

    Returns
    -------
    matplotlib.pyplot.Axes
        The plotted axes.
    """
    mapping = 'Swanson'
    br = BrainRegions() if br is None else br
    br.compute_hierarchy()
    s2a = swanson()
    # both hemispheres
    if hemisphere == 'both':
        _s2a = s2a + np.sum(br.id > 0)
        _s2a[s2a == 0] = 0
        _s2a[s2a == 1] = 1
        s2a = np.r_[s2a, np.flipud(_s2a)]
        mapping = 'Swanson-lr'
    elif hemisphere == 'mirror':
        s2a = np.r_[s2a, np.flipud(s2a)]
    if orientation == 'portrait':
        s2a = np.transpose(s2a)
    if acronyms is None:
        regions = br.mappings[mapping][s2a]
        im = br.rgba[regions]
        iswan = None
    else:
        ibr, vals = br.propagate_down(acronyms, values)
        # we now have the mapped regions and aggregated values, map values onto swanson map
        iswan, iv = ismember(s2a, ibr)
        im = np.zeros_like(s2a, dtype=np.float32)
        im[iswan] = vals[iv]
        im[~iswan] = np.nan
    if not ax:
        ax = plt.gca()
        ax.set_axis_off()  # unless provided we don't need scales here
    ax.imshow(im, **kwargs)
    # overlay the boundaries if value plot
    imb = np.zeros((*s2a.shape[:2], 4), dtype=np.uint8)
    # fill in the empty regions with the blank regions colours if necessary
    if iswan is not None:
        imb[~iswan] = (np.array(colors.to_rgba(empty_color)) * 255).astype('uint8')
    imb[s2a == 0] = 255
    # imb[s2a == 1] = np.array([167, 169, 172, 255])
    imb[s2a == 1] = np.array([0, 0, 0, 255])
    ax.imshow(imb)
    if annotate:
        annotate_swanson(ax=ax, orientation=orientation, br=br)

    # provides the mean to see the region on axis
    def format_coord(x, y):
        ind = s2a[int(y), int(x)]
        ancestors = br.ancestors(br.id[ind])['acronym']
        return f'sw-{ind}, {ancestors}, aid={br.id[ind]}-{br.acronym[ind]} \n {br.name[ind]}'

    ax.format_coord = format_coord
    return ax


def annotate_swanson(ax, acronyms=None, orientation='landscape', br=None, thres=20000, **kwargs):
    """
    Display annotations on a Swanson flatmap.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        An axis object to plot onto.
    acronyms : array_like
        A list or numpy array of acronyms or Allen region IDs. If None plot all acronyms.
    orientation : {landscape', 'portrait'}, default='landscape'
        The plot orientation.
    br : ibllib.atlas.BrainRegions
        A brain regions object.
    thres : int, default=20000
        The number of pixels above which a region is labelled.
    **kwargs
        See matplotlib.pyplot.Axes.annotate.

    """
    br = br or BrainRegions()
    if acronyms is None:
        indices = np.arange(br.id.size)
    else:  # TODO we should in fact remap and compute labels for hierarchical regions
        aids = br.parse_acronyms_argument(acronyms)
        _, indices, _ = np.intersect1d(br.id, br.remap(aids, 'Swanson-lr'), return_indices=True)
    labels = _swanson_labels_positions(thres=thres)
    for ilabel in labels:
        # do not display unwanted labels
        if ilabel not in indices:
            continue
        # rotate the labels if the display is in portrait mode
        xy = np.flip(labels[ilabel]) if orientation == 'portrait' else labels[ilabel]
        ax.annotate(br.acronym[ilabel], xy=xy, ha='center', va='center', **kwargs)

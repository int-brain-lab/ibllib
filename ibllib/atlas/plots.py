"""
Module that has convenience plotting functions for 2D atlas slices and flatmaps.
"""

import iblatlas.plots as atlas_plots
from ibllib.atlas import deprecated_decorator


@deprecated_decorator
def prepare_lr_data(acronyms_lh, values_lh, acronyms_rh, values_rh):
    """
    Prepare data in format needed for plotting when providing different region values per hemisphere

    :param acronyms_lh: array of acronyms on left hemisphere
    :param values_lh: values for each acronym on left hemisphere
    :param acronyms_rh: array of acronyms on right hemisphere
    :param values_rh: values for each acronym on left hemisphere
    :return: combined acronyms and two column array of values
    """

    return atlas_plots.prepare_lr_data(acronyms_lh, values_lh, acronyms_rh, values_rh)


@deprecated_decorator
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

    return atlas_plots.reorder_data(acronyms, values, brain_regions=brain_regions)


@deprecated_decorator
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

    return (atlas_plots.plot_scalar_on_slice(regions, values, coord=coord, slice=slice, mapping=mapping,
                                             hemisphere=hemisphere, background=background, cmap=cmap, clevels=clevels,
                                             show_cbar=show_cbar, empty_color=empty_color, brain_atlas=brain_atlas,
                                             ax=ax, vector=vector, slice_files=slice_files, **kwargs))


@deprecated_decorator
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

    return atlas_plots.plot_scalar_on_slice(regions, values, depth=depth, flatmap=flatmap, mapping=mapping,
                                            hemisphere=hemisphere, background=background, cmap=cmap, clevels=clevels,
                                            show_cbar=show_cbar, flmap_atlas=flmap_atlas, ax=ax)


@deprecated_decorator
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

    return atlas_plots.plot_volume_on_slice(volume, coord=coord, slice=slice, mapping=mapping, background=background,
                                            cmap=cmap, clevels=clevels, show_cbar=show_cbar, brain_atlas=brain_atlas,
                                            ax=ax)


@deprecated_decorator
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

    return atlas_plots.plot_points_on_slice(xyz, values=values, coord=coord, slice=slice, mapping=mapping,
                                            background=background, cmap=cmap, clevels=clevels, show_cbar=show_cbar,
                                            aggr=aggr, fwhm=fwhm, brain_atlas=brain_atlas, ax=ax)


@deprecated_decorator
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

    return atlas_plots.plot_scalar_on_barplot(acronyms, values, errors=errors, order=order, ax=ax,
                                              brain_regions=brain_regions)


@deprecated_decorator
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

    return atlas_plots.plot_swanson_vector(acronyms=acronyms, values=values, ax=ax, hemisphere=hemisphere, br=br,
                                           orientation=orientation, empty_color=empty_color, vmin=vmin, vmax=vmax,
                                           cmap=cmap, annotate=annotate, annotate_n=annotate_n,
                                           annotate_order=annotate_order, annotate_list=annotate_list, mask=mask,
                                           mask_color=mask_color, fontsize=fontsize, **kwargs)


@deprecated_decorator
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

    return atlas_plots.plot_swanson(acronyms=acronyms, values=values, ax=ax, hemisphere=hemisphere, br=br,
                                    orientation=orientation, annotate=annotate, empty_color=empty_color, **kwargs)

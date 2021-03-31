"""
Functions which map metrics to the Allen atlas.

Code by G. Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ibllib import atlas


def _label2values(imlabel, fill_values, ba):
    """
    Fills a slice from the label volume with values to display
    :param imlabel: 2D np-array containing label ids (slice of the label volume)
    :param fill_values: 1D np-array containing values to fill into the slice
    :return: 2D np-array filled with values
    """
    im_unique, ilabels, iim = np.unique(imlabel, return_index=True, return_inverse=True)
    _, ir_unique, _ = np.intersect1d(ba.regions.id, im_unique, return_indices=True)
    im = np.squeeze(np.reshape(fill_values[ir_unique[iim]], (*imlabel.shape, 1)))
    return im


def plot_atlas(regions, values, ML=-1, AP=0, DV=-1, hemisphere='left', color_palette='Reds',
               minmax=None, axs=None, custom_region_list=None):
    """
    Plot a sagittal, coronal and horizontal slice of the Allen atlas with regions colored in
    according to any value that the user specifies.

    Parameters
    ----------
    regions : 1D array
        Array of strings with the acronyms of brain regions (in Allen convention) that should be
        filled with color
    values : 1D array
        Array of values that correspond to the brain region acronyms
    ML, AP, DV : float
        The coordinates of the slices in mm
    hemisphere : string
        Which hemisphere to color, options are 'left' (default), 'right', 'both'
    color_palette : any input that can be interpreted by sns.color_palette
        The color palette of the plot
    minmax : 2 element array
        The min and max of the color map, if None it uses the min and max of values
    axs : 3 element list of axis
        A list of the three axis in which to plot the three slices
    custom_region_list : 1D array with shape the same as ba.regions.acronym.shape
        Input any custom list of acronyms that replaces the default list of acronyms
        found in ba.regions.acronym. For example if you want to merge certain regions you can
        give them the same name in the custom_region_list
    """

    # Import Allen atlas
    ba = atlas.AllenAtlas(25)

    # Check input
    assert regions.shape == values.shape
    if minmax is not None:
        assert len(minmax) == 2
    if axs is not None:
        assert len(axs) == 3
    if custom_region_list is not None:
        assert custom_region_list.shape == ba.regions.acronym.shape

    # Get region boundaries volume
    boundaries = np.diff(ba.label, axis=0, append=0)
    boundaries = boundaries + np.diff(ba.label, axis=1, append=0)
    boundaries = boundaries + np.diff(ba.label, axis=2, append=0)
    boundaries[boundaries != 0] = 1

    # Get all brain region names, use custom list if inputted
    if custom_region_list is None:
        all_regions = ba.regions.acronym
    else:
        all_regions = custom_region_list

    # Set values outside colormap bounds
    if minmax is not None:
        values[values < minmax[0] + np.abs(minmax[0] / 1000)] = (minmax[0]
                                                                 + np.abs(minmax[0] / 1000))
        values[values > minmax[1] - np.abs(minmax[1] / 1000)] = (minmax[1]
                                                                 - np.abs(minmax[0] / 1000))

    # Add values to brain region list
    region_values = np.ones(ba.regions.acronym.shape) * (np.min(values) - (np.max(values) + 1))
    for i, region in enumerate(regions):
        region_values[all_regions == region] = values[i]

    # Set 'void' to default white
    region_values[0] = np.min(values) - (np.max(values) + 1)

    # Get slices with fill values
    slice_sag = ba.slice(ML / 1000, axis=0, volume=ba.label)  # saggital
    slice_sag = _label2values(slice_sag, region_values, ba)
    bound_sag = ba.slice(ML / 1000, axis=0, volume=boundaries)
    slice_cor = ba.slice(AP / 1000, axis=1, volume=ba.label)  # coronal
    slice_cor = _label2values(slice_cor, region_values, ba)
    bound_cor = ba.slice(AP / 1000, axis=1, volume=boundaries)
    slice_hor = ba.slice(DV / 1000, axis=2, volume=ba.label)  # horizontal
    slice_hor = _label2values(slice_hor, region_values, ba)
    bound_hor = ba.slice(DV / 1000, axis=2, volume=boundaries)

    # Only color specified hemisphere
    if hemisphere == 'left':
        slice_cor[:int(slice_cor.shape[0] / 2), :] = np.min(values) - (np.max(values) + 1)
        slice_hor[:, int(slice_cor.shape[0] / 2):] = np.min(values) - (np.max(values) + 1)
    elif hemisphere == 'right':
        slice_cor[int(slice_cor.shape[0] / 2):, :] = np.min(values) - (np.max(values) + 1)
        slice_hor[:, :int(slice_cor.shape[0] / 2)] = np.min(values) - (np.max(values) + 1)
    if ((hemisphere == 'left') & (ML > 0)) or ((hemisphere == 'right') & (ML < 0)):
        slice_sag[:] = np.min(values) - (np.max(values) + 1)

    # Add boundaries to slices outside of the fill value region and set to grey
    if minmax is None:
        slice_sag[bound_sag == 1] = np.max(values) + 1
        slice_cor[bound_cor == 1] = np.max(values) + 1
        slice_hor[bound_hor == 1] = np.max(values) + 1
    else:
        slice_sag[bound_sag == 1] = minmax[1] + 1
        slice_cor[bound_cor == 1] = minmax[1] + 1
        slice_hor[bound_hor == 1] = minmax[1] + 1

    # Construct color map
    color_map = sns.color_palette(color_palette, 1000)
    color_map.append((0.8, 0.8, 0.8))  # color of the boundaries between regions
    color_map.insert(0, (1, 1, 1))  # color of the background and regions without a value

    # Get color scale
    if minmax is None:
        cmin = np.min(values)
        cmax = np.max(values)
    else:
        cmin = minmax[0]
        cmax = minmax[1]

    # Plot
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    # Saggital
    sns.heatmap(np.rot90(slice_sag, 3), cmap=color_map, cbar=True, vmin=cmin, vmax=cmax, ax=axs[0])
    axs[0].set(title='ML: %.1f mm' % ML)
    plt.axis('off')
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    # Coronal
    sns.heatmap(np.rot90(slice_cor, 3), cmap=color_map, cbar=True, vmin=cmin, vmax=cmax, ax=axs[1])
    axs[1].set(title='AP: %.1f mm' % AP)
    plt.axis('off')
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    # Horizontal
    sns.heatmap(np.rot90(slice_hor, 3), cmap=color_map, cbar=True, vmin=cmin, vmax=cmax, ax=axs[2])
    axs[2].set(title='DV: %.1f mm' % DV)
    plt.axis('off')
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

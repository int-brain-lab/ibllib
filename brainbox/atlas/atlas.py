'''
Functions which map metrics to the Allen atlas.

Code by G. Meijer
'''

import numpy as np
import seaborn as sns
from ibllib import atlas
     
       
def _get_slice(coordinate, axis, fill_values):
    ''
    index = ba.bc.xyz2i(np.array([coordinate / 1000] * 3))[axis]
    imlabel = ba.label.take(index, axis=ba.xyz2dims[axis])
    im_unique, ilabels, iim = np.unique(imlabel, return_index=True, return_inverse=True)
    _, ir_unique, _ = np.intersect1d(ba.regions.id, im_unique, return_indices=True)
    im = np.squeeze(np.reshape(fill_values[ir_unique[iim]], (*imlabel.shape, 1)))
    return im


def plot_atlas(regions, values, AP=0, ML=-1, DV=-1, color_palette='Reds', axs=None,
               custom_region_list=None):
    
    # Check input
    assert regions.shape == values.shape
    if custom_region_list is not None:
        assert len(custom_region_list) == ba.regions.id.shape        
    
    # Import Allen atlas
    ba = atlas.AllenAtlas(25)
    
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
        
    # Add values to brain region list
    region_values = np.ones(ba.regions.id.shape) * (np.min(values) - 1)
    for i, region in enumerate(regions):
        region_values[all_regions == region] = values[i]
        
    # Get slices with fill values
    slice_sag = _get_slice(ML, 0, region_values)  # saggital
    slice_cor = _get_slice(AP, 1, decod_block)  # coronal
    slice_hor = _get_slice(DV, 2, decod_block)  # horizontal
    
    # Get slices with boundaries
    bound_sag = boundaries.take(ba.bc.xyz2i(np.array([coordinate / 1000] * 3))[0],
                                axis=ba.xyz2dims[0])  # saggital
    bound_cor = boundaries.take(ba.bc.xyz2i(np.array([coordinate / 1000] * 3))[0],
                                axis=ba.xyz2dims[0])  # coronal
    bound_hor = boundaries.take(ba.bc.xyz2i(np.array([coordinate / 1000] * 3))[0],
                                axis=ba.xyz2dims[0])  # horizontal
    
    # Add boundaries to slices outside of the fill value region    
    slice_sag[bound_sag == 1] = np.max(values) + 1
    slice_cor[bound_cor == 1] = np.max(values) + 1
    slice_hor[bound_hor == 1] = np.max(values) + 1
    
    # Construct color map
    color_map = sns.color_palette(color_palette, 1000)
    color_map.append((0.8, 0.8, 0.8))  # color of the boundaries between regions
    color_map.insert(0, (0, 0, 0))  # color of the background and regions without a value
    
    # Plot
    if axs is None:
        fig, axs = plt.subplots(1, 3)
    
    # Saggital
    sns.heatmap(np.rot90(slice_sag, 3), cmap=color_map, cbar=True,
                vmin=np.min(values), vmax=np.max(values), ax=axs[0])
    axs[0].set(title='AP: %.1f mm' % AP)
    plt.axis('off')
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    
    # Coronal
    sns.heatmap(np.rot90(slice_cor, 3), cmap=color_map, cbar=True,
                vmin=np.min(values), vmax=np.max(values), ax=axs[0])
    axs[1].set(title='AP: %.1f mm' % AP)
    plt.axis('off')
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    
    # Horizontal
    sns.heatmap(np.rot90(slice_hor, 3), cmap=color_map, cbar=True,
                vmin=np.min(values), vmax=np.max(values), ax=axs[0])
    axs[2].set(title='AP: %.1f mm' % AP)
    plt.axis('off')
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

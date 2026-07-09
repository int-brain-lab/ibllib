"""Registration of MPCI fields of view (FOVs) to a common reference frame.

This module provides tools for registering widefield reference stack images to one another, and
for extracting atlas coordinates from FOV mean images.
"""
import json
import enum
import logging
from pathlib import Path

from one.alf.path import ALFPath
import one.alf.io as alfio

import skimage.io
import skimage.transform
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2


_logger = logging.getLogger(__name__)
Provenance = enum.Enum('Provenance', ['ESTIMATE', 'FUNCTIONAL', 'LANDMARK', 'HISTOLOGY'])  # py3.11 make StrEnum


def get_window_center(meta):
    """Get the window offset from image center in mm.

    Previously this was not extracted in the reference stack metadata,
    but can now be found in the centerMM.x and centerMM.y fields.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The window center offset in mm (x, y).
    """
    try:
        param = next(
            x.split('=')[-1].strip() for x in meta['rawScanImageMeta']['Software'].split('\n')
            if x.startswith('SI.hDisplay.circleOffset')
        )
        return np.fromiter(map(float, param[1:-1].split()), dtype=float) / 1e3  # μm -> mm
    except StopIteration:
        return np.array([0, 0], dtype=float)


def get_px_per_um(meta):
    """Get the reference image pixel density in pixels per μm.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The reference image pixel density in pixels (y, x) per μm
    """
    if meta['rawScanImageMeta']['ResolutionUnit'].casefold() != 'centimeter':
        raise NotImplementedError('Reference image resolution unit must be in centimeters')

    yx_res = np.array([
        meta['rawScanImageMeta']['YResolution'],
        meta['rawScanImageMeta']['XResolution']
    ])
    return yx_res * 1e-4  # NB: these values are (y, x) in μm


def get_window_px(meta):
    """Get the window center and size in pixels.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The window center in pixels (y, x).
    int
        The window radius in pixels.
    numpy.array
        The reference image size in pixels (y, x).
    """
    diameter = next(
        float(x.split('=')[-1].strip()) for x in meta['rawScanImageMeta']['Software'].split('\n')
        if x.startswith('SI.hDisplay.circleDiameter')
    )
    offset = get_window_center(meta) * 1e3  # mm -> μm

    si_rois = meta['rawScanImageMeta']['Artist']['RoiGroups']['imagingRoiGroup']['rois']
    si_rois = list(filter(lambda x: x['enable'], si_rois))

    # Get the pixel size in μm from the reference image metadata
    px_per_um = get_px_per_um(meta)

    # Get image size in pixels
    # Scanfields comprise long, vertical rectangles tiled along the x-axis.
    max_y = max(fov['scanfields']['pixelResolutionXY'][1] for fov in si_rois)
    total_x = sum(fov['scanfields']['pixelResolutionXY'][0] for fov in si_rois)
    image_size = np.array([max_y, total_x], dtype=int)  # (y, x) in pixels

    diameter_px = diameter * px_per_um  # in pixels
    radius_px = np.round(diameter_px / 2).astype(int)
    center_px = np.round(np.flip(offset) * px_per_um).astype(int)  # (y, x) in pixels
    return center_px, radius_px, image_size


def preprocess_vasculature(image_stack, sigma=1.0, low_percentile=1, high_percentile=99, crop_size=390):
    """
    Preprocess image to enhance vasculature and suppress fluorescence.

    Parameters
    ----------
    image_stack : ndarray
        3D stack or 2D image.
    sigma : float
        Gaussian blur sigma for noise reduction.
    low_percentile : float
        Lower percentile for contrast stretching.
    high_percentile : float
        Upper percentile for contrast stretching.
    crop_size : int
        Size of the crop to apply to the image.  This should be the size of resulting image in pixels.

    Returns
    -------
    processed : ndarray
        Processed 2D image.
    """
    # If 3D stack, take max projection
    if image_stack.ndim == 3:
        image = np.max(image_stack, axis=0)
    else:
        image = image_stack.copy()

    # Apply median filter to reduce noise while preserving edges
    image = median_filter(image, size=3)

    # Gaussian blur to reduce high-frequency noise
    image = gaussian_filter(image, sigma=sigma)

    # Contrast stretching to enhance vessel contrast
    p_low, p_high = np.percentile(image, [low_percentile, high_percentile])
    image = np.clip((image - p_low) / (p_high - p_low), 0, 1)

    # Invert if vessels are dark (which they typically are)
    # Assume vessels are darker than background
    if np.mean(image) > 0.5:
        image = 1 - image

    # Crop the image to remove cranial window boundry and image edges
    if crop_size:
        if isinstance(crop_size, tuple):
            assert len(crop_size) == 2
            a, b = crop_size
            image = image[a, b]
        else:
            assert isinstance(crop_size, int)
            assert crop_size < min(image.shape)
            c = ((np.array(image.shape) - crop_size) / 2).astype(int)
            image = image[c[0]:(c[0] + crop_size), c[1]:(c[1] + crop_size)]

    return (image * 255).astype(np.uint8)


def register_reference_stacks(stack_path, target_stack_path, save_path=None, display=False, **kwargs):
    """Register one reference stack to another.

    Register using Enhanced Correlation Coefficient (ECC) optimization. This method can handle small rotations
    and translations, and is robust to intensity variations.

    Parameters
    ----------
    stack_path : str, pathlib.Path
        Path to the reference stack to register.  May be a session path or full file path.
    target_stack_path : str, pathlib.Path
        Path to the target stack to register to.  May be a session path or full file path.
    apply_threshold : bool, int
        Apply a binary threshold to the image before registration.  This can help with feature
        detection in noisy images.
    save_path : str, pathlib.Path
        Path to save the registration results.
    display : bool, pathlib.Path
        Display the registration results in a matplotlib figure.  If a path is provided, the
        figures will be saved to that location.

    Returns
    -------
    aligned : ndarray
        Aligned image
    params : dict
        Registration parameters including transformation details
    """
    img_data = {}
    # Load and process the two stacks
    for key, path in zip(('stack', 'target_stack'), (stack_path, target_stack_path)):
        if (path := ALFPath(path)).is_file():
            pass
        elif path.is_session_path():
            try:  # glob for reference stack in session folder
                path = next(path.glob('raw_imaging_data_??/reference/referenceImage.stack.*'))
            except StopIteration:
                raise FileNotFoundError(f'No reference stack found in session path {path}')
        else:
            raise ValueError(f'Invalid stack path {path}')
        # Load the image data
        # img_data[key] = ScanImageTiffReader(str(path)).data()
        img_data[key] = skimage.io.imread(path)
        if kwargs.get('crop_size') is True:
            try:
                # Attempt to determine crop size based on window size
                meta = alfio.load_file_content(path.with_name('referenceImage.meta.json'))
                center_px, radius_px, image_size = get_window_px(meta)
                crop_size = slice(max(0, int(center_px[0] - radius_px[0])), min(image_size[0], int(center_px[0] + radius_px[0]))), \
                            slice(max(0, int(center_px[1] - radius_px[1])), min(image_size[1], int(center_px[1] + radius_px[1])))
                kwargs['crop_size'] = crop_size
                _logger.info(f'Determined crop size for {path.session_path_short()}: {crop_size}')
            except StopIteration:
                _logger.warning(f'Could not determine crop size for {path}, using default')
                kwargs['crop_size'] = 390  # Default crop size for 5mm window at pix_per_um=0.8

        img_data[key + '_processed'] = preprocess_vasculature(img_data[key], **kwargs).astype(np.float32)

    # Calculate quality metric (normalized cross-correlation)
    ref, stack = img_data['target_stack_processed'], img_data['stack_processed']

    # Define 2x3 affine transformation matrix for Euclidean transform
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations and termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    # Run the ECC algorithm
    (cc, warp_matrix) = cv2.findTransformECC(ref, stack, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)

    # Apply the transformation to the processed image
    aligned_processed = cv2.warpAffine(
        stack, warp_matrix, (stack.shape[1], stack.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Extract parameters
    rotation = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])  # in radians
    translation = warp_matrix[:, 2]

    # Normalize images
    ref_norm = (ref - np.mean(ref)) / np.std(ref)
    aligned_norm = (aligned_processed - np.mean(aligned_processed)) / np.std(aligned_processed)

    # Calculate normalized cross-correlation
    ncc = np.mean(ref_norm * aligned_norm)

    params = {
        'translation': translation,
        'rotation': rotation,
        'correlation': cc,
        'quality_ncc': ncc,
        'warp_matrix': warp_matrix,
        'method': 'ecc'
    }
    # The same warp we will use on the MLAPDV array
    transform_robust = (skimage.transform.EuclideanTransform(rotation=params['rotation']) +
                        skimage.transform.EuclideanTransform(translation=params['translation']))
    img_data['aligned'] = skimage.transform.warp(
        np.max(img_data['stack'], axis=0), transform_robust,
        order=1, mode='constant', cval=0, clip=True, preserve_range=True)
    img_data['aligned_processed'] = skimage.transform.warp(
        img_data['stack_processed'], transform_robust,
        order=1, mode='constant', cval=0, clip=True, preserve_range=True)

    write_stack_registration_qc(img_data, params, save_path=save_path, display=display)

    return img_data['aligned'], params


def write_stack_registration_qc(img_data, params, save_path=None, display=False, plot_processed=False):
    """Write QC figure for stack registration.

    Writes an animated figure with three subplots, the first switching between aligned and
    unaligned, the second switching between reference and aligned, the third switching between the
    difference images.

    Parameters
    ----------
    img_data : dict
        The original image data.
    params : dict
        The registration parameters.
    save_path : str, optional
        The path to save the QC figure.
    display : bool, optional
        Whether to display the QC figure.
    plot_processed : bool, optional
        Whether to plot the processed images instead of the raw images.

    Returns
    -------
    save_path : str
        The path to the saved QC figure.
    fig : matplotlib.figure.Figure
        The figure object for the QC plot.

    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Reference session (*) vs session stack', fontsize=16)

    # Max project the images
    target_stack = img_data['target_stack_processed'] if plot_processed else np.max(img_data['target_stack'], axis=0)
    stack = img_data['stack_processed'] if plot_processed else np.max(img_data['stack'], axis=0)
    aligned = img_data['aligned_processed'] if plot_processed else img_data['aligned']

    # Calculate difference images
    if plot_processed:
        unaligned_diff_raw = img_data['target_stack_processed'] - img_data['stack_processed']
        aligned_diff_raw = img_data['target_stack_processed'] - img_data['aligned_processed']
        maxmax = max(np.max(unaligned_diff_raw), np.max(aligned_diff_raw))
        minmin = min(np.min(unaligned_diff_raw), np.min(aligned_diff_raw))
        unaligned_diff = (((unaligned_diff_raw - minmin) / (maxmax - minmin)) * 255).astype(np.uint8)
        aligned_diff = (((aligned_diff_raw - minmin) / (maxmax - minmin)) * 255).astype(np.uint8)
    else:
        unaligned_diff_raw = img_data['target_stack'] - img_data['stack']
        aligned_diff_raw = img_data['target_stack'] - img_data['aligned']
        maxmax = max(np.max(unaligned_diff_raw), np.max(aligned_diff_raw))
        minmin = min(np.min(unaligned_diff_raw), np.min(aligned_diff_raw))
        unaligned_diff = np.max((((unaligned_diff_raw - minmin) / (maxmax - minmin)) * 255), axis=0).astype(np.uint8)
        aligned_diff = np.max((((aligned_diff_raw - minmin) / (maxmax - minmin)) * 255), axis=0).astype(np.uint8)

    # Initial plots
    unaligned_plot = axs[0][0].imshow(target_stack)
    axs[0][0].set_title('Unaligned')
    aligned_plot = axs[0][1].imshow(target_stack)
    axs[0][1].set_title('Aligned')
    trans_plot = axs[1][0].imshow(stack)
    axs[1][0].set_title('Transform (unaligned vs aligned)')
    diff_plot = axs[1][1].imshow(unaligned_diff, cmap='grey')
    axs[1][1].set_title('Difference')

    # Add large asterisk in top left corner to indicate which image is currently displayed
    txt = [axs[0][i].text(
        0.05, 0.95, '*', color='red', fontsize=20,
        verticalalignment='top', horizontalalignment='left',
        transform=axs[0][i].transAxes) for i in range(2)]
    # Remove all axes ticks and labels
    for ax in axs.flatten():
        ax.axis('off')
    # Write params at bottom of figure
    par_text = [f'{k.capitalize()}: {v}' for k, v in params.items()]
    fig.text(0., 0.01, ', \n'.join(par_text), ha='left')
    # Reduce space between subplots
    # fig.tight_layout()

    def update(frame):
        if frame % 2 == 0:
            unaligned_plot.set_data(stack)
            aligned_plot.set_data(aligned)
            trans_plot.set_data(aligned)
            diff_plot.set_data(aligned_diff)
            for t in txt:
                t.set_text('')
        else:
            unaligned_plot.set_data(target_stack)
            aligned_plot.set_data(target_stack)
            trans_plot.set_data(stack)
            diff_plot.set_data(unaligned_diff)
            for t in txt:
                t.set_text('*')
        return aligned_plot, unaligned_plot, diff_plot

    ani = animation.FuncAnimation(fig, update, frames=10, interval=1, blit=False)
    # Save animated figure as a GIF
    if save_path:
        save_path = Path(save_path)
        if save_path.suffix != '.gif':
            save_path = save_path.with_suffix('.gif')
        ani.save(save_path, writer='pillow', fps=2)
        _logger.info(f'Saved stack registration QC to {save_path}')
        # Also save the parameters as a JSON file
        params = params.copy()
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                params[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                params[k] = float(v)
            else:
                params[k] = v
        with open(save_path.with_suffix('.json'), 'w') as fp:
            json.dump(params, fp, indent=4)

    if display:
        plt.show()
    else:
        plt.close(fig)

    return save_path, fig

"""Registration of MPCI fields of view (FOVs) to a common reference frame.

This module provides tools for registering widefield reference stack images to one another, and
for extracting atlas coordinates from FOV mean images.
"""
from one.alf.path import ALFPath
from ScanImageTiffReader import ScanImageTiffReader  # NB: use skimage if possible
from scipy.ndimage import median_filter
import skimage.transform
from skimage.feature.orb import ORB
from skimage.feature import match_descriptors
from skimage.measure import ransac
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2


def register_reference_stacks(stack_path, target_stack_path, crop_size=390, apply_threshold=False, display=False, **kwargs):
    """Register one reference stack to another.

    This uses ORB feature matching to find the euclidean transformation between two stacks.

    Parameters
    ----------
    stack_path : str, pathlib.Path
        Path to the reference stack to register.  May be a session path or full file path.
    target_stack_path : str, pathlib.Path
        Path to the target stack to register to.  May be a session path or full file path.
    crop_size : int
        Size of the crop to apply to the image before registration.  This should be the size of
        resulting image in pixels.
    apply_threshold : bool, int
        Apply a binary threshold to the image before registration.  This can help with feature
        detection in noisy images.
    display : bool, pathlib.Path
        Display the registration results in a matplotlib figure.  If a path is provided, the
        figures will be saved to that location.

    Returns
    -------
    """
    img_data = {}
    # Load and process the two stacks
    for key, path in zip(('stack', 'target_stack'), (stack_path, target_stack_path)):
        if (path := ALFPath(path)).is_file:
            pass
        elif path.is_session_path():
            try:  # glob for reference stack in session folder
                path = next(path.glob('raw_imaging_data_??/referenceImage.stack.*'))
            except StopIteration:
                raise FileNotFoundError(f'No reference stack found in session path {path}')
        else:
            raise ValueError(f'Invalid stack path {path}')
        # Load the image data
        img_data[key] = ScanImageTiffReader(str(path)).data()

        # Process image to improve registration
        # Apply a 3D median filter to smooth out the fluorescence signal in the neuropil
        processed = median_filter(img_data[key], size=3)
        # Apply max projection
        processed = np.max(processed, axis=0)
        # Min-max normalize the image
        processed = cv2.normalize(processed, processed, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Apply a low-pass filter to remove cell body fluorescence
        processed = cv2.medianBlur(processed, 3)
        # Crop the image to remove cranial window boundry and image edges
        if crop_size:
            c = ((np.array(processed.shape) - crop_size) / 2).astype(int)
            processed = processed[c[0]:(c[0] + crop_size), c[1]:(c[1] + crop_size)]

        # Apply a binary threshold to the image
        if apply_threshold:
            # _, img_norm = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if apply_threshold is True:
                apply_threshold = 11
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, apply_threshold, 2)

        img_data[key + '_processed'] = processed
    
    # Extract and match ORB features from both images
    # https://stackoverflow.com/questions/62280342/image-alignment-with-orb-and-ransac-in-scikit-image
    descriptor_extractor = ORB(n_keypoints=400, harris_k=0.0005)  # TODO tune these parameters; make kwargs
    descriptor_extractor.detect_and_extract(img_data['target_stack_processed'])
    descriptors_target, keypoints_target = descriptor_extractor.descriptors, descriptor_extractor.keypoints
    descriptor_extractor.detect_and_extract(img_data['stack_processed'])
    descriptors, keypoints = descriptor_extractor.descriptors, descriptor_extractor.keypoints

    # Match features in both images
    matches = match_descriptors(descriptors_target, descriptors, cross_check=True)

    # Filter keypoints to remove non-matching
    matches_target, matches = keypoints_target[matches[:, 0]], keypoints[matches[:, 1]]

    # Robustly estimate transform model with RANSAC
    # github.com/scikit-image/scikit-image/issues/1749
    matches_flipped = (np.flip(matches_target, axis=-1), np.flip(matches, axis=-1))
    transform_robust, inliers = ransac(matches_flipped, skimage.transform.EuclideanTransform,
                                       min_samples=5, residual_threshold=0.5, max_trials=1000)

    # Invert the translation
    transform_robust = (skimage.transform.EuclideanTransform(rotation=transform_robust.rotation) +
                        skimage.transform.EuclideanTransform(translation=-np.flip(transform_robust.translation)))

    # Apply transformation to image
    aligned = skimage.transform.warp(
        img_data['stack_processed'], transform_robust.inverse, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
    
    # Make plots
    if display:
        # Plot the keypoint matches as lines between the reference and aligned images
        f, ax = plt.subplots()
        ax.imshow(np.hstack((img_data['target_stack_processed'], img_data['stack_processed'])), cmap = 'gray')
        for i in range(matches_target.shape[0]):
            # Plot with 80% opacity
            ax.plot(
                [matches_target[i, 1], matches[i, 1] + img_data['target_stack_processed'].shape[1]],
                [matches_target[i, 0], matches[i, 0]], 'r-o', alpha = 0.4, mfc='none')
        ax.set_title('Keypoint Matches')
        ax.axis('off')

        # Display the images and their differences side by side
        # TODO Plot before and after processing
        f, axs = plt.subplots(2, 3)

        axs[0][0].imshow(img)
        axs[0][0].set_title('Unaligned')
        axs[0][1].imshow(aligned)
        axs[0][1].set_title(f'Aligned')
        axs[0][2].imshow(reference)
        axs[0][2].set_title('Reference')

        unaligned_diff_raw = reference - img
        aligned_diff_raw = reference - aligned
        maxmax = max(np.max(unaligned_diff_raw), np.max(aligned_diff_raw))
        minmin = min(np.min(unaligned_diff_raw), np.min(aligned_diff_raw))
        unaligned_diff = (((unaligned_diff_raw - minmin)/(maxmax - minmin)) * 255).astype(np.uint8)
        aligned_diff = (((aligned_diff_raw - minmin)/(maxmax - minmin)) * 255).astype(np.uint8)
        delta_diff = cv2.normalize(unaligned_diff - aligned_diff, unaligned_diff - aligned_diff, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        axs[1][0].imshow(unaligned_diff)
        axs[1][0].set_title('Unaligned diff')
        axs[1][1].imshow(aligned_diff)
        axs[1][1].set_title('Aligned diff')    
        axs[1][2].imshow(delta_diff)
        axs[1][2].set_title('Delta diff')

        # Remove all axes ticks and labels
        for ax in axs.flatten():
            ax.axis('off')

        # New animated figure with three subplots,
        # the first switching between aligned and unaligned,
        # the second switching between reference and aligned,
        # the third switching between the difference images
        fig, axs = plt.subplots(2, 2)
        
        unaligned_plot = axs[0][0].imshow(reference)
        axs[0][0].set_title('Unaligned')
        aligned_plot = axs[0][1].imshow(reference)
        axs[0][1].set_title('Aligned')
        trans_plot = axs[1][0].imshow(img)
        axs[1][0].set_title('Transform')
        diff_plot = axs[1][1].imshow(unaligned_diff, cmap='grey')
        axs[1][1].set_title('Difference')
        
        # Add large asterisk in top left corner to indicate which image is currently displayed
        txt = [axs[0][i].text(0.05, 0.95, '*', color='red', fontsize=20,
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
                unaligned_plot.set_data(img)
                aligned_plot.set_data(aligned)
                trans_plot.set_data(aligned)
                diff_plot.set_data(aligned_diff)
                for t in txt:
                    t.set_text('')
            else:
                unaligned_plot.set_data(reference)
                aligned_plot.set_data(reference)
                trans_plot.set_data(img)
                diff_plot.set_data(unaligned_diff)
                for t in txt:
                    t.set_text('*')
            return aligned_plot, unaligned_plot, diff_plot

        ani = animation.FuncAnimation(fig, update, frames=10, interval=1, blit=False)
        # Save animated figure as a GIF
        from pathlib import Path
        ani.save(Path.home().joinpath('Pictures', 'aligned.gif'), writer='pillow', fps=2)
        plt.show()

    return aligned,  {'rotation': transform_robust.rotation, 'translation': transform_robust.translation}


def get_brain_surface_plane_from_ref_points(ref_surface_points: dict, ref_img_meta: dict):
    """Get the brain surface plane from reference surface points.

    From the reference points, calculate a plane that approximates the brain surface and it's
    normal. Additionally, returns the average depth of the three points that is later used to
    adjust the apparent depth of a cell.

    Parameters
    ----------
    ref_surface_points : dict
        A dictionary containing the reference surface points. TODO add more info
    ref_img_meta : dict
        Metadata for the reference image.

    Returns
    -------
    p_ref : np.ndarray
        The point on the plane.
    n_ref : np.ndarray
        The normal vector of the plane.
    dv_avg : float
        The average depth value of the surface points.

    """

    stack_ixs = [point['stack_idx'] for point in ref_surface_points['points']]
    stack_dv = np.array(ref_img_meta['scanImageParams']['hStackManager']['zs'])[
        stack_ixs
    ]
    dv_avg = np.average(stack_dv)
    ref_points_rel = np.array(
        [point['coords'] for point in ref_surface_points['points']]
    )
    ref_points_mlap = cs2d.transform(ref_points_rel, 'image', 'mlap')
    ref_points_ = np.concatenate(
        [ref_points_mlap, stack_dv[:, np.newaxis] - dv_avg], axis=1
    )
    p_ref, n_ref = plane_normal_form(ref_points_)
    # invert if pointing downards
    if n_ref[2] < 0:
        n_ref *= -1
    return p_ref, n_ref, dv_avg


# Step 1
# If there is a histology image, load that as our reference stack MLAP coordinates

# If the image is from a different session, calculate the transformation between the two
# and apply that to the reference stack

# Re-calculate center MM coordinates by using the MLAP coordinates of the centre pixel,
# then add the offset? TODO Confirm whether we need to use offset. I guess it's cleaner to
# use the craniotomy centre for computing the normal vector

# If there is no histology image, compute the estimate coordinates from the convex hull
# of the atlas and the craniotomy coordinates

# Step 2
# Used 


# p_ref, n_ref, dv_avg = get_brain_surface_plane_from_ref_points(
#     ref_surface_points, ref_img_meta
# )

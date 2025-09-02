"""Registration of MPCI fields of view (FOVs) to a common reference frame.

This module provides tools for registering widefield reference stack images to one another, and
for extracting atlas coordinates from FOV mean images.
"""
import json
import time
import enum
import uuid
import logging
import pickle
import tempfile, shutil
from collections import Counter
from pathlib import Path

from one.alf.path import ALFPath, filename_parts
from one.alf.spec import to_alf
import one.alf.io as alfio

from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM, MRITorontoAtlas, AllenAtlas
from iblutil.util import Bunch
from ScanImageTiffReader import ScanImageTiffReader  # NB: use skimage if possible
import tifffile
from tqdm import tqdm
import skimage.transform
from skimage.feature.orb import ORB
from skimage.feature import match_descriptors
from skimage.measure import ransac
from scipy.interpolate import interpn, NearestNDInterpolator
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numba as nb
import cv2

from ibllib.mpci.brain_meshes import get_plane_at_point_mlap, get_surface_points
from ibllib.mpci.linalg import intersect_line_plane, surface_normal, find_triangle, _update_points
from ibllib.pipes.base_tasks import MesoscopeTask
from ibllib.io.extractors import mesoscope
import ibllib.oneibl.data_handlers as dh


_logger = logging.getLogger(__name__)
Provenance = enum.Enum('Provenance', ['ESTIMATE', 'FUNCTIONAL', 'LANDMARK', 'HISTOLOGY'])  # py3.11 make StrEnum


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
    TODO outputs
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
    descriptor_extractor = ORB(n_keypoints=400, harris_k=0.0005, fast_n=13)  # TODO tune these parameters; make kwargs
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
    assert not np.isnan(aligned).all(), 'Alignment resulted in NaN values'
    params = {'rotation': transform_robust.rotation, 'translation': transform_robust.translation}

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

        img = np.max(img_data['stack'], axis=0)
        reference = np.max(img_data['target_stack'], axis=0)
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
        ani.save(Path.home().joinpath('Pictures', 'aligned.gif'), writer='pillow', fps=2)
        plt.show()

    return aligned,  params


class MesoscopeFOV(MesoscopeTask):
    """Create FOV and FOV location objects in Alyx from metadata."""

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            ('mpciROIs.stackPos.npy', 'alf/FOV*', True)],
            'output_files': [('mpciMeanImage.brainLocationIds*.npy', 'alf/FOV_*', True),
                             ('mpciMeanImage.mlapdv*.npy', 'alf/FOV_*', True),
                             ('mpciROIs.mlapdv*.npy', 'alf/FOV_*', True),
                             ('mpciROIs.brainLocationIds*.npy', 'alf/FOV_*', True),
                             ('_ibl_rawImagingData.meta.json', self.device_collection, True)]
        }
        return signature

    def _run(self, *args, provenance=Provenance.ESTIMATE, **kwargs):
        """
        Register fields of view (FOV) to Alyx and extract the coordinates and IDs of each ROI.

        Steps:
            1. Save the mpciMeanImage.brainLocationIds_estimate and mlapdv datasets.
            2. Use mean image coordinates and ROI stack position datasets to extract brain location
             of each ROI.
            3. Register the location of each FOV in Alyx.

        Parameters
        ----------
        provenance : Provenance
            The provenance of the coordinates in the meta file. For all but 'HISTOLOGY', the
            provenance is added as a dataset suffix.  Defaults to ESTIMATE.

        Returns
        -------
        dict
            The newly created FOV Alyx record.
        list
            The newly created FOV location Alyx records.

        Notes
        -----
        - Once the FOVs have been registered they cannot be updated with this task. Rerunning this
          task will result in an error.
        - This task modifies the first meta JSON file.  All meta files are registered by this task.
        """
        # Load necessary data
        (filename, collection, _), *_ = self.signature['input_files']
        meta_files = sorted(self.session_path.glob(f'{collection}/{filename}'))
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))

        suffix = None if provenance is Provenance.HISTOLOGY else provenance.name.lower()
        _logger.info('Extracting %s MLAPDV datasets', suffix or 'final')

        # Extract mean image MLAPDV coordinates and brain location IDs
        mean_image_mlapdv, mean_image_ids = self.project_mlapdv(meta)

        # Save the meta data file with new coordinate fields
        with open(meta_files[0], 'w') as fp:
            json.dump(meta, fp)

        # Save the mean image datasets
        mean_image_files = []
        assert set(mean_image_mlapdv.keys()) == set(mean_image_ids.keys()) and len(mean_image_ids) == nFOV
        for i in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{i:02}')
            alf_path.mkdir(parents=True, exist_ok=True)
            for attr, arr, sfx in (('mlapdv', mean_image_mlapdv[i], suffix),
                                   ('brainLocationIds', mean_image_ids[i], ('ccf', '2017', suffix))):
                mean_image_files.append(alf_path / to_alf('mpciMeanImage', attr, 'npy', timescale=sfx))
                np.save(mean_image_files[-1], arr)

        # Extract ROI MLAPDV coordinates and brain location IDs
        roi_mlapdv, roi_brain_ids = self.roi_mlapdv(nFOV, suffix=suffix)

        # Write MLAPDV + brain location ID of ROIs to disk
        roi_files = []
        assert set(roi_mlapdv.keys()) == set(roi_brain_ids.keys()) and len(roi_mlapdv) == nFOV
        for i in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{i:02}')
            for attr, arr, sfx in (('mlapdv', roi_mlapdv[i], suffix),
                                   ('brainLocationIds', roi_brain_ids[i], ('ccf', '2017', suffix))):
                roi_files.append(alf_path / to_alf('mpciROIs', attr, 'npy', timescale=sfx))
                np.save(roi_files[-1], arr)

        # Register FOVs in Alyx
        self.register_fov(meta, suffix)

        return sorted([*meta_files, *roi_files, *mean_image_files])

    def update_surgery_json(self, meta, normal_vector):
        """
        Update surgery JSON with surface normal vector.

        Adds the key 'surface_normal_unit_vector' to the most recent surgery JSON, containing the
        provided three element vector.  The recorded craniotomy center must match the coordinates
        in the provided meta file.

        Parameters
        ----------
        meta : dict
            The imaging meta data file containing the 'centerMM' key.
        normal_vector : array_like
            A three element unit vector normal to the surface of the craniotomy center.

        Returns
        -------
        dict
            The updated surgery record, or None if no surgeries found.
        """
        if not self.one or self.one.offline:
            _logger.warning('failed to update surgery JSON: ONE offline')
            return
        # Update subject JSON with unit normal vector of craniotomy centre (used in histology)
        subject = self.one.path2ref(self.session_path, parse=False)['subject']
        surgeries = self.one.alyx.rest('surgeries', 'list', subject=subject, procedure='craniotomy')
        if not surgeries:
            _logger.error(f'Surgery not found for subject "{subject}"')
            return
        surgery = surgeries[0]  # Check most recent surgery in list
        center = (meta['centerMM']['ML'], meta['centerMM']['AP'])
        match = (k for k, v in (surgery['json'] or {}).items() if
                 str(k).startswith('craniotomy') and np.allclose(v['center'], center))
        if (key := next(match, None)) is None:
            _logger.error('Failed to update surgery JSON: no matching craniotomy found')
            return surgery
        data = {key: {**surgery['json'][key], 'surface_normal_unit_vector': tuple(normal_vector)}}
        surgery['json'] = self.one.alyx.json_field_update('subjects', subject, data=data)
        return surgery

    def roi_mlapdv(self, nFOV: int, suffix=None):
        """
        Extract ROI MLAPDV coordinates and brain location IDs.

        MLAPDV coordinates are in μm relative to bregma.  Location IDs are from the 2017 Allen
        common coordinate framework atlas.

        Parameters
        ----------
        nFOV : int
            The number of fields of view acquired.
        suffix : {None, 'estimate'}
            The attribute suffix of the mpciMeanImage datasets to load. If generating from
            estimates, the suffix should be 'estimate'.

        Returns
        -------
        dict of int : numpy.array
            A map of field of view to ROI MLAPDV coordinates.
        dict of int : numpy.array
            A map of field of view to ROI brain location IDs.
        """
        all_mlapdv = {}
        all_brain_ids = {}
        for n in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{n:02}')

            # Load neuron centroids in pixel space
            stack_pos_file = next(alf_path.glob('mpciROIs.stackPos*'), None)
            if not stack_pos_file:
                raise FileNotFoundError(alf_path / 'mpci.stackPos*')
            stack_pos = alfio.load_file_content(stack_pos_file)

            # Load MLAPDV + brain location ID maps of pixels
            mpciMeanImage = alfio.load_object(
                alf_path, 'mpciMeanImage', attribute=['mlapdv', 'brainLocationIds'])

            # Get centroid MLAPDV + brainID by indexing pixel-map with centroid locations
            mlapdv = np.full(stack_pos.shape, np.nan)
            brain_ids = np.full(stack_pos.shape[0], np.nan)
            for i in np.arange(stack_pos.shape[0]):
                idx = (stack_pos[i, 0], stack_pos[i, 1])
                sfx = f'_{suffix}' if suffix else ''
                mlapdv[i, :] = mpciMeanImage['mlapdv' + sfx][idx]
                brain_ids[i] = mpciMeanImage['brainLocationIds_ccf_2017' + sfx][idx]
            assert ~np.isnan(brain_ids).any()
            all_brain_ids[n] = brain_ids.astype(int)
            all_mlapdv[n] = mlapdv

        return all_mlapdv, all_brain_ids

    @staticmethod
    def get_provenance(filename):
        """
        Get the field of view provenance from a mpciMeanImage or mpciROIs dataset.

        Parameters
        ----------
        filename : str, pathlib.Path
            A filename to get the provenance from.

        Returns
        -------
        Provenance
            The provenance of the file.
        """
        filename = ALFPath(filename).name
        timescale = (filename_parts(filename)[3] or '').split('_')
        provenances = [i.name.lower() for i in Provenance]
        provenance = (Provenance[x.upper()] for x in timescale if x in provenances)
        return next(provenance, None) or Provenance.HISTOLOGY

    def register_fov(self, meta: dict, suffix: str = None) -> (list, list):
        """
        Create FOV on Alyx.

        Assumes field of view recorded perpendicular to objective.
        Assumes field of view is plane (negligible volume).

        Required Alyx fixtures:
            - experiments.ImagingType(name='mesoscope')
            - experiments.CoordinateSystem(name='IBL-Allen')

        Parameters
        ----------
        meta : dict
            The raw imaging meta data from _ibl_rawImagingData.meta.json.
        suffix : str
            The file attribute suffixes to load from the mpciMeanImage object. Either 'estimate' or
            None. No suffix means the FOV location provenance will be L (Landmark).

        Returns
        -------
        list of dict
            A list registered of field of view entries from Alyx.

        TODO Determine dual plane ID for JSON field
        """
        dry = self.one is None or self.one.offline
        alyx_fovs = []
        # Count the number of slices per stack ID: only register stacks that contain more than one slice.
        slice_counts = Counter(f['roiUUID'] for f in meta.get('FOV', []))
        # Create a new stack in Alyx for all stacks containing more than one slice.
        # Map of ScanImage ROI UUID to Alyx ImageStack UUID.
        if dry:
            stack_ids = {i: uuid.uuid4() for i in slice_counts if slice_counts[i] > 1}
        else:
            stack_ids = {i: self.one.alyx.rest('imaging-stack', 'create', data={'name': i})['id']
                         for i in slice_counts if slice_counts[i] > 1}

        for i, fov in enumerate(meta.get('FOV', [])):
            assert set(fov.keys()) >= {'MLAPDV', 'nXnYnZ', 'roiUUID'}
            # Field of view
            alyx_FOV = {
                'session': self.session_path.as_posix() if dry else str(self.path2eid()),
                'imaging_type': 'mesoscope', 'name': f'FOV_{i:02}',
                'stack': stack_ids.get(fov['roiUUID'])
            }
            if dry:
                print(alyx_FOV)
                alyx_FOV['location'] = []
                alyx_fovs.append(alyx_FOV)
            else:
                alyx_fovs.append(self.one.alyx.rest('fields-of-view', 'create', data=alyx_FOV))

            # Field of view location
            data = {'field_of_view': alyx_fovs[-1].get('id'),
                    'default_provenance': True,
                    'coordinate_system': 'IBL-Allen',
                    'n_xyz': fov['nXnYnZ']}
            if suffix:
                data['provenance'] = suffix[0].upper()

            # Convert coordinates to 4 x 3 array (n corners by n dimensions)
            # x1 = top left ml, y1 = top left ap, y2 = top right ap, etc.
            coords = [fov['MLAPDV'][key] for key in ('topLeft', 'topRight', 'bottomLeft', 'bottomRight')]
            coords = np.vstack(coords).T
            data.update({k: arr.tolist() for k, arr in zip('xyz', coords)})

            # Load MLAPDV + brain location ID maps of pixels
            filename = 'mpciMeanImage.brainLocationIds_ccf_2017' + (f'_{suffix}' if suffix else '') + '.npy'
            filepath = self.session_path.joinpath('alf', f'FOV_{i:02}', filename)
            mean_image_ids = alfio.load_file_content(filepath)

            data['brain_region'] = np.unique(mean_image_ids).astype(int).tolist()

            if dry:
                print(data)
                alyx_FOV['location'].append(data)
            else:
                alyx_fovs[-1]['location'].append(self.one.alyx.rest('fov-location', 'create', data=data))
        return alyx_fovs

    def load_triangulation(self):
        """
        Load the surface triangulation file.

        A triangle mesh of the smoothed convex hull of the dorsal surface of the mouse brain,
        generated from the 2017 Allen 10um annotation volume. This triangulation was generated in
        MATLAB.

        Returns
        -------
        points : numpy.array
            An N by 3 float array of x-y vertices, defining all points of the triangle mesh. These
            are in μm relative to the IBL bregma coordinates.
        connectivity_list : numpy.array
            An N by 3 integer array of vertex indices defining all points that form a triangle.
        """
        fixture_path = Path(mesoscope.__file__).parent.joinpath('mesoscope')
        surface_triangulation = np.load(fixture_path / 'surface_triangulation.npz')
        points = surface_triangulation['points'].astype('f8')
        connectivity_list = surface_triangulation['connectivity_list']
        surface_triangulation.close()
        return points, connectivity_list

    def project_mlapdv(self, meta, atlas=None):
        """
        Calculate the mean image pixel locations in MLAPDV coordinates and determine the brain
        location IDs.

        MLAPDV coordinates are in μm relative to bregma.  Location IDs are from the 2017 Allen
        common coordinate framework atlas.

        Parameters
        ----------
        meta : dict
            The raw imaging data meta file, containing coordinates for the centre of each field of
            view.
        atlas : ibllib.atlas.Atlas
            An atlas instance.

        Returns
        -------
        dict
            A map of FOV number (int) to mean image MLAPDV coordinates as a 2D numpy array.
        dict
            A map of FOV number (int) to mean image brain location IDs as a 2D numpy int array.
        """
        mlapdv = {}
        location_id = {}
        # Use the MRI atlas as this applies scaling, particularly along the DV axis to (hopefully)
        # more accurately represent the living brain.
        atlas = atlas or MRITorontoAtlas(res_um=10)
        # The centre of the craniotomy / imaging window
        coord_ml = meta['centerMM']['ML'] * 1e3  # mm -> μm
        coord_ap = meta['centerMM']['AP'] * 1e3  # mm -> μm
        pt = np.array([coord_ml, coord_ap])

        points, connectivity_list = self.load_triangulation()
        # Only keep faces that have normals pointing up (positive DV value).
        # Calculate the normal vector pointing out of the convex hull.
        triangles = points[connectivity_list, :]
        normals = surface_normal(triangles)
        up_faces, = np.where(normals[:, -1] > 0)
        # only keep triangles that have normal vector with positive DV component
        dorsal_connectivity_list = connectivity_list[up_faces, :]
        # Flatten triangulation by dropping the dorsal coordinates and find the location of the
        # window center (we convert mm -> μm here)
        face_ind = find_triangle(pt * 1e-3, points[:, :2] * 1e-3, dorsal_connectivity_list.astype(np.intp))
        assert face_ind != -1

        # find the coordDV that sits on the triangular face and had [coordML, coordAP] coordinates;
        # the three vertices defining the triangle
        face_vertices = points[dorsal_connectivity_list[face_ind, :], :]

        # Get the surface normal unit vector of dorsal triangle
        normal_vector = surface_normal(face_vertices)

        # Update the surgery JSON field with normal unit vector, for use in histology alignment
        self.update_surgery_json(meta, normal_vector)

        # all the vertices should be on the plane ax + by + cz = 1, so we can find
        # the abc coefficients by inverting the three equations for the three vertices
        abc, *_ = np.linalg.lstsq(face_vertices, np.ones(3), rcond=None)

        # and then find a point on that plane that corresponds to a given x-y
        # coordinate (which is ML-AP coordinate)
        coord_dv = (1 - pt @ abc[:2]) / abc[2]

        # We should not use the actual surface of the brain for this, as it might be in one of the sulci
        # DO NOT USE THIS:
        # coordDV = interp2(axisMLmm, axisAPmm, surfaceDV, coordML, coordAP)

        # Now we need to span the plane of the coverslip with two orthogonal unit vectors.
        # We start with vY, because the order is important and we usually have less
        # tilt along AP (pitch), which will cause less deviation in vX from pure ML.
        vY = np.array([0, normal_vector[2], -normal_vector[1]])  # orthogonal to the normal of the plane
        vX = np.cross(vY, normal_vector)  # orthogonal to n and to vY
        # normalize and flip the sign if necessary
        vX = vX / np.sqrt(vX @ vX) * np.sign(vX[0])  # np.sqrt(vY @ vY) == LR norm of vX
        vY = vY / np.sqrt(vY @ vY) * np.sign(vY[1])

        # what are the dimensions of the data arrays (ap, ml, dv)
        (nAP, nML, nDV) = atlas.image.shape
        # Let's shift the coordinates relative to bregma
        voxel_size = atlas.res_um  # [μm] resolution of the atlas
        bregma_coords = ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / voxel_size  # (ml, ap, dv)
        axis_ml_um = (np.arange(nML) - bregma_coords[0]) * voxel_size
        axis_ap_um = (np.arange(nAP) - bregma_coords[1]) * voxel_size * -1.
        axis_dv_um = (np.arange(nDV) - bregma_coords[2]) * voxel_size * -1.

        # projection of FOVs on the brain surface to get ML-AP-DV coordinates
        _logger.info('Projecting in 3D')
        for i, fov in enumerate(meta['FOV']):  # i, fov = next(enumerate(meta['FOV']))
            start_time = time.time()
            _logger.info(f'FOV {i + 1}/{len(meta["FOV"])}')
            y_px_idx, x_px_idx = np.mgrid[0:fov['nXnYnZ'][0], 0:fov['nXnYnZ'][1]]

            # xx and yy are in mm in coverslip space
            points = ((0, fov['nXnYnZ'][0] - 1), (0, fov['nXnYnZ'][1] - 1))
            # The four corners of the FOV, determined by taking the center of the craniotomy in MM,
            # the x-y coordinates of the imaging window center (from the tiled reference image) in
            # galvanometer units, and the x-y coordinates of the FOV center in galvanometer units.
            values = [[fov['MM']['topLeft'][0], fov['MM']['topRight'][0]],
                      [fov['MM']['bottomLeft'][0], fov['MM']['bottomRight'][0]]]
            values = np.array(values) * 1e3  # mm -> μm
            xx = interpn(points, values, (y_px_idx, x_px_idx))

            values = [[fov['MM']['topLeft'][1], fov['MM']['topRight'][1]],
                      [fov['MM']['bottomLeft'][1], fov['MM']['bottomRight'][1]]]
            values = np.array(values) * 1e3  # mm -> μm
            yy = interpn(points, values, (y_px_idx, x_px_idx))

            xx = xx.flatten() - coord_ml
            yy = yy.flatten() - coord_ap

            # rotate xx and yy in 3D
            # the coords are still on the coverslip, but now have 3D values
            coords = np.outer(xx, vX) + np.outer(yy, vY)  # (vX * xx) + (vY * yy)
            coords = coords + [coord_ml, coord_ap, coord_dv]

            # for each point of the FOV create a line parametrization (trajectory normal to the coverslip plane).
            # start just above the coverslip and go 3 mm down, should be enough to 'meet' the brain
            t = np.arange(-voxel_size, 3e3, voxel_size)

            # Find the MLAPDV atlas coordinate and brain location of each pixel.
            MLAPDV, annotation = _update_points(
                t, normal_vector, coords, axis_ml_um, axis_ap_um, axis_dv_um, atlas.label)
            annotation = atlas.regions.index2id(annotation)  # convert annotation indices to IDs

            if np.any(np.isnan(MLAPDV)):
                _logger.warning('Areas of FOV lie outside the brain')
            _logger.info(f'done ({time.time() - start_time:3.1f} seconds)\n')
            MLAPDV = np.reshape(MLAPDV, [*x_px_idx.shape, 3])
            annotation = np.reshape(annotation, x_px_idx.shape)

            fov['MLAPDV'] = {
                'topLeft': MLAPDV[0, 0, :].tolist(),
                'topRight': MLAPDV[0, -1, :].tolist(),
                'bottomLeft': MLAPDV[-1, 0, :].tolist(),
                'bottomRight': MLAPDV[-1, -1, :].tolist(),
                'center': MLAPDV[round(x_px_idx.shape[0] / 2) - 1, round(x_px_idx.shape[1] / 2) - 1, :].tolist()
            }

            # Save the brain regions of the corners/centers of FOV (annotation field)
            fov['brainLocationIds'] = {
                'topLeft': int(annotation[0, 0]),
                'topRight': int(annotation[0, -1]),
                'bottomLeft': int(annotation[-1, 0]),
                'bottomRight': int(annotation[-1, -1]),
                'center': int(annotation[round(x_px_idx.shape[0] / 2) - 1, round(x_px_idx.shape[1] / 2) - 1])
            }

            mlapdv[i] = MLAPDV
            location_id[i] = annotation
        return mlapdv, location_id


class Reprojection(MesoscopeFOV):
    """Reprojection of mesoscope FOVs into Allen atlas space.
    
    After imaging, the brain undergoes histology whereby the tissue is sectioned and imaged, then registered to the Allen atlas producing a file containing the Allen atlas volume coordinates for each pixel of the full-field image of the craniotomy.
    With this file we have the 3-D MLAPDV coordinates of the brain surface in meters of the full-field image. This registration step involves some warping, therefore the distance between two pixels may not be uniform in Allen coordinate space.
    The full-field ('reference') image contains the location of each zoomed-in field of view (FOV) in the craniotomy.
    The reference image and FOVs microscope coordinates are in the same space with μm coordinates. We can use this reference space to transform the FOVs into the Allen atlas space using nearest neighbor interpolation.
    This is done in the 'interpolate_FOVs' method. After this step we have the 3-D coordinates for each pixel of the FOVs in the Allen atlas space.
    These are the coordinates of the brain surface from histology, however the imaging plane is not perfectly parallel with the brain surface, and the FOVs are imaged beneath the brain's surface.
    To account for this, we use three points chosen from the reference image where the surface is in focus at different imaging depths. From these three points we can construct a plane to determine the difference between the optical axis and the brain surface.
    Using the difference in optical depth between this plane and the FOV we can adjust the FOV coordinates accordingly. This is done in the 'get_mlapdv_rel' method
    After we have the adjusted MLAP coordinates (without depth), we can use them to obtain the final MLAPDV coordinates in the Allen atlas space.
    This is done in the 'mlapdv_from_rel' method by computing a triangulation of the brain's surface from the Allen atlas volume, then finding the surface location for each FOV pixel, then applying the adjusted depth.
    """

    @property
    def signature(self):
        I = dh.ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            I('mpciROIs.stackPos.npy', 'alf/FOV*', True),
                            # New additions
                            I('referenceImage.stack.tif', 'raw_imaging_data_??/reference', False, unique=True),
                            I('referenceImage.meta.json', 'raw_imaging_data_??/reference', False, unique=True),
                            I('referenceImage.points.json', 'raw_imaging_data_??/reference', False, unique=True),
                            # ('registered_mlapdv.npy', 'histology', False)  # may be in another session folder!
                            ],
            'output_files': [('mpciMeanImage.brainLocationIds*.npy', 'alf/FOV_*', True),
                             ('mpciMeanImage.mlapdv*.npy', 'alf/FOV_*', True),
                             ('mpciROIs.mlapdv*.npy', 'alf/FOV_*', True),
                             ('mpciROIs.brainLocationIds*.npy', 'alf/FOV_*', True),
                             ('_ibl_rawImagingData.meta.json', self.device_collection, True)]
        }
        return signature

    def _run(self, *args, provenance=Provenance.HISTOLOGY, atlas_resolution=25, display=False):
        self.atlas = AllenAtlas(res_um=atlas_resolution)  # TODO Change to MRI
        # Load the reference stack & (down)load the registered MLAPDV coordinates
        reference_image = self._load_reference_stack()
        # Load main meta
        _, meta_files, _ = self.input_files[0].find_files(self.session_path)
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))
        # Update the craniotomy center
        self.update_craniotomy_center(reference_image)
        # Interpolate the FOVs to the reference stack
        mlapdv = self.interpolate_FOVs(reference_image, meta)

        # Account for optical plane tilt
        mlapdv_rel = self.correct_fov_depth_and_surface_projection(mlapdv, meta, reference_image)
        mean_image_mlapdv = self.project_mlapdv_from_surface(mlapdv_rel)

        # Look up brain location IDs from coordinates
        mean_image_ids = []
        for xyz in mean_image_mlapdv:
            labels = self.atlas.get_labels(xyz.reshape(-1, 3))
            mean_image_ids.append(labels.reshape(xyz.shape[:2]))

        # Save the mean image datasets
        suffix = None if provenance is Provenance.HISTOLOGY else provenance.name.lower()
        mean_image_files = []
        assert len(mean_image_mlapdv) == nFOV
        for i in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{i:02}')
            alf_path.mkdir(parents=True, exist_ok=True)
            for attr, arr, sfx in (('mlapdv', mean_image_mlapdv[i], suffix),
                                   ('brainLocationIds', mean_image_ids[i], ('ccf', '2017', suffix))):
                mean_image_files.append(alf_path / to_alf('mpciMeanImage', attr, 'npy', timescale=sfx))
                np.save(mean_image_files[-1], arr)

        # Extract ROI MLAPDV coordinates and brain location IDs
        roi_mlapdv, roi_brain_ids = self.roi_mlapdv(nFOV, suffix=suffix)

        # Write MLAPDV + brain location ID of ROIs to disk
        roi_files = []
        assert set(roi_mlapdv.keys()) == set(roi_brain_ids.keys()) and len(roi_mlapdv) == nFOV
        for i in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{i:02}')
            for attr, arr, sfx in (('mlapdv', roi_mlapdv[i], suffix),
                                   ('brainLocationIds', roi_brain_ids[i], ('ccf', '2017', suffix))):
                roi_files.append(alf_path / to_alf('mpciROIs', attr, 'npy', timescale=sfx))
                np.save(roi_files[-1], arr)

        if display:
            from ibllib.mpci.plotters import plot_brain_surface_points
            axes = plt.figure(figsize=[10, 10]).add_subplot(projection='3d')
            brain_surface_points = get_surface_points(self.atlas)
            axes = plot_brain_surface_points(brain_surface_points, ds=4, axes=axes)
            # Plot ROIs
            for i, fov in enumerate(roi_mlapdv.values()):
                axes.scatter(*fov.T, ".", c="k", s=1, alpha=0.05, label='ROIs in imaging plane')

        # Register FOVs in Alyx
        # self.register_fov(meta, suffix)

        return sorted([*meta_files, *roi_files, *mean_image_files])

    def get_atlas_registered_reference_mlap(self, reference_session_path, clobber=False, client_name='server'):
        """Download the aligned reference stack Allen atlas indices.

        This is the file created by the histology pipeline, one per subject.
        This file contains the Allen atlas image volume indices for each pixel of the reference stack.

        Returns
        -------
        pathlib.Path
            The local filepath of the aligned reference stack.
            A uint16 array with shape (h, w, 3), comprising Allen atlas image volume indices for
            dimensions representing (ml, ap, dv).  The first two dimensions (h, w) should equal
            those of the reference stack.
        """
        assert reference_session_path.subject == self.session_path.subject
        assert self.one, 'ONE required'
        local_file = self.session_path.parents[2] / reference_session_path.relative_to_lab() / 'histology' / 'referenceImage.mlapdv.npy'
        if clobber or not local_file.exists():
            # Download remote file
            assert self.one, 'ONE required'
            local_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                # assert isinstance(self.data_handler, dh.ServerGlobusDataHandler)  # If not, assume Globus not configured
                handler = dh.ServerGlobusDataHandler(reference_session_path, {'input_files': [], 'output_files': []}, one=self.one)
                endpoint_id = next(v['id'] for k, v in handler.globus.endpoints.items() if k.startwith('flatiron'))
                handler.globus.add_endpoint(endpoint_id, label='flatiron_histology', root_path='/histology/')
                remote_file = f'{reference_session_path.lab}/{reference_session_path.session_path_short()}/{local_file.name}'
                handler.globus.mv('flatiron_histology', 'local', [remote_file], ['/'.join(local_file.parts[-5:])])
                assert local_file.exists(), f'Failed to download {remote_file} to {local_file}'
            except Exception as e:
                _logger.error(f'Failed to download via Globus: {e}')
                with tempfile.TemporaryDirectory() as tmpdir:
                    remote_file = f'{self.one.alyx._par.HTTP_DATA_SERVER}/histology/{reference_session_path.lab}/{reference_session_path.subject}/referenceImage.mlapdv.npy'
                    _logger.warning(f'Using HTTP download for {remote_file}')
                    file = self.one.alyx.download_file(remote_file, target_dir=tmpdir)
                    shutil.move(file, local_file)
        return local_file

    def load_metadata_from_tif(self):
        # load meta from tif
        # from ScanImageTiffReader import ScanImageTiffReader
        # tif = ScanImageTiffReader(str(next(self.session_path.glob('raw_imaging_data_??/reference/referenceImage.stack.tif'))))
        # meta = tif.metadata()  # fails - is empty str
        raise NotImplementedError

    @staticmethod
    def get_window_center(meta):
        """Get the window offset from image center in mm.

        Previously this was not extracted in the reference stack metadata, but can now be found in the centerMM.x and centerMM.y fields.

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
            param = next(x.split('=')[-1].strip() for x in meta['rawScanImageMeta']['Software'].split('\n') if x.startswith('SI.hDisplay.circleOffset'))
            return np.fromiter(map(float, param[1:-1].split()), dtype=float) / 1e3  # μm -> mm
        except StopIteration:
            return np.array([0, 0], dtype=float)

    def get_reference_image_extent(self, ref_meta):
        """Get the reference image extent along the imaging plane in mm from the window center.

        Parameters
        ----------
        ref_meta : dict
            The reference stack metadata.

        Returns
        -------
        numpy.array
            The reference image extent in mm from the craniotomy center (left, right, top, bottom).
        """
        # Resolution of the objective in mm/degree of the scan angle
        objective_resolution = ref_meta['scanImageParams']['objectiveResolution'] / 1000  # μm -> mm
        center_offset = self.get_window_center(ref_meta)  # (x, y) offset in mm

        # find centers, sizes and nLines of each FOV
        si_rois = ref_meta['rawScanImageMeta']['Artist']['RoiGroups']['imagingRoiGroup']['rois']
        si_rois = list(filter(lambda x: x['enable'], si_rois))
        nFOVs = len(si_rois)
        cXY = np.full((nFOVs, 2), np.nan)
        sXY = np.full((nFOVs, 2), np.nan)
        nLines = np.zeros(nFOVs, dtype=int)

        for i, fov in enumerate(si_rois):
            # Get the center and size of the FOV in ScanImage coordinates
            cXY[i, :] = fov['scanfields']['centerXY']
            sXY[i, :] = fov['scanfields']['sizeXY']
            nLines[i] = fov['scanfields']['pixelResolutionXY'][1]
        cXY += (center_offset / objective_resolution)

        # Find extent.  Scanfields comprise long, vertical rectangles tiled along the x-axis.
        fov_order = np.argsort(cXY[:, 0])  # 0 = left-most, -1 = right-most
        
        # Convert the ScanImage coordinates to pixel coordinates for each FOV
        max_y = np.argmax(sXY[:, 1])  # longest scanfield in y
        left = cXY[fov_order[0], 0] - sXY[fov_order[0], 0] / 2
        right = cXY[fov_order[-1], 0] + sXY[fov_order[-1], 0] / 2
        bottom = cXY[max_y, 1] + sXY[max_y, 1] / 2
        top = cXY[max_y, 1] - sXY[max_y, 1] / 2
        # Scale
        ref_extent = np.array([left, right, top, bottom]) * objective_resolution  # mm
        return ref_extent

    def get_fov_objective_extent(self, meta):
        objective_resolution = meta['scanImageParams']['objectiveResolution'] / 1000  # μm -> mm
        center_offset = self.get_window_center(meta) / objective_resolution
        si_rois = meta['rawScanImageMeta']['Artist']['RoiGroups']['imagingRoiGroup']['rois']
        si_rois = filter(lambda x: x['enable'], si_rois)
        # Sort by ALF FOV number by matching ROI UUID
        si_rois = sorted(si_rois, key=lambda x: next(i for i, y in enumerate(meta['FOV']) if y['roiUUID'] == x['roiUuid']))
        coordinates = []
        for roi in si_rois:
            scanfield = roi['scanfields']
            angle = scanfield['rotationDegrees']
            center_xy = np.array(scanfield['centerXY']) + center_offset  # [x,y] center
            center_mm = center_xy * objective_resolution  # mm
            size_mm = np.array(scanfield['sizeXY']) * objective_resolution  # size
            # [left, top, right, bottom] -> [left, right, top, bottom]
            extent = (np.r_[center_mm, center_mm] + np.r_[-size_mm, size_mm] / 2)[[0, 2, 1, 3]]
            coordinates.append({'extent': extent, 'angle': angle, 'center': center_mm, 'size': size_mm})
        return coordinates

    def plot_FOVs_on_ref_stack(self):
        """Plot the FOVs on the reference image stack.
        
        This method will plot the reference image stack in ScanImage mm from the craniotomy center,
        and overlay the FOV mean images onto it.
        
        Returns
        -------
        numpy.array
            The reference image stack extent in mm from the craniotmoy center (left, right, top, bottom).
        list of dict
            A list of dictionaries containing the FOV coordinates and angles.
            Each dictionary contains:
                - 'extent': The extent of the FOV in mm from the craniotomy center.
                - 'angle': The angle of the FOV in degrees.
        """
        # Load the reference image and all metadata
        stack, ref_meta = self._load_reference_stack()
        meta_files = sorted(self.session_path.glob(self.signature['input_files'][0].glob_pattern))
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        f, ax = plt.subplots()
        import cv2
        stack_max = np.max(stack, axis=0)
        stack_max = cv2.normalize(stack_max, stack_max, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        ref_extent, window_center, objective_resolution = self.get_reference_image_extent(ref_meta)       
        ax.matshow(stack_max, extent=ref_extent, cmap='gray', norm='symlog')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        # FOV rectangles
        assert objective_resolution == meta['scanImageParams']['objectiveResolution'] / 1000
        coordinates = self.get_fov_objective_extent(meta)
        for i, fov in enumerate(coordinates):
            # Draw rectangle by defining bottom left corner, width, height and angle
            rect = plt.Rectangle(fov['extent'][[0, -1]], fov['size'][0], fov['size'][1],
                                 angle=fov['angle'], rotation_point='center', edgecolor='r', facecolor='none', linewidth=2)
            ax.add_patch(rect)

            # Plot the mean image for the FOV
            img_path = self.session_path.joinpath('alf', f'FOV_{i:02}', 'mpciMeanImage.images.npy')
            if img_path.exists():
                img = np.load(img_path)
                img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # Overlay the scaled mean image onto the reference image
                ax.imshow(img, extent=fov['extent'], alpha=0.5, cmap='jet', aspect='equal')
        ax.set_xlim(xlim), ax.set_ylim(ylim)  # Reset limits to original
        ax.set_xlabel('x axis / mm'), ax.set_ylabel('y axis / mm')
        plt.show()
        return f, ax

    def _load_reference_stack_mlapdv(self, display=True):
        """Load the registered MLAPDV coordinates for the reference stack.
        
        Returns
        -------
        numpy.array
            A float array with shape (h, w, 3), comprising Allen atlas MLAPDV coordinates.
            The first two dimensions (h, w) should equal those of the reference stack.
        
        """
        assert self.reference_session
        reference_session_path = self.one.eid2path(self.reference_session)
        file = self.get_atlas_registered_reference_mlap(reference_session_path, clobber=False)
        ccf_idx = np.load(file)  # shape (h, w, 3) - ml, ap, dv indices
        # ccf_idx = self._load_reference_stack_registered()  # height, width, mlapdv
        # ba = MRITorontoAtlas(res_um=25)  # TODO Confirm atlas type with Steven
        ba = self.atlas  # NB: 25um matches the resolution used in the alignment
        # assert ccf_idx.shape == reference_stack.shape
        # ccf_idx * ba.res_um
        # 
        # ccf_idx = np.flip(ccf_idx, axis=1)
        # ba.label = np.flip(ba.label, axis=0)
        # ba.label = np.rot90(ba.label, k=2)

        """
        The Allen volume used for structural alignment is backward: the origin of CCF index array
        is the posterior dorsal left voxel, whereas the origin of Atlas label volume is the
        anterior dorsal left voxel. Additionally, the CCF index array is (ml, ap, dv) while the
        Allen atlas labels volume has the shape (ap, ml, dv).

        Below we transform the AP origin from posterior to anterior.
        """
        ccf_idx[:, :, 1] = np.abs(ccf_idx[:, :, 1].astype('int64') - ba.label.shape[0]).astype(ccf_idx.dtype)
        # ccf_idx = ccf_idx.astype('int64')
        # Look up the MLAPDV coordinates using the registered CCF indices
        xyz = ba.ccf2xyz(ccf_idx * ba.res_um, ccf_order='mlapdv') * 1e6  # m -> μm

        if display:  # FIXME Display using flatmap plot instead
            labels = ba.get_labels(xyz)
            acronyms = ba.regions.id2acronym(labels)

            # Generate a colour map
            L = np.arange(np.unique(acronyms).size)
            # plt.pcolor(X, Y, v, cmap=cm)
            # TODO This plots as x=AP, y=ML - should rotate first
            l = np.fromiter(map(np.unique(acronyms).tolist().index, acronyms.flat), 'uint8').reshape(*labels.shape)
            ax = plt.imshow(l, cmap='hsv')
            # Add discrete colorbar with acronyms as labels
            from matplotlib import ticker
            cbar = plt.colorbar(ax, ticks=L, orientation='vertical')
            cbar.ax.set_yticklabels(np.unique(acronyms)[L])
            cbar.ax.set_ylabel('Acronym')
            # cbar.ax.set_xlabel('Region')

            plt.show()
            # from iblatlas.flatmaps import FlatMap
            # flmap = FlatMap(flatmap='dorsal_cortex', res_um=ba.res_um)

            # from iblatlas.plots import plot_scalar_on_flatmap, plot_scalar_on_slice
            # fig, ax = plot_scalar_on_slice(np.array(['void']), np.array([0]), slice='top', mapping=None, hemisphere='left',
            #                         background='image', cmap='viridis', clevels=None, show_cbar=False, empty_color='silver',
            #                         brain_atlas=ba, ax=None, vector=False, slice_files=None)
            # TODO Plot rectangle over imaging window area
            # ax.patch(xyz[0, 0, 0], xyz[0, 0, 1], xyz.shape[0], xyz.shape[1])

        if reference_session_path.session_parts != self.session_path.session_parts:
            # Ensure reference session files present
            signature = {'input_files': self.signature['input_files'][-3:], 'output_files': []}
            assert all(x.identifiers[-1].startswith('reference') for x in signature['input_files'])
            if self.location == 'server' and self.force:
                handler = dh.ServerGlobusDataHandler(reference_session_path, signature, one=self.one)
            else:
                handler = self.data_handler.__class__(reference_session_path, signature, one=self.one)
            handler.setUp()
            # Apply transform
            display=True
            aligned, transform = register_reference_stacks(self.session_path, reference_session_path, crop_size=390, apply_threshold=False, display=display)

        return xyz  # reference_stack_mlapdv

    def update_craniotomy_center(self, referenceImage):
        """Update subject JSON with atlas-aligned craniotomy coordinates."""
        assert not self.one.offline
        yx_res = np.array([
            referenceImage['meta']['rawScanImageMeta']['YResolution'],
            referenceImage['meta']['rawScanImageMeta']['XResolution']
        ])
        if referenceImage['meta']['rawScanImageMeta']['ResolutionUnit'].casefold() == 'centimeter':
            # NB: these values are (y, x) in μm
            px_per_um = yx_res * 1e-4
            um_per_px = 1 / px_per_um
        else:
            raise NotImplementedError('Reference image resolution unit must be in centimeters')

        ref_stack_n_px = np.array(referenceImage['mlapdv'].shape[:2])  # in (y, x)
        craniotomy_center_offset = np.flip(self.get_window_center(referenceImage['meta']) * 1e3)  # (y, x) center offset mm -> μm

        image_center_px = ref_stack_n_px / 2
        # TODO Verify whether offset is added or subtracted
        #  empirically, it seems to be added looking at SP037/2023-02-20/001
        craniotomy_pixel = image_center_px + (craniotomy_center_offset / um_per_px)
        craniotomy_pixel = np.round(craniotomy_pixel).astype(int)  # convert to pixel coordinates
        _logger.debug('Craniotomy pixel coordinates: (%d, %d)', *craniotomy_pixel)

        craniotomy_resolved = referenceImage['mlapdv'][*craniotomy_pixel] / 1e3  # ML AP DV, μm -> mm

        subject = self.session_path.subject
        json = self.one.alyx.rest('subjects', 'read', id=subject)['json']
        # TODO Assert only one craniotomy key
        if sum(k.startswith('craniotomy_') for k in json.keys()) > 1:
            raise NotImplementedError('Multiple craniotomies found')
        
        # TODO update ['craniotomy_00']['center_resolved']
        data = {'craniotomy_00': json['craniotomy_00'].copy()}
        data['craniotomy_00']['center_resolved'] = np.round(craniotomy_resolved[:2], 3).tolist()
        _logger.info(
            'Craniotomy target: (%.2f, %.2f), actual: (%.2f, %.2f), difference: (%.2f, %.2f)',
            *json['craniotomy_00']['center'], *data['craniotomy_00']['center_resolved'],
            *np.array(json['craniotomy_00']['center']) - craniotomy_resolved[:2]
        )

        return self.one.alyx.json_field_update('subjects', subject, data=data)

    def interpolate_FOVs(self, referenceImage, meta, display=False):
        """Interpolate the FOV coordinates from reference stack coordinates.

        """
        # Extract the reference image and mean image extents in mm along the coverslip, relative to the craniotomy center
        assert np.all(self.get_window_center(referenceImage['meta']) == self.get_window_center(meta))
        assert referenceImage['meta']['scanImageParams']['objectiveResolution'] == meta['scanImageParams']['objectiveResolution']
        coordinates = self.get_fov_objective_extent(meta)

        # Reference image contains 3-D coordinates in m for each pixel of the reference image
        height, width = referenceImage['mlapdv'].shape[:2]
        
        # The fields of view and reference image extents are in the same coordinate space (objective space in mm)
        # Create objective coordinates directly using linear transformation
        ref_extent = self.get_reference_image_extent(referenceImage['meta'])
        r_left, r_right, r_top, r_bottom = ref_extent
        
        # Create coordinate arrays
        x_coords = np.linspace(r_left, r_right, width)  # x coordinates for each column
        y_coords = np.linspace(r_top, r_bottom, height)  # y coordinates for each row

        # Create meshgrid for all pixel locations in objective space
        xx_ref, yy_ref = np.meshgrid(x_coords, y_coords)
        
        # This approach guarantees unique coordinates (no precision issues from interpolation)
        _logger.debug(f"Grid construction: {height}x{width} -> {height*width} points")
        _logger.debug(f"X range: {x_coords[0]:.6f} to {x_coords[-1]:.6f}")
        _logger.debug(f"Y range: {y_coords[0]:.6f} to {y_coords[-1]:.6f}")
        
        # Get interpolator for mlapdv coordinates at each reference image objective coordinate
        # Use nearest neighbour interpolation to get the nearest mlapdv coordinate for each pixel
        points = np.column_stack((xx_ref.ravel(), yy_ref.ravel()))
        values = referenceImage['mlapdv'].reshape(-1, referenceImage['mlapdv'].shape[-1])
        interp = NearestNDInterpolator(points, values)

        # Sanity check: center of the window
        centre = ((r_left + r_right)/2, (r_top + r_bottom)/2)
        # Test the interpolation with the exact center point from our grid
        center_flat_idx = np.ravel_multi_index((height // 2, width // 2), (height, width))
        center_point_from_grid = points[center_flat_idx]
        center_mlapdv = interp(center_point_from_grid)
        expected = referenceImage['mlapdv'][height // 2, width // 2]
        assert np.allclose(center_mlapdv, expected), f'Expected {expected}, got {center_mlapdv} at centre={centre}'

        if display:
            # For sanity, plot a rectangle of the reference image window extent, then plot each pixel of each FOV
            fig, ax = plt.subplots()
        else:
            fig, ax = None, None

        # Interpolate FOV coordinates from reference mlapdv coordinates
        mlapdv = []
        for fov, fov_meta in tqdm(zip(coordinates, meta['FOV'])):
            # Plot pixel locations of each FOV
            width, height = fov_meta['nXnYnZ'][:2]

            # Define the values at the corners of the grid
            if fov['angle'] != 0:
                raise NotImplementedError(f'FOV {i} has non-zero angle of {fov["angle"]} degrees')
            # The four corners of the FOV in mm along coverslip.
            left, right, top, bottom = fov['extent']
            # Create coordinate arrays
            x_coords = np.linspace(left, right, width)  # x coordinates for each column
            y_coords = np.linspace(top, bottom, height)  # y coordinates for each row
            # Create meshgrid for all pixel locations in objective space
            xx, yy = np.meshgrid(x_coords, y_coords)

            # For each of these points, interpolate the mlapdv coordinates using the reference mlapdv
            # NearestNDInterpolator expects points as a 2D array
            points_to_interpolate = np.column_stack((xx.ravel(), yy.ravel()))
            interpolated_values = interp(points_to_interpolate)
            mlapdv.append(interpolated_values.reshape(height, width, 3))

            if display:
                # For each pixel, plot a specific color based on duplicate values along the first axis for interpolated values
                unique_values = np.unique(interpolated_values, axis=0)
                for i in range(len(unique_values)):
                    colour = plt.cm.tab20(i % 20)  # Use a colormap to get distinct colors
                    indices = np.where(np.all(interpolated_values == unique_values[i], axis=1))[0]
                    ax.scatter(xx.ravel()[indices],
                               yy.ravel()[indices],
                               s=1, color=colour, alpha=0.5, label=f'FOV {fov_meta["roiUUID"]}')

            assert not np.any(np.isnan(mlapdv[-1]))

        if display:
            # Simple display of each FOV's interpolated pixels in objective space
            ax.add_patch(plt.Rectangle((r_left, r_top), r_right - r_left, r_bottom - r_top, fill=False, color='black'))
            ax.set_xlim([ref_extent[0] - 1, ref_extent[1] + 1])
            ax.set_ylim([ref_extent[2] - 1, ref_extent[3] + 1])
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')

        return mlapdv

    def get_brain_surface_plane_from_ref_points(self, reference_image: dict):
        """Get the brain surface plane from reference surface points.

        From the reference points, calculate a plane that approximates the brain surface and it's
        normal. Additionally, returns the average depth of the three points that is later used to
        adjust the apparent depth of a cell.

        Parameters
        ----------
        reference_image : dict
            A referenceImage object with keys ('meta', 'points').

        Returns
        -------
        p_ref : np.ndarray
            The point on the plane.
        n_ref : np.ndarray
            The normal vector of the plane.
        dv_avg : float
            The average depth value of the surface points.

        """
        points = reference_image['points']['points']
        stack_ixs = [point['stack_idx'] for point in points]
        # The depth of each point is the z-coordinate from the stack
        stack_dv = np.array(reference_image['meta']['scanImageParams']['hStackManager']['zs'])[stack_ixs]
        dv_avg = np.average(stack_dv)
        # Image coordinates here are fractional x, y values where (0, 0) = left top; (1, 1) = right bottom
        ref_points_rel = np.fliplr(np.array([point['coords'] for point in points], dtype=float))  # (y, x)
        if 'mlapdv' in reference_image:
            # Convert to pixel coordinates
            ref_points_px = (ref_points_rel * reference_image['mlapdv'].shape[:2]).astype(int)
            # MLAPDV coordinates for each of the three points chosen
            ref_points_mlap = reference_image['mlapdv'][*np.hsplit(ref_points_px, 2)].squeeze()
        else:
            raise NotImplementedError
            # ref_points_mlap = cs2d.transform(ref_points_rel, 'image', 'mlap')

        # FIXME review dv - ignoring the resolved DV here
        stack_dv_m = (stack_dv[:, np.newaxis] - dv_avg) / 1e6  # μm -> m
        # stack_dv_m = stack_dv[:, np.newaxis] / 1e6
        ref_points_ = np.c_[ref_points_mlap[:, :-1], stack_dv_m]

        n_ref = surface_normal(ref_points_)
        # invert if pointing downwards
        if n_ref[2] < 0:
            n_ref *= -1
        # Store for plotting the optical axis plane on top of brain surface triangulation later
        self._optical_axis_plane = (ref_points_, n_ref)
        return ref_points_[0, :], n_ref, dv_avg

    def correct_fov_depth_and_surface_projection(
            self,
            fov_mlapdv: np.ndarray,
            meta: dict,
            reference_image: dict,
    ) -> np.ndarray:
        """
        Correct FOV pixel coordinates for imaging depth and project onto the brain surface plane.

        This method accounts for tilt between the imaging plane and the brain surface, adjusting
        ML/AP coordinates and depth (DV) for each pixel. It projects each pixel onto the brain
        surface plane (defined by reference points), then computes the true depth below the surface.

        Parameters
        ----------
        fov_mlapdv : np.ndarray
            Array of MLAPDV coordinates for each FOV pixel.
        meta : dict
            Imaging metadata containing FOV information.
        reference_image : dict
            Reference image dictionary with surface points.

        Returns
        -------
        np.ndarray
            Array of corrected MLAPDV coordinates for each FOV pixel.
        """
        fov_mlap_rel = []
        # Get brain surface plane and normal from reference points
        p_ref, n_ref, dv_avg = self.get_brain_surface_plane_from_ref_points(reference_image)
        for i, (fov, fov_meta) in enumerate(zip(fov_mlapdv, meta['FOV'])):
            # Convert FOV depth from micrometers to meters
            z = -1 * (fov_meta['Zs'] - dv_avg) # depth below reference plane (μm), positive = deeper
            _logger.info(f"FOV {i}: Original Zs={fov_meta['Zs']:.1f}μm, dv_avg={dv_avg:.1f}μm, converted depth z={z:.6f}m")

            # Replace surface dv with imaging depth
            fov_ = fov.copy()
            fov_[:, :, 2] = z
            fov_ = fov_.reshape(-1, 3)

            mlap_rel = np.empty_like(fov_)

            for i, px_mlapdv in enumerate(fov_):
                # Project pixel onto brain surface plane
                mlap_rel[i] = intersect_line_plane(px_mlapdv, n_ref, p_ref, n_ref)
                # Compute true depth below surface
                mlap_rel[i, 2] = np.sqrt(np.sum((mlap_rel[i] - px_mlapdv) ** 2))

            fov_mlap_rel.append(mlap_rel.reshape(*fov.shape))

        return fov_mlap_rel

    def mlapdv_from_rel(
        self,
        mlapdv_rel: np.ndarray,
        atlas_res: int = 50,
    ):
        """now we have corrected ml ap coordinates of cells in the imaging plane
        we first need to determine where those cells are on the surface of the atlas
        and from that point, move down along the local brain normal by the true dv

        project onto the atlas either along the brain normal of the atlas, or the adjusted brain
        normal as calculated from the reference points. I think the atlas normal makes more sense
        (as the influence of the difference between the two angles has been accounted for) but
        left in here optionally to test.

        Args:
            mlapdv_rel (np.ndarray): rois_mlapdv as expressed in the imaging plane. from previous step
            ref_img_meta (dict): _description_
            ref_surface_points (dict): _description_
            project_along_reference (bool, optional): . Defaults to False.

        Returns:
            _type_: _description_
        """
        from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM, MRITorontoAtlas
        from ibllib.mpci.brain_meshes import calculate_surface_triangulation, get_plane_at_point_mlap
        # atlas = MRITorontoAtlas(atlas_res)        
        atlas = AllenAtlas(res_um=25)  # 25 μm resolution
        atlas.compute_surface()
        # vertices, connectivity_list = calculate_surface_triangulation(atlas)  # Doesn't work for some reason
        # Load triangulation in μm
        vertices, connectivity_list = self.load_triangulation(atlas=atlas)  # atlas=atlas

        _logger.info(f'Min-max ML vertex: {vertices[:, 0].min():.6f} to {vertices[:, 0].max():.6f} meters')
        _logger.info(f'Min-max AP vertex: {vertices[:, 1].min():.6f} to {vertices[:, 1].max():.6f} meters')
        _logger.info(f'Min-max DV vertex: {vertices[:, 2].min():.6f} to {vertices[:, 2].max():.6f} meters')

        mlapdv = []
        mlapdv_surface = []
        # Pre-calculate triangulation data for optimization
        # Cache triangle equations (plane coefficients) for all triangles
        for fov in tqdm(mlapdv_rel):
            fov_flat = fov.reshape(-1, 3) * 1e6  # fov is in m, here we convert toum. Now both triangulation and fov values are in μm
            fov_mlapdv = np.empty_like(fov_flat)
            fov_mlapdv_surface = np.empty_like(fov_flat)

            # Vectorized triangle finding
            mlap_points = fov_flat[:, :2]

            # Find triangles for all points at once (more efficient than loop)
            face_indices = np.fromiter(
                (find_triangle(mlap, vertices[:, :2], connectivity_list.astype(np.intp)) for mlap in mlap_points),
                dtype=np.intp
            )
 
            # Group points by triangle for batch processing
            unique_faces = np.unique(face_indices)

            for face_idx in unique_faces:
                # Get all points that belong to this triangle
                point_mask = face_indices == face_idx
                point_indices = np.where(point_mask)[0]

                if len(point_indices) == 0:
                    continue

                # Get triangle vertices and calculate normal once per triangle
                face_vertices = vertices[connectivity_list[face_idx, :], :]
                n = surface_normal(face_vertices)

                # Ensure normal points deeper into brain (positive DV direction)
                # Since DV increases with depth, normal should point in +DV direction
                if n[2] < 0:
                     n *= -1
                # TODO cache
                abc, *_ = np.linalg.lstsq(face_vertices, np.ones(3), rcond=None)

                # Vectorized surface point calculation for all points in this triangle
                mlap_batch = fov_flat[point_indices, :2]
                coord_dv_batch = (1 - mlap_batch @ abc[:2]) / abc[2]
                surface_points = np.column_stack([mlap_batch, coord_dv_batch])

                # Apply depths vectorized
                depths = fov_flat[point_indices, 2]

                # Debug: Check depth values
                _logger.info(f"Triangle {face_idx}: depths range {depths.min():.6f} to {depths.max():.6f} meters")
                _logger.info(f"Normal vector: {n}")

                final_points = surface_points + np.outer(depths, n)
                final_points_ = np.array([p + n * -1 * d for p, d in zip(surface_points, depths)])
                # final_points = surface_points + n * -1 * depths
                # fov_mlapdv[i] = p + n * -1 * fov_flat[i, 2]  # <- the true depth

                # Store results
                fov_mlapdv_surface[point_indices] = surface_points / 1e6
                fov_mlapdv[point_indices] = final_points / 1e6

            mlapdv.append(fov_mlapdv.reshape(*fov.shape))
            mlapdv_surface.append(fov_mlapdv_surface.reshape(*fov.shape))
        import pickle
        with open(self.session_path / 'mlapdv_final.pkl', 'wb') as f:
            pickle.dump([mlapdv, mlapdv_surface], f)
        
        #### PLOTS ####
        # Old surf
        # _vertices, _ = self.load_triangulation(legacy=True)
        # axes = plt.figure(figsize=[10, 10]).add_subplot(projection="3d")
        # axes.plot_trisurf(_vertices[:, 0], _vertices[:, 1], _vertices[:, 2], linewidth=0.2, antialiased=True)
        # New surf
        axes = plt.figure(figsize=[10, 10]).add_subplot(projection="3d")
        axes.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], linewidth=0.2, antialiased=True, alpha=0.5)
        # Plot optical axis plane (optical axis is 3x3 array of three 3-D vertices)
        optical_axis, normal = self._optical_axis_plane
        axes.plot_trisurf(optical_axis[:, 0], optical_axis[:, 1], optical_axis[:, 2], linewidth=0.2, antialiased=True)
        # Plot the optical axis as a much larger plane so we can see where it intersects with the surface
        
        # Create a large rectangular plane extended from the optical axis plane
        # Use the center point of the optical axis triangle and the normal vector
        center_point = np.mean(optical_axis, axis=0)
        # Define the size of the large rectangular plane
        plane_size = 0.01  # 10mm extension in each direction
        # Create two orthogonal vectors in the plane
        # First, find a vector that's not parallel to the normal
        if abs(normal[0]) < 0.9:
            v1 = np.cross(normal, [1, 0, 0])
        else:
            v1 = np.cross(normal, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)  # normalize
        # Second orthogonal vector
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)  # normalize
        # Create the four corners of the rectangular plane
        large_plane_corners = np.array([
            center_point - plane_size * v1 - plane_size * v2,
            center_point + plane_size * v1 - plane_size * v2,
            center_point + plane_size * v1 + plane_size * v2,
            center_point - plane_size * v1 + plane_size * v2
        ])
        # Plot the large rectangular plane
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        axes.add_collection3d(Poly3DCollection([large_plane_corners], alpha=0.3, linewidths=1, 
                                              edgecolors='red', facecolors='yellow', label='Extended optical axis plane'))


        # plot brain surface points
        from ibllib.mpci.brain_meshes import get_surface_points
        from ibllib.mpci.plotters import plot_brain_surface_points
        brain_surface_points = get_surface_points(atlas)
        axes = plot_brain_surface_points(brain_surface_points, ds=4, axes=axes)
        # Plot orginal MLAPDV points (but only the ROIs) as 3D scatter
        for i, fov in enumerate(mlapdv_rel):
            yx_pos = alfio.load_file_content(self.session_path / f'alf/FOV_{i:02}' / 'mpciROIs.stackPos.npy')
            roi_mlapdv = fov[yx_pos[:, 0], yx_pos[:, 1], :]
            axes.scatter(*roi_mlapdv.T, ".", c="b", s=1, alpha=0.05, label='Original MLAPDV points')
        # Plot the triangles that were used
        unique_faces = np.unique(face_indices)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for face in unique_faces:
            face_vertices = vertices[connectivity_list[face, :], :]
            axes.add_collection3d(Poly3DCollection([face_vertices], alpha=.25, linewidths=1, edgecolors='r'))

        # Plot ROIs
        for i, fov in enumerate(mlapdv):
            # Plot the points in mlapdv space
            yx_pos = alfio.load_file_content(self.session_path / f'alf/FOV_{i:02}' / 'mpciROIs.stackPos.npy')
            roi_mlapdv = fov[yx_pos[:, 0], yx_pos[:, 1], :] * 1e6
            axes.scatter(*roi_mlapdv.T, ".", c="k", s=1, alpha=0.05, label='ROIs in imaging plane')
            # roi_surf = mlapdv_surface[i][yx_pos[:, 0], yx_pos[:, 1], :] * 1e6
            # axes.scatter(*roi_surf.T, ".", c="r", s=1, alpha=0.05, label='ROIs on brain surface')
        
        # Load Georg's
        import pickle
        with open(self.session_path / 'mlapdv_final_georg_simplified.pkl', 'rb') as f:
            processed = pickle.load(f)
        for i, fov in enumerate(processed):
            yx_pos = alfio.load_file_content(self.session_path / f'alf/FOV_{i:02}' / 'mpciROIs.stackPos.npy')
            roi_mlapdv = fov[yx_pos[:, 0], yx_pos[:, 1], :] * 1e6
            axes.scatter(*roi_mlapdv.T, ".", c="r", s=1, alpha=0.05, label='ROIs in imaging plane')

        return mlapdv
        # brain_surface_points = get_surface_points(atlas)

        # get the brain normal from ml, ap
        # FIXME Use updated center coordinates

        # center_mlap = np.array([ref_img_meta["centerMM"][d] for d in ["ML", "AP"]]) * 1e3
        # center_mlapdv, brain_normal = get_plane_at_point_mlap(
            # *center_mlap, vertices, connectivity_list, numba=True
        # )
        # # create a new tilted coordinate system at the imaged plane
        # cs3d = setup_coordinate_systems_3d(center_mlapdv, brain_normal)

        # get the roi mlapdv values in the atlas space
        # again, just the incorrectly termed ml,ap coordinates. Setting DV to zero because the imaging plane is flat
        # and the points are just used for projection onto the brain surface
        
        # mlap0 = np.copy(mlapdv_rel)

        # mlap0[:, 2] = 0
        # _rois_mlapdv = cs3d.transform(mlap0 - center_mlapdv, "imaging_plane", "mlapdv")

        # project the rois onto the brain surface along the brain normal
        # adjusted for the sessions tilt
        # FIXME Skip this for histology resolved data!
        # rois_on_surface = np.zeros_like(_rois_mlapdv)
        # for i, roi in enumerate(tqdm(_rois_mlapdv)):
        #     faces, ips, ix = intersect_line_mesh_nb(
        #         vertices,
        #         connectivity_list,
        #         roi,
        #         brain_normal * -1,  # discuss if brain_normal or brain_normal_ref
        #     )
        #     face, ix = get_closest_face(faces, roi)
        #     rois_on_surface[i] = ips[ix]

        # and now go inward along the local brain normal, with true depth
        # this is the step that should be sensitive to the atlas resolution
        # as the local brain normal will differ more from ROI to ROI
        # rois_mlapdv = np.zeros_like(rois_on_surface)
        # for i, point in enumerate(tqdm(rois_on_surface)):
        #     p, n = get_plane_at_point_mlap(
        #         point[0], point[1], vertices, connectivity_list, numba=True
        #     )
        #     rois_mlapdv[i] = p + (n * -1 * mlapdv_rel[i, 2])  # <- the true depth

        # return (  # these are just now returned for plotting purposes
        #     rois_mlapdv,
        #     rois_on_surface,
        #     _rois_mlapdv,
        #     center_mlapdv,
        #     brain_normal,
        #     atlas,
        #     cs3d,
        # )

    def project_mlapdv_from_surface(self, mlapdv_rel: np.ndarray):
        """
        Project corrected MLAPDV coordinates from the imaging plane onto the Allen atlas surface.

        For each pixel in the FOV, this method finds the corresponding location on the atlas brain surface
        using the ML/AP coordinates, then projects the pixel along the local surface normal by the true depth (DV).
        This accounts for tilt and depth differences between the imaging plane and the brain surface.

        Parameters
        ----------
        mlapdv_rel : np.ndarray
            List of arrays, one per FOV, each with shape (height, width, 3), containing MLAPDV coordinates
            in microns for each pixel in the imaging plane (with corrected ML/AP and DV).

        Returns
        -------
        list of np.ndarray
            List of arrays, one per FOV, each with shape (height, width, 3), containing MLAPDV coordinates
            in microns for each pixel projected onto the Allen atlas surface.

        Notes
        -----
        - This method uses the Allen atlas surface triangulation to find the local surface normal and position.
        - The output can be used for downstream analysis or registration to atlas space.
    
        TODO combine with correct_fov_depth_and_surface_projection
        """
        vertices, connectivity_list = self.load_triangulation()
        processed = []
        for n, fov in enumerate(mlapdv_rel):
            fov_flat = fov.reshape(-1, 3)
            fov_mlapdv = np.empty_like(fov_flat)
            mlap_points = fov_flat[:, :2] * 1e6  # Convert m -> μm for precision
            for i, point in tqdm(enumerate(mlap_points), total=len(mlap_points), desc=f'Projecting MLAPDV points {n+1}/{len(mlapdv_rel)}'):
                p, n_vec = get_plane_at_point_mlap(point[0], point[1], vertices, connectivity_list)
                fov_mlapdv[i] = p + n_vec * -1 * fov_flat[i, 2]  # Project by true depth
            processed.append(fov_mlapdv.reshape(fov.shape))

        return processed

    def load_triangulation(self, legacy=False, atlas=None, display=False):
        """Load surface triangulation.

        Parameters
        ----------
        legacy : bool
            If True, load the legacy triangulation from file (MATLAB generated).

        Returns
        -------
        vertices : numpy.ndarray
            A numpy array containing the surface vertices in μm.
        connectivity_list : numpy.ndarray
            An (N, 3) numpy array containing the surface connectivity information.
        
        TODO Used atlas attribute to save loading multiple times
        """
        if legacy:
            # Load legacy triangulation from file
            points, dorsal_connectivity_list = super().load_triangulation()
            return points / 1e6, dorsal_connectivity_list
        elif atlas is None:
            atlas = self.atlas
        if not atlas.surface:
            atlas.compute_surface()
        # Get surface points
        ap_grid, ml_grid = np.meshgrid(atlas.bc.yscale, atlas.bc.xscale)  # now this indexes into AP, ML
        points = (
            np.stack(
                [ml_grid.T.flatten(), ap_grid.T.flatten(), atlas.top.flatten()], axis=1
            ) * 1e6
        )
        points = points[~np.isnan(points[:, 2])]
        # Compute triangulation
        hull = ConvexHull(points)
        connectivity_list = hull.simplices
        # Remove junk points (those that do not participate in the convex hull)
        k_unique = np.unique(connectivity_list)
        points = points[k_unique, :]
        # Recompute with updated points
        hull = ConvexHull(points)
        connectivity_list = hull.simplices
        # # Only keep faces that have normals pointing up (positive DV value).
        # # Calculate the normal vector pointing out of the convex hull.
        # triangles = points[connectivity_list, :]
        # normals = surface_normal(triangles)
        # up_faces, = np.where(normals[:, -1] > 0)
        # # only keep triangles that have normal vector with positive DV component
        # dorsal_connectivity_list = connectivity_list[up_faces, :]
        dorsal_connectivity_list = connectivity_list
        if display:  # Display here or outside?
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], linewidth=0.2, antialiased=True)
        return points, dorsal_connectivity_list

    def reproject(self, points, display=False):
        """TODO Document and rename.

        This method will estimate position based on a plane drawn from experimenter defined points at the brain's surface.
        For this a JSON of selected points is required as well as the reference image and its meta file.

        Parameters
        ----------
        points : dict
            A dictionary of points selected by the experimenter.  Expected format:
            {'points': [{'stack_idx': int, 'coords': [int, int]}, ...], 'range': [int, int]}.

        Assumptions: all on the same X plane, all the same width
        """
        stack, ref_meta = self._load_reference_stack()
        xy_res = np.array([
            ref_meta['rawScanImageMeta']['XResolution'],
            ref_meta['rawScanImageMeta']['YResolution']
        ])
        if ref_meta['rawScanImageMeta']['ResolutionUnit'].casefold() == 'centimeter':
            # NB: these values are (x, y) in μm and shouldn't be used with mlap coordinates without rotation
            px_per_um = xy_res * 1e-4
            um_per_px = 1 / px_per_um
        else:
            raise NotImplementedError('Reference image resolution unit must be in centimeters')
        # these can not be used because they seem to refer to the raw acquisition and not the stack
        # ref_meta["rawScanImageMeta"]["Height"], ref_meta["rawScanImageMeta"]["Width"]

        # need to get the image shape from the .stack and not the .raw
        ref_stack_n_px = np.flip(np.array(stack.shape[1:]))  # in (x, y)
        # known: the point in the image in pixel coordinates and mlap space
        center = ref_meta['centerMM']
        # An array of the positive ML and AP directions in (x, y) pixel space
        rotation_matrix = np.c_[ref_meta['imageOrientation']['positiveML'],
                                ref_meta['imageOrientation']['positiveAP']]
        craniotomy_center_offset = np.array([center['x'], center['y']]) * 1e3  # μm from center
        mlap = np.array([center['ML'], center['AP']]) * 1e3  # μm from bregma

        # Where is bregma in pixel space?
        # ml is along the y axis, and is positive, so longditudinal fissure is at a large +ve y value
        # ref_stack_n_px[1]/2 = 270 px, which is mlap[0] = 2600 μm from bregma, so absolute bregma ml
        # distance is mlap[0] * px_per_um[1] = 260 so bregma is at 260 + 270 = 530 px
        #
        # ap is along the x axis, and is negative, so the back of the brain is at a large -ve x value
        # ref_stack_n_px[0]/2 = 245 px, which is mlap[1] = -2000 μm from bregma, so absolute bregma ap
        # distance is mlap[1] * px_per_um[0] = 200 so bregma is at 245 - 200  = 45 px

        def px2um(px):
            """Map pixel (x, y) coordinates to MLAP coordinates."""
            # Calculate the pixel offset from the image center
            image_center_px = ref_stack_n_px / 2
            craniotomy_pixel = image_center_px + (craniotomy_center_offset / um_per_px)
            pixel_offset = px - craniotomy_pixel  # origin now the craniotomy pixel

            # Apply the scaling factor to convert pixel distances to mlap units
            pixel_offset_um = pixel_offset * um_per_px

            # Rotate the pixel offset using the orientation vector
            inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            rotated_offset = np.dot(pixel_offset_um, inv_rotation_matrix)

            # Translate the rotated coordinates to the mlap space using the craniotomy coordinates
            mlap_coords = mlap + rotated_offset

            return mlap_coords

        def um2px(um):
            """Maps mlap coordinates to pixel space."""
            # Calculate the mlap offset from the craniotomy coordinates
            mlap_offset = np.array(um) - np.array(mlap)
            # Rotate the mlap offset using the orientation vector
            rotated_offset_xy = np.dot(mlap_offset, rotation_matrix)

            # Apply the scaling factor to convert mlap distances to pixel units
            rotated_offset_px = rotated_offset_xy / um_per_px

            # Translate the rotated coordinates to the pixel space using the craniotomy pixel coordinates
            image_center_px = ref_stack_n_px / 2
            craniotomy_pixel = image_center_px + (craniotomy_center_offset / um_per_px)
            return craniotomy_pixel + rotated_offset_px


        # Sanity checks
        bregma_px = np.array([45, 530])  # (x, y) coordinates of bregma in pixel space
        craniotomy_px = um2px(mlap)
        np.testing.assert_array_equal(um2px(mlap), ref_stack_n_px / 2)
        np.testing.assert_array_equal(px2um(craniotomy_px), mlap)
        np.testing.assert_array_equal(np.round(um2px([0, 0])).astype(int), bregma_px)
        np.testing.assert_array_equal(px2um(um2px([0, 0])), [0, 0])

        # Check works with multiple points
        bregma_um = np.zeros((3, 2))
        np.testing.assert_array_equal(np.round(um2px(bregma_um)), np.tile(bregma_px, (3, 1)))
        np.testing.assert_array_equal(px2um(np.tile(craniotomy_px, (3, 1))), np.tile(mlap, (3, 1)))

        # TODO All px
        x1, x2 = np.meshgrid(np.arange(ref_stack_n_px[0]), np.arange(ref_stack_n_px[1]))
        xy_coords = np.array((x1, x2)).T.reshape(-1, 2)
        px2um(xy_coords)

        if display:  # pragma: no cover
            for i, point in enumerate(points['points']):
                # Convert the point to pixels
                x, y = np.array(point['coords']) * ref_stack_n_px
                fig, ax = plt.subplots()
                ax.matshow(stack[point['stack_idx'], :, :], cmap='gray')
                ax.set_xlabel('n px'), ax.set_ylabel('n px')
                ax.xaxis.set_label_position('top')
                ax.plot([x - 24, x + 24], [y, y], lw=2, alpha=0.7, color='r')
                ax.plot([x, x], [y - 24, y + 24], lw=2, alpha=0.7, color='r')
                landmark_coords_um = px2um([x, y])
                ax.annotate(f'landmark ({landmark_coords_um[0]:.4g}, {landmark_coords_um[1]:.4g})',
                            (x, y), color='r', xytext=(10, -10), textcoords='offset points')
                ax.xaxis.tick_top()

                # Plot bregma
                ax.axhline(um2px([0, 0])[1], lw=1, color='blue', alpha=0.5)
                ax.axvline(um2px([0, 0])[0], lw=1, color='blue', alpha=0.5)
                ax.annotate('bregma (0, 0)', um2px([0, 0]), color='b', xytext=(10, -10), textcoords='offset points')

                image_center_px = ref_stack_n_px / 2
                craniotomy_pixel = image_center_px - (craniotomy_center_offset / um_per_px)

                ax.plot(craniotomy_pixel[0], craniotomy_pixel[1], 'go', alpha=0.7, markersize=12, markerfacecolor='none')
                ax.annotate(f'craniotomy ({mlap[0]:.4g}, {mlap[1]:.4g})',
                            craniotomy_px, color='g', xytext=(10, -10), textcoords='offset points')
                # Sadly, secondary axes are not supported for matshow
                # # secax_x = ax.secondary_xaxis('bottom', functions=(px2um, um2px))
                # # secax_x.set_xlabel(('ML' if positive_mlap[0][0] else 'AP') + ' / um')
                # # secax_x.xaxis.set_tick_params(rotation=70
                # # secax_y = ax.secondary_yaxis('right', functions=(px2um, um2px))
                # # secax_y.set_ylabel(('ML' if positive_mlap[0][1] else 'AP') + ' / um')
                # # secax_y.yaxis.set_tick_params(rotation=70)

                fig.canvas.manager.set_window_title(f'Point {i} - stack #{point["stack_idx"]}')
            plt.show()

    def _load_reference_stack(self):
        """Load the referenceImage.stack.tif file and its metadata.

        Loads the files:
        - referenceImage.stack.tif
        - referenceImage.meta.json
        - referenceImage.points.json
        - referenceImage.mlapdv.npy

        The latter is loaded if reference_session is not None.

        Returns
        -------
        iblutil.util.Bunch
            The reference image object with keys ('stack', 'meta', 'points').
            The stack is an array of size (nZ, nY, nX).
        """
        try:
            stack_path = next(self.session_path.glob('raw_imaging_data_??/reference/referenceImage.stack.tif'))
        except StopIteration:
            raise FileNotFoundError('Reference stack not found')
        meta_path = stack_path.with_name('referenceImage.meta.json')
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_path) or {})
        reference_image = {'stack': tifffile.imread(stack_path), 'meta': meta}
        if stack_path.with_name('referenceImage.points.json').exists():
            points_path = stack_path.with_name('referenceImage.points.json')
            reference_image['points'] = alfio.load_file_content(points_path)
        if self.reference_session:
            # Load the mlapdv coordinates for the reference stack
            reference_image['mlapdv'] = self._load_reference_stack_mlapdv(display=False)
            assert reference_image['stack'].shape[1:] == reference_image['mlapdv'].shape[:2], 'Reference stack and MLAPDV coordinates must have the same shape'
        return Bunch(reference_image)


import unittest
import unittest.mock
from one.api import ONE
class TestReferenceSession(unittest.TestCase):
    """Test extraction of FOV coordinates for aligned reference session."""

    def setUp(self):
        self.one = ONE()
        # self.session_path = ALFPath(r'D:\Flatiron\alyx.internationalbrainlab.org\cortexlab\Subjects\SP037\2024-08-01\001')
        self.reference_session = '839bb5b1-120f-49d0-b7c9-5174c0c66b5a'  # SP037/2023-02-20/001
        self.session_path = self.one.eid2path(self.reference_session)
        self.reprojection = Reprojection(self.session_path, one=self.one)
        self.reprojection.reference_session = self.reference_session
        self.reprojection.get_signatures()
        self.reprojection.data_handler = self.reprojection.get_data_handler()
        # Download required datasets
        dsets = self.one.list_datasets(self.session_path)
        required = ['mpciROIs.stackPos.npy', 'experiment.description.yaml',
                    'referenceImage.meta.json', 'referenceImage.stack.tif',
                    '_ibl_rawImagingData.meta.json']
        dsets = [d for d in dsets if any(d.endswith(r) for r in required)]
        # self.one.load_datasets(self.session_path, dsets, download_only=True)  # commented out because of size mismatches


    def test_get_brain_surface_plane_from_ref_points(self):
        """This tests that the output exactly matches Georg's original code for this session."""
        self.reprojection.atlas = AllenAtlas(res_um=25)
        reference_image = self.reprojection._load_reference_stack()
        p_ref, n_ref, dv_avg = self.reprojection.get_brain_surface_plane_from_ref_points(reference_image)
        expected_p_ref = np.array([0.003011, -0.001025, -0.000125])
        expected_n_ref = np.array([7.34976797e-04, 1.20536195e-01, 9.92708661e-01])
        expected_dv_avg = 150.0
        np.testing.assert_array_almost_equal(p_ref, expected_p_ref, decimal=5)
        np.testing.assert_array_almost_equal(n_ref, expected_n_ref, decimal=5)
        self.assertAlmostEqual(dv_avg, expected_dv_avg, delta=1e-2)

    def test_ref_session(self):
        # 1. Download the MLAPDV coordinates of the reference session
        # registered_mlapdv = self.reprojection.get_atlas_registered_reference_mlap(self.reference_session)
        # ref_mlapdv = np.load(registered_mlapdv)
        # mlapdv = self.reprojection.interpolate_FOVs()
        # # 2. 
        # get_rois_mlapdv_rel
        # rois_mlapdv_from_rel
        with unittest.mock.patch.object(self.reprojection, 'update_craniotomy_center'):
            self.reprojection._run()

    def test_ref_session_with_save(self):
        # 1. Download the MLAPDV coordinates of the reference session
        # registered_mlapdv = self.reprojection.get_atlas_registered_reference_mlap(self.reference_session)
        # ref_mlapdv = np.load(registered_mlapdv)
        # mlapdv = self.reprojection.interpolate_FOVs()
        # # 2. 
        # get_rois_mlapdv_rel
        # rois_mlapdv_from_rel
        self.reprojection.atlas = AllenAtlas(res_um=25)
        # Load the reference stack & (down)load the registered MLAPDV coordinates
        reference_image = self.reprojection._load_reference_stack()
        # Load main meta
        _, meta_files, _ = self.reprojection.input_files[0].find_files(self.reprojection.session_path)
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))

        with open(self.session_path / 'interpolated_fovs_rel.pkl', 'rb') as f:
            mlapdv_rel = pickle.load(f)

        # Account for optical plane tilt
        if (self.session_path / 'mlapdv_final_georg.pkl').exists():
            with open(self.session_path / 'mlapdv_final_georg.pkl', 'rb') as f:
                fovs = pickle.load(f)
        else:
            fovs = dict.fromkeys(range(nFOV), None)
        
        # mlapdv_rel = self.correct_fov_depth_and_surface_projection(mlapdv, meta, reference_image)
        done = sum(v is not None for v in fovs.values())
        _logger.info('%i/%i processed', done, nFOV)
        if done == nFOV:
            return
        i = next(i for i in fovs if fovs[i] is None)
        _logger.info('Processing FOV %i', i)

        mean_image_mlapdv = self.reprojection.project_mlapdv_from_surface(mlapdv_rel[i:i+1])
        fovs[i] = mean_image_mlapdv[0]
        with open(self.session_path / 'mlapdv_final_georg.pkl', 'wb') as f:
            pickle.dump(fovs, f)

    def test_project_mlapdv_from_surface_georg(self):
        """This tests that the output exactly matches Georg's original code for this session."""
        self.reprojection.atlas = AllenAtlas(res_um=25)
        # Load test points from file
        with open(self.session_path / 'interpolated_fovs_rel.pkl', 'rb') as f:
            file_result = pickle.load(f)
        fov, idx = np.unique(file_result[0].reshape(-1, 3), axis=0, return_index=True)
        # Load expected results from file
        with open(self.session_path / 'mlapdv_final_georg_simplified.pkl', 'rb') as f:
            expected_result = pickle.load(f)
        # Run the method
        result = self.reprojection.project_mlapdv_from_surface([fov])
        # Compare results
        expected = expected_result[0].reshape(-1, 3)[idx]
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_save_roi(self):
        with open(self.session_path / 'mlapdv_final_georg.pkl', 'rb') as f:
            fovs = pickle.load(f)
        
        with unittest.mock.patch.object(self.reprojection, 'update_craniotomy_center'), \
                unittest.mock.patch.object(self.reprojection, 'interpolate_FOVs'), \
                unittest.mock.patch.object(self.reprojection, 'correct_fov_depth_and_surface_projection'), \
                unittest.mock.patch.object(self.reprojection, 'project_mlapdv_from_surface', return_value=list(fovs.values())):
            self.reprojection._run()


class TestSession(unittest.TestCase):
    """Test extraction of FOV coordinates for non-reference session."""

    def setUp(self):
        self.one = ONE()
        self.session_path = ALFPath(r'D:\Flatiron\alyx.internationalbrainlab.org\cortexlab\Subjects\SP037\2023-03-09\001')
        self.reference_session = '839bb5b1-120f-49d0-b7c9-5174c0c66b5a'  # SP037/2023-02-20/001
        # Download required datasets
        dsets = self.one.list_datasets(self.session_path)
        required = ['mpciROIs.stackPos.npy', 'experiment.description.yaml',
                    'referenceImage.meta.json', 'referenceImage.stack.tif',
                    '_ibl_rawImagingData.meta.json']
        dsets = [d for d in dsets if any(d.endswith(r) for r in required)]
        # self.one.load_datasets(self.session_path, dsets, download_only=True)  # commented out because of size mismatches
        self.reprojection = Reprojection(self.session_path, one=self.one)
        self.reprojection.reference_session = self.reference_session
        self.reprojection.get_signatures()
        self.reprojection.data_handler = self.reprojection.get_data_handler()

    def test_session_with_save(self):
        # 1. Download the MLAPDV coordinates of the reference session
        # registered_mlapdv = self.reprojection.get_atlas_registered_reference_mlap(self.reference_session)
        # ref_mlapdv = np.load(registered_mlapdv)
        # mlapdv = self.reprojection.interpolate_FOVs()
        # # 2. 
        # get_rois_mlapdv_rel
        # rois_mlapdv_from_rel
        self.reprojection.atlas = AllenAtlas(res_um=25)
        # Load the reference stack & (down)load the registered MLAPDV coordinates
        reference_image = self.reprojection._load_reference_stack()
        # Load main meta
        _, meta_files, _ = self.reprojection.input_files[0].find_files(self.reprojection.session_path)
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))

        f = self.session_path / 'interpolated_fovs.pkl'
        if f.exists():
            with open(f, 'rb') as f:
                mlapdv = pickle.load(f)
        else:
            mlapdv = self.reprojection.interpolate_FOVs(reference_image, meta)
            with open(f, 'wb') as f:
                pickle.dump(mlapdv, f)


        f = self.session_path / 'interpolated_fovs_rel.pkl'
        if f.exists():
            with open(f, 'rb') as f:
                mlapdv_rel = pickle.load(f)
        else:
            mlapdv_rel = self.reprojection.correct_fov_depth_and_surface_projection(mlapdv)
            with open(f, 'wb') as f:
                pickle.dump(mlapdv_rel, f)

        # Account for optical plane tilt
        if (self.session_path / 'mlapdv_final_georg.pkl').exists():
            with open(self.session_path / 'mlapdv_final_georg.pkl', 'rb') as f:
                fovs = pickle.load(f)
        else:
            fovs = dict.fromkeys(range(nFOV), None)
        
        # mlapdv_rel = self.correct_fov_depth_and_surface_projection(mlapdv, meta, reference_image)
        done = sum(v is not None for v in fovs.values())
        _logger.info('%i/%i processed', done, nFOV)
        if done == nFOV:
            return
        i = next(i for i in fovs if fovs[i] is None)
        _logger.info('Processing FOV %i', i)

        mean_image_mlapdv = self.reprojection.project_mlapdv_from_surface(mlapdv_rel[i:i+1])
        fovs[i] = mean_image_mlapdv[0]
        with open(self.session_path / 'mlapdv_final_georg.pkl', 'wb') as f:
            pickle.dump(fovs, f)



if __name__ == '__main__':
    suite = unittest.TestSuite()
    # suite.addTest(TestReprojectionUnit("test_interpolate_FOVs"))
    # suite.addTest(TestReprojection('test_load_mlapdv'))
    # suite.addTest(TestReferenceSession('test_save_roi'))
    # suite.addTest(TestReferenceSession('test_project_mlapdv_from_surface_georg'))
    suite.addTest(TestSession('test_session_with_save'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    exit()

r"""
     191688/262144 [05:50<01:28, 794.84it/s]
Projecting MLAPDV points 6/6:  73%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                           | 191760/262144 [05:50<02:08, 547.65it/s]
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "c:\Users\Work\Documents\github\ibllib-repo\ibllib\mpci\registration.py", line 1669, in project_mlapdv_from_surface
    p, n_vec = get_plane_at_point_mlap(point[0], point[1], vertices, connectivity_list)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Work\Documents\github\ibllib-repo\ibllib\mpci\brain_meshes.py", line 87, in get_plane_at_point_mlap
    face, ix = get_closest_face(faces, ln0)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\Work\Documents\iblenv\Lib\site-packages\numba\np\old_arraymath.py", line 615, in array_argmin_impl_float
    raise ValueError("attempt to get argmin of an empty sequence")
ValueError: attempt to get argmin of an empty sequence
"""
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

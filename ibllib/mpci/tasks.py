"""Preprocessing tasks to determine field of view location.

See also :mod:`ibllib.pipes.mesoscope_tasks`.
"""
from pathlib import Path
from collections import Counter
import time
import uuid
import logging
import json
import pickle  # TODO remove once stable

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interpn, NearestNDInterpolator
from scipy.spatial import ConvexHull
import skimage.transform

from one.alf.path import ALFPath, filename_parts
from one.alf.spec import to_alf
import one.alf.io as alfio
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM, MRITorontoAtlas
from iblutil.util import Bunch

from ibllib.io.extractors import mesoscope
import ibllib.oneibl.data_handlers as dh
from ibllib.mpci.brain_meshes import get_plane_at_point_mlap, get_surface_points
from ibllib.mpci.linalg import intersect_line_plane, surface_normal, find_triangle, _update_points
from ibllib.pipes.base_tasks import MesoscopeTask, RegisterRawDataTask
from ibllib.mpci.registration import Provenance, register_reference_stacks

_logger = logging.getLogger(__name__)


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
        - Once the FOVs have been registered they cannot be deleted with this task. Rerunning this
          task will update the FOV locations in Alyx, however if the number of FOVs has changed
          then any extra FOVs will need to be deleted manually in Alyx.
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
        mean_image_mlapdv, mean_image_ids = self.project_mlapdv(meta, provenance=provenance)

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
        self.register_fov(meta, provenance)

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

    def register_fov(self, meta: dict, provenance: Provenance) -> (list, list):
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
        provenance : Provenance
            The provenance of the FOV location.

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
                # Check if FOV already exists
                existing = self.one.alyx.rest('fields-of-view', 'list', session=alyx_FOV['session'],
                                              name=alyx_FOV['name'], imaging_type=alyx_FOV['imaging_type'])
                if any(existing):
                    alyx_fovs.append(existing[0])
                    _logger.debug(f'FOV {alyx_FOV["name"]} already exists in Alyx')
                else:
                    alyx_fovs.append(self.one.alyx.rest('fields-of-view', 'create', data=alyx_FOV))

            # Field of view location
            data = {
                'field_of_view': alyx_fovs[-1].get('id'),
                'default_provenance': True,
                'coordinate_system': 'IBL-Allen',
                'n_xyz': fov['nXnYnZ'],
                'provenance': provenance.name[0]
            }

            # Convert coordinates to 4 x 3 array (n corners by n dimensions)
            # x1 = top left ml, y1 = top left ap, y2 = top right ap, etc.
            d = fov['MLAPDV'][provenance.name.lower()]
            coords = [d[key] for key in ('topLeft', 'topRight', 'bottomLeft', 'bottomRight')]
            coords = np.vstack(coords).T
            data.update({k: arr.tolist() for k, arr in zip('xyz', coords)})

            # Load MLAPDV + brain location ID maps of pixels
            suffix = '' if provenance is Provenance.HISTOLOGY else f'_{provenance.name.lower()}'
            filename = 'mpciMeanImage.brainLocationIds_ccf_2017' + suffix + '.npy'
            filepath = self.session_path.joinpath('alf', f'FOV_{i:02}', filename)
            mean_image_ids = alfio.load_file_content(filepath)

            data['brain_region'] = np.unique(mean_image_ids).astype(int).tolist()

            if dry:
                print(data)
                alyx_FOV['location'].append(data)
            else:
                # Whether to patch or create a new location
                existing = self.one.alyx.rest(
                    'fov-location', 'list', field_of_view=data['field_of_view'], provenance=provenance.name)
                if any(existing):
                    _logger.info(f'Patching FOV location for {alyx_fovs[-1]["name"]}')
                    loc = self.one.alyx.rest('fov-location', 'partial_update', id=existing[0]['id'], data=data)
                else:
                    loc = self.one.alyx.rest('fov-location', 'create', data=data)
                alyx_fovs[-1]['location'].append(loc)
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

    def project_mlapdv(self, meta, atlas=None, provenance=Provenance.ESTIMATE):
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
        provenance : Provenance
            The provenance of the coordinates.  Defaults to ESTIMATE.

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

            if 'MLAPDV' not in fov:
                fov['MLAPDV'] = {}
                fov['brainLocationIds'] = {}
            fov['MLAPDV'][provenance.name.lower()] = {
                'topLeft': MLAPDV[0, 0, :].tolist(),
                'topRight': MLAPDV[0, -1, :].tolist(),
                'bottomLeft': MLAPDV[-1, 0, :].tolist(),
                'bottomRight': MLAPDV[-1, -1, :].tolist(),
                'center': MLAPDV[round(x_px_idx.shape[0] / 2) - 1, round(x_px_idx.shape[1] / 2) - 1, :].tolist()
            }

            # Save the brain regions of the corners/centers of FOV (annotation field)
            fov['brainLocationIds'][provenance.name.lower()] = {
                'topLeft': int(annotation[0, 0]),
                'topRight': int(annotation[0, -1]),
                'bottomLeft': int(annotation[-1, 0]),
                'bottomRight': int(annotation[-1, -1]),
                'center': int(annotation[round(x_px_idx.shape[0] / 2) - 1, round(x_px_idx.shape[1] / 2) - 1])
            }

            mlapdv[i] = MLAPDV
            location_id[i] = annotation
        return mlapdv, location_id


class MesoscopeFOVHistology(MesoscopeFOV):
    """Reprojection of mesoscope FOVs into Allen atlas space.

    After imaging, the brain undergoes histology whereby the tissue is sectioned and imaged, then
    registered to the Allen atlas producing a file containing the Allen atlas volume coordinates
    for each pixel of the full-field image of the craniotomy. With this file we have the 3-D MLAPDV
    coordinates of the brain surface in meters of the full-field image.

    This registration step involves some warping, therefore the distance between two pixels may not
    be uniform in Allen coordinate space. The full-field ('reference') image contains the location
    of each zoomed-in field of view (FOV) in the craniotomy. The reference image and FOVs
    microscope coordinates are in the same space with μm coordinates. We can use this reference
    space to transform the FOVs into the Allen atlas space using nearest neighbor interpolation.
    This is done in the 'interpolate_FOVs' method.

    After this step we have the 3-D coordinates for each pixel of the FOVs in the Allen atlas
    space. These are the coordinates of the brain surface from histology, however the imaging plane
    is not perfectly parallel with the brain surface, and the FOVs are imaged beneath the brain's
    surface. To account for this, we use three points chosen from the reference image where the
    surface is in focus at different imaging depths. From these three points we can construct a
    plane to determine the difference between the optical axis and the brain surface. Using the
    difference in optical depth between this plane and the FOV we can adjust the FOV coordinates
    accordingly. This is done in the 'get_mlapdv_rel' method.

    After we have the adjusted MLAP coordinates (without depth), we can use them to obtain the
    final MLAPDV coordinates in the Allen atlas space. This is done in the 'mlapdv_from_rel' method
    by computing a triangulation of the brain's surface from the Allen atlas volume, then finding
    the surface location for each FOV pixel, then applying the adjusted depth.
    """

    cpu = 4  # Currently uses a lot of parallel loops in numba

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_session = kwargs.get('reference_session')  # an eid of the aligned histology session
        self.provenance = Provenance.HISTOLOGY if self.reference_session else Provenance.ESTIMATE

    @property
    def signature(self):
        I = dh.ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            I('mpciROIs.stackPos.npy', 'alf/FOV*', True),
                            # New additions  # FIXME should be self.device_collection (may require patching exp desc files)
                            I('referenceImage.stack.tif', 'raw_imaging_data_??/reference', True, unique=True),
                            I('referenceImage.meta.json', 'raw_imaging_data_??/reference', True, unique=True),
                            I('referenceImage.points.json', 'raw_imaging_data_??/reference', False, unique=True),
                            # ('referenceImage.mlapdv.npy', 'histology', False)  # may be in another session folder!
                            ],
            'output_files': [('mpciMeanImage.brainLocationIds.npy', 'alf/FOV_*', True),
                             ('mpciMeanImage.mlapdv.npy', 'alf/FOV_*', True),
                             ('mpciROIs.mlapdv.npy', 'alf/FOV_*', True),
                             ('mpciROIs.brainLocationIds.npy', 'alf/FOV_*', True),
                             ('_ibl_rawImagingData.meta.json', self.device_collection, True),
                             ('referenceImage.meta.json', 'raw_imaging_data_??/reference', True)]
        }
        # TODO This should be updated to handle changes in provenance suffix and device collection
        return signature

    def _run(self, *args, atlas_resolution=25, display=False):
        self.atlas = MRITorontoAtlas(res_um=atlas_resolution)  # TODO Check scaling appied to underlying volume
        # Load the reference stack & (down)load the registered MLAPDV coordinates
        reference_image = self.load_reference_stack()
        # Load main meta
        _, meta_files, _ = self.input_files[0].find_files(self.session_path)
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))
        if self.provenance is Provenance.HISTOLOGY:
            _logger.info('Extracting histology MLAPDV datasets')
            # Update the craniotomy center
            self.update_craniotomy_center(reference_image)
            meta['centerMM'] = reference_image['meta']['centerMM']
            with open(meta_files[0], 'w') as fp:
                json.dump(meta, fp)
            # Add reference meta data to meta_files list for registration
            meta_files.append(next(self.session_path.glob('raw_imaging_data_??/reference/referenceImage.meta.json')))
            # Interpolate the FOVs to the reference stack
            mlapdv = self.interpolate_FOVs(reference_image, meta)
        elif 'points' in reference_image['meta']:
            _logger.info('Extracting estimate MLAPDV datasets')
            mlapdv, _ = self.project_mlapdv(meta, atlas=self.atlas, provenance=self.provenance)
            # Convert to list of arrays for processing
            mlapdv = [mlapdv[i] for i in range(nFOV)]
        else:
            _logger.warning('No reference image points found; will not account for optical plane tilt')
            return self.super()._run(*args)

        # Account for optical plane tilt
        mlapdv_rel = self.correct_fov_depth_and_surface_projection(mlapdv, meta, reference_image)
        mean_image_mlapdv = self.project_mlapdv_from_surface(mlapdv_rel)

        # Because generating the projected coordinates takes so long, for now we will save them
        _logger.info('Saving mlapdv projection file to %s', self.session_path / 'mlapdv_projection.pkl')
        with open(self.session_path / 'mlapdv_projection.pkl', 'wb') as fp:
            pickle.dump((mlapdv_rel, mean_image_mlapdv), fp)

        # Look up brain location IDs from coordinates
        mean_image_ids = []
        for xyz in mean_image_mlapdv:
            labels = self.atlas.get_labels(xyz.reshape(-1, 3) / 1e6)  # in m
            mean_image_ids.append(labels.reshape(xyz.shape[:2]))

        # Update the FOV meta data fields (used in register_fov)
        for i, fov in enumerate(meta.get('FOV', [])):
            if 'MLAPDV' not in fov:
                fov['MLAPDV'] = {}
                fov['brainLocationIds'] = {}
            fov['MLAPDV'][self.provenance.name.lower()] = {
                'topLeft': mean_image_mlapdv[i][0, 0, :].tolist(),
                'topRight': mean_image_mlapdv[i][0, -1, :].tolist(),
                'bottomLeft': mean_image_mlapdv[i][-1, 0, :].tolist(),
                'bottomRight': mean_image_mlapdv[i][-1, -1, :].tolist(),
                'center': mean_image_mlapdv[i][round(mean_image_mlapdv[i].shape[0] / 2) - 1,
                                               round(mean_image_mlapdv[i].shape[1] / 2) - 1, :].tolist()
            }
            fov['brainLocationIds'][self.provenance.name.lower()] = {
                'topLeft': int(mean_image_ids[i][0, 0]),
                'topRight': int(mean_image_ids[i][0, -1]),
                'bottomLeft': int(mean_image_ids[i][-1, 0]),
                'bottomRight': int(mean_image_ids[i][-1, -1]),
                'center': int(mean_image_ids[i][round(mean_image_ids[i].shape[0] / 2) - 1,
                                                round(mean_image_ids[i].shape[1] / 2)])
            }

        # Save the mean image datasets
        suffix = None if self.provenance is Provenance.HISTOLOGY else self.provenance.name.lower()
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
        self.register_fov(meta, self.provenance)

        return sorted([*meta_files, *roi_files, *mean_image_files])

    def _get_atlas_registered_reference_mlap(self, reference_session_path, clobber=False):
        """Download the aligned reference stack Allen atlas indices.

        This is the file created by the histology pipeline, one per subject.
        This file contains the Allen atlas image volume indices for each pixel of the reference stack.

        Parameters
        ----------
        reference_session_path : pathlib.Path
            The session path of the reference session for this subject.
        clobber : bool
            If True, re-download the file even if it exists locally.

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
        # Ensure reference session reference files present
        signature = {'input_files': self.signature['input_files'][-3:], 'output_files': []}
        assert all(x.identifiers[-1].startswith('reference') for x in signature['input_files'])
        if self.location == 'server' and self.force:
            handler = dh.ServerGlobusDataHandler(reference_session_path, signature, one=self.one)
        else:
            handler = self.data_handler.__class__(reference_session_path, signature, one=self.one)
        handler.setUp()

        _logger.info('Looking for reference MLAPDV in %s', reference_session_path.joinpath(self.device_collection, 'reference'))
        # NB: The local reference folder is expected to exist after handler.setUp()
        local_file = next(reference_session_path.glob(f'{self.device_collection}/reference')) / 'referenceImage.mlapdv.npy'
        if clobber or not local_file.exists():
            # Download remote file
            assert self.one, 'ONE required'
            local_file.parent.mkdir(parents=True, exist_ok=True)
            lab = self.one.get_details(reference_session_path)['lab']
            remote_file = f'{lab}/{reference_session_path.session_path_short()}/{local_file.name}'
            try:
                # assert isinstance(self.data_handler, dh.ServerGlobusDataHandler)  # If not, assume Globus not configured
                handler = dh.ServerGlobusDataHandler(
                    reference_session_path, {'input_files': [], 'output_files': []}, one=self.one)
                endpoint_id = next(v['id'] for k, v in handler.globus.endpoints.items() if k.startswith('flatiron'))
                handler.globus.add_endpoint(endpoint_id, label='flatiron_histology', root_path='/histology/')
                handler.globus.mv('flatiron_histology', 'local', [remote_file], ['/'.join(local_file.parts[-5:])])
                assert local_file.exists(), f'Failed to download {remote_file} to {local_file}'
            except Exception as e:
                _logger.error(f'Failed to download via Globus: {e}')
                remote_file = f'{self.one.alyx._par.HTTP_DATA_SERVER}/histology/' + remote_file
                _logger.warning(f'Using HTTP download for {remote_file}')
                local_file = self.one.alyx.download_file(remote_file, target_dir=local_file.parent)
                assert local_file.exists(), f'Failed to download {remote_file} to {local_file}'
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
        stack, ref_meta = self.load_reference_stack()
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

    def _load_reference_stack_mlapdv(self, display=False):
        """Load the registered MLAPDV coordinates for the reference stack.

        Returns
        -------
        numpy.array
            A float array with shape (h, w, 3), comprising Allen atlas MLAPDV coordinates in μm.
            The first two dimensions (h, w) should equal those of the reference stack.

        """
        assert self.reference_session, 'Reference session eID not set'
        assert (reference_session_path := self.one.eid2path(self.reference_session)), \
            f'Reference session not found for eid {self.reference_session}'
        # Ensure reference session shares the same root dir as the task session path
        reference_session_path = self.session_path.parents[2] / reference_session_path.session_path_short()
        file = self._get_atlas_registered_reference_mlap(reference_session_path, clobber=False)
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

        if reference_session_path.session_parts != self.session_path.session_parts:
            # Apply transform
            save_path = next(self.session_path.glob('raw_imaging_data_??/reference')) / 'reference_stack_ecc_transform.gif'
            _, params = register_reference_stacks(
                self.session_path, reference_session_path, save_path=save_path, display=display, crop_size=None)
            transform_robust = (skimage.transform.EuclideanTransform(rotation=params['rotation']) +
                                skimage.transform.EuclideanTransform(translation=params['translation']))
            xyz = skimage.transform.warp(xyz, transform_robust, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            # Upload the saved image to Alyx as a note
            RegisterRawDataTask(self.session_path, one=self.one).upload_images(images=[save_path])

        if display:
            labels = ba.get_labels(xyz / 1e6)  # μm -> m
            acronyms = ba.regions.id2acronym(labels)

            # Generate a colour map
            L = np.arange(np.unique(acronyms).size)
            # plt.pcolor(X, Y, v, cmap=cm)
            # TODO This plots as x=AP, y=ML - should rotate first
            lab = np.fromiter(map(np.unique(acronyms).tolist().index, acronyms.flat), 'uint8').reshape(*labels.shape)
            ax = plt.imshow(lab, cmap='hsv')
            # Add discrete colorbar with acronyms as labels
            cbar = plt.colorbar(ax, ticks=L, orientation='vertical')
            cbar.ax.set_yticklabels(np.unique(acronyms)[L])
            cbar.ax.set_ylabel('Acronym')
            # cbar.ax.set_xlabel('Region')

            plt.show()

        return xyz  # reference_stack_mlapdv

    def update_craniotomy_center(self, reference_image):
        """Update subject JSON with atlas-aligned craniotomy coordinates."""
        assert not self.one.offline
        yx_res = np.array([
            reference_image['meta']['rawScanImageMeta']['YResolution'],
            reference_image['meta']['rawScanImageMeta']['XResolution']
        ])
        if reference_image['meta']['rawScanImageMeta']['ResolutionUnit'].casefold() == 'centimeter':
            # NB: these values are (y, x) in μm
            px_per_um = yx_res * 1e-4
            um_per_px = 1 / px_per_um
        else:
            raise NotImplementedError('Reference image resolution unit must be in centimeters')

        ref_stack_n_px = np.array(reference_image['mlapdv'].shape[:2])  # in (y, x)
        craniotomy_center_offset = np.flip(self.get_window_center(reference_image['meta']) * 1e3)  # (y, x) center offset mm -> μm

        image_center_px = ref_stack_n_px / 2
        # TODO Verify whether offset is added or subtracted
        #  empirically, it seems to be added looking at SP037/2023-02-20/001
        craniotomy_pixel = image_center_px + (craniotomy_center_offset / um_per_px)
        craniotomy_pixel = np.round(craniotomy_pixel).astype(int)  # convert to pixel coordinates
        _logger.debug('Craniotomy pixel coordinates: (%d, %d)', *craniotomy_pixel)

        # This doesn't work in python 3.10, numpy 2.24
        # craniotomy_resolved = referenceImage['mlapdv'][craniotomy_pixel] / 1e3  # py 3.11 # ML AP DV, μm -> mm
        craniotomy_resolved = reference_image['mlapdv'][craniotomy_pixel[0], craniotomy_pixel[1]] / 1e3

        # Update metadata
        reference_image['meta']['centerMM']['ML_resolved'] = craniotomy_resolved[0]
        reference_image['meta']['centerMM']['AP_resolved'] = craniotomy_resolved[1]
        meta_path = next(self.session_path.glob('raw_imaging_data_??/reference/referenceImage.meta.json'))
        with open(meta_path, 'w') as f:
            json.dump(reference_image['meta'], f)

        subject = self.session_path.subject
        subject_json = self.one.alyx.rest('subjects', 'read', id=subject)['json']
        # TODO Assert only one craniotomy key
        if sum(k.startswith('craniotomy_') for k in subject_json.keys()) > 1:
            raise NotImplementedError('Multiple craniotomies found')

        data = {'craniotomy_00': subject_json['craniotomy_00'].copy()}
        data['craniotomy_00']['center_resolved'] = np.round(craniotomy_resolved[:2], 3).tolist()
        _logger.info(
            'Craniotomy target: (%.2f, %.2f), actual: (%.2f, %.2f), difference: (%.2f, %.2f)',
            *subject_json['craniotomy_00']['center'], *data['craniotomy_00']['center_resolved'],
            *np.array(subject_json['craniotomy_00']['center']) - craniotomy_resolved[:2]
        )

        return self.one.alyx.json_field_update('subjects', subject, data=data)

    def interpolate_FOVs(self, reference_image, meta, display=False):
        """Interpolate the FOV coordinates from reference stack coordinates.

        Parameters
        ----------
        reference_image : dict
            The reference image object containing metadata and mlapdv histology data.
        meta : dict
            The FOV metadata.
        display : bool
            Whether to display the FOVs.

        Returns
        -------
        list of numpy.array
            The interpolated MLAPDV coordinates for each FOV.
        """
        # Extract the reference image and mean image extents in mm along the coverslip, relative to the craniotomy center
        assert np.all(self.get_window_center(reference_image['meta']) == self.get_window_center(meta))
        assert reference_image['meta']['scanImageParams']['objectiveResolution'] == meta['scanImageParams']['objectiveResolution']
        coordinates = self.get_fov_objective_extent(meta)

        # Reference image contains 3-D coordinates in m for each pixel of the reference image
        height, width = reference_image['mlapdv'].shape[:2]

        # The fields of view and reference image extents are in the same coordinate space (objective space in mm)
        # Create objective coordinates directly using linear transformation
        ref_extent = self.get_reference_image_extent(reference_image['meta'])
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
        values = reference_image['mlapdv'].reshape(-1, reference_image['mlapdv'].shape[-1])
        interp = NearestNDInterpolator(points, values)

        # Sanity check: center of the window
        centre = ((r_left + r_right) / 2, (r_top + r_bottom) / 2)
        # Test the interpolation with the exact center point from our grid
        center_flat_idx = np.ravel_multi_index((height // 2, width // 2), (height, width))
        center_point_from_grid = points[center_flat_idx]
        center_mlapdv = interp(center_point_from_grid)
        expected = reference_image['mlapdv'][height // 2, width // 2]
        assert np.allclose(center_mlapdv, expected), f'Expected {expected}, got {center_mlapdv} at centre={centre}'

        if display:
            # For sanity, plot a rectangle of the reference image window extent, then plot each pixel of each FOV
            _, ax = plt.subplots()
        else:
            ax = None

        # Interpolate FOV coordinates from reference mlapdv coordinates
        mlapdv = []
        for i, (fov, fov_meta) in tqdm(enumerate(zip(coordinates, meta['FOV']))):
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

        Now we have corrected ml ap coordinates of cells in the imaging plane we first need to
        determine where those cells are on the surface of the atlas and from that point, move down
        along the local brain normal by the true DV depth of the cell.

        We project onto the atlas either along the brain normal of the atlas.

        Parameters
        ----------
        reference_image : dict
            A referenceImage object with keys ('meta', 'points').

        Returns
        -------
        p_ref : np.ndarray
            The point on the plane in μm.
        n_ref : np.ndarray
            The normal vector of the plane.
        dv_avg : float
            The average depth value of the surface points in μm.

        """
        points = reference_image['meta']['points']
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
            # ref_points_mlap = reference_image['mlapdv'][*np.hsplit(ref_points_px, 2)].squeeze()  # py 3.11
            a, b = np.hsplit(ref_points_px, 2)
            ref_points_mlap = reference_image['mlapdv'][a, b].squeeze()
        else:
            raise NotImplementedError
            # ref_points_mlap = cs2d.transform(ref_points_rel, 'image', 'mlap')

        # replace the resolved DV with optical plane depth
        stack_dv_m = (stack_dv[:, np.newaxis] - dv_avg)  # μm
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
            z = -1 * (fov_meta['Zs'] - dv_avg)  # depth below reference plane (μm), positive = deeper
            _logger.info(f"FOV {i}: Original Zs={fov_meta['Zs']:.1f}μm, dv_avg={dv_avg:.1f}μm, converted depth z={z:.1f}μm")

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
            mlap_points = fov_flat[:, :2]
            desc = f'Projecting MLAPDV points {n+1}/{len(mlapdv_rel)}'
            for i, point in tqdm(enumerate(mlap_points), total=len(mlap_points), desc=desc):
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

    def load_reference_stack(self):
        """Load the referenceImage.stack.tif file and its metadata.

        Loads the files:
        - referenceImage.stack.tif
        - referenceImage.meta.json
        - referenceImage.points.json
        - referenceImage.mlapdv.npy

        referenceImage.points may be present in the meta data file.
        referenceImage.mlapdv is loaded if reference_session is not None.

        Returns
        -------
        iblutil.util.Bunch
            The reference image object with keys ('stack', 'meta').
            The stack is an array of size (nZ, nY, nX).
        """
        try:
            stack_path = next(self.session_path.glob('raw_imaging_data_??/reference/referenceImage.stack.tif'))
        except StopIteration:
            raise FileNotFoundError('Reference stack not found')
        meta_path = stack_path.with_name('referenceImage.meta.json')
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_path) or {})
        reference_image = {'stack': skimage.io.imread(stack_path), 'meta': meta}
        if stack_path.with_name('referenceImage.points.json').exists():
            points_path = stack_path.with_name('referenceImage.points.json')
            # Copy to meta data
            meta.update(alfio.load_file_content(points_path))
            meta['stack_idx_range'] = meta.pop('range')  # rename for clarity
        if self.reference_session:
            # Load the mlapdv coordinates for the reference stack
            reference_image['mlapdv'] = self._load_reference_stack_mlapdv(display=False)
            assert reference_image['stack'].shape[1:] == reference_image['mlapdv'].shape[:2], \
                'Reference stack and MLAPDV coordinates must have the same shape'
        return Bunch(reference_image)

"""
Histology: probe tracks traced in lasagna, conversion to IBL coordinates and registration to Alyx.

Spaces
------
- ARA: brainregister's 25 um Allen CCF grid (``coronal_rc_25``, ``sample2ARA/STD_*.tif``), 456 x 320 x 528 voxels;
  column, row, slice = ML, DV (ventral +), AP (anterior +).
- sample: an individual brain's downsampled image (``downsampledStacks_25/ds_*.tif``), 25 um voxels, size varies;
  axes only approximately ML, DV, AP, so points go through the sample2ARA registration (ARA -> sample directly,
  sample -> ARA by numerical inversion).

Coordinate conventions, (n, 3) arrays named by anatomical column order
----------------------------------------------------------------------
=========== ======================================= ============ ======== ===============
name        what                                    columns      units    AP increases
=========== ======================================= ============ ======== ===============
apmldv      lasagna csv (Z,X,Y), either space       AP, ML, DV   voxels   anteriorly
mldvap      elastix (x, y, z), either space         ML, DV, AP   voxels   anteriorly
ixiyiz      IBL atlas indices (brain_atlas.bc)      ML, AP, DV   voxels   posteriorly
xyz         IBL coordinates from bregma, ARA only   ML, AP, DV   m        anteriorly
=========== ======================================= ============ ======== ===============

Voxel indices are fractional. Images read with tifffile are indexed [slice, row, column] = mldvap[:, ::-1].

Functions
---------
=========================== ============================== ====================================
function                    input                          output
=========================== ============================== ====================================
``read_lasagna_csv``        csv file                       ``apmldv`` (either space)
``write_lasagna_csv``       ``apmldv`` (either space)      csv file
``lasagna_to_ibl``          ``apmldv`` ARA                 ``xyz``
``ibl_to_lasagna``          ``xyz``                        ``apmldv`` ARA
``lasagna_sample_to_ara``   ``apmldv`` sample              ``apmldv`` ARA (inverse transform)
``lasagna_ara_to_sample``   ``apmldv`` ARA                 ``apmldv`` sample (direct transform)
``ibl_sample_to_ara``       ``mldvap`` sample              ``xyz`` (inverse transform)
``ibl_ara_to_sample``       ``xyz``                        ``mldvap`` sample (direct transform)
``transform_lasagna_csv``   csv file (either space)        csv file (other space)
``load_track_csv``          csv file (ARA, or sample with  ``xyz``, sorted dorsal to ventral
                            ``transform_dir``)
=========================== ============================== ====================================
"""

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from one.api import ONE
import one.alf.io as alfio

from neuropixel import TIP_SIZE_UM, trace_header
import spikeglx
import iblatlas.atlas as atlas
from ibllib.ephys.spikes import probes_description as extract_probes
from ibllib.qc import base

from ibldsp.utils import fcn_cosine


_logger = logging.getLogger(__name__)


def load_track_csv(file_track, brain_atlas=None, transform_dir=None):
    """
    Loads a lasagna track and convert to IBL-Allen coordinate framework.

    :param file_track: lasagna points csv (rows of Z,X,Y voxels = (slice, column, row) ~ (AP, ML, DV))
    :param brain_atlas: iblatlas.atlas.AllenAtlas at 25 um (default AllenAtlas(25))
    :param transform_dir: None (default) if the track was traced in ARA space (sample2ARA/STD_*.tif).
        If the track was traced in the brainregister sample space (downsampledStacks_25/ds_*.tif), the directory
        containing the sample2ARA affine and bspline elastix parameter files: the points are first transformed to
        ARA space with the inverse of the sample2ARA registration.
    :return: (n, 3) xyz IBL coordinates (m), (ML, AP, DV), sorted along the dorso-ventral axis
    """
    file_track = Path(file_track)
    if file_track.stat().st_size == 0:
        return np.array([])
    brain_atlas = _check_atlas_25(brain_atlas)
    apmldv = read_lasagna_csv(file_track)
    if transform_dir is not None:
        apmldv = lasagna_sample_to_ara(apmldv, transform_dir)
    apmldv = apmldv[np.argsort(apmldv[:, 2]), :]  # sort by row (DV), dorsal first
    return lasagna_to_ibl(apmldv, brain_atlas)


# -------------------------------------------------------------------------------------------------
# Lasagna <-> IBL coordinates and brainregister (elastix) sample <-> ARA transforms
# See the module docstring for the spaces and coordinate conventions (apmldv, mldvap, ixiyiz, xyz).
# -------------------------------------------------------------------------------------------------
_ARA_SHAPE_MLDVAP = (456, 320, 528)  # (columns, rows, slices) = (ML, DV, AP)
# brainregister parameter file names: current (brainregister-affine_to-target.txt, brainregister-bspline_to-target.txt)
# and older outputs (TransformParameters.0.txt, TransformParameters.1.txt)
_TRANSFORM_FILE_PATTERNS = ('brainregister-*.txt', 'TransformParameters*.txt')


def _check_atlas_25(brain_atlas):
    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    nxyz = tuple(int(n) for n in brain_atlas.bc.nxyz)  # IBL atlas order (ML, AP, DV)
    if nxyz != (_ARA_SHAPE_MLDVAP[0], _ARA_SHAPE_MLDVAP[2], _ARA_SHAPE_MLDVAP[1]):  # (columns, slices, rows)
        raise ValueError(f'A 25 um atlas is required (brainregister ARA space), got volume shape {nxyz}')
    return brain_atlas


def lasagna_to_ibl(apmldv, brain_atlas=None):
    """
    Convert lasagna points in ARA space to IBL coordinates.

    Equivalent to iblatlas xyz2ccf in 25 um voxels, with the AP axis reversed (brainregister ARA volume
    is stored posterior to anterior).

    :param apmldv: (n, 3) lasagna Z,X,Y points (slice, column, row) = (AP, ML, DV) in 25 um ARA voxels
        (e.g. traced on sample2ARA/STD_*.tif)
    :param brain_atlas: iblatlas.atlas.AllenAtlas at 25 um (default AllenAtlas(25))
    :return: (n, 3) xyz IBL coordinates (m), (ML, AP, DV)
    """
    brain_atlas = _check_atlas_25(brain_atlas)
    # (slice, column, row) = (AP, ML, DV) -> (column, slice, row) = (ML, AP, DV)
    ixiyiz = np.atleast_2d(np.asarray(apmldv, dtype=float))[:, [1, 0, 2]]
    # the ARA slice index runs anterior-wards, the IBL atlas AP index posterior-wards
    ixiyiz[:, 1] = _ARA_SHAPE_MLDVAP[2] - 1 - ixiyiz[:, 1]
    return brain_atlas.bc.i2xyz(ixiyiz)


def ibl_to_lasagna(xyz, brain_atlas=None):
    """
    Convert IBL coordinates to lasagna points in ARA space (inverse of lasagna_to_ibl).

    :param xyz: (n, 3) IBL coordinates (m), (ML, AP, DV), e.g. Alyx xyz_picks / 1e6
    :param brain_atlas: iblatlas.atlas.AllenAtlas at 25 um (default AllenAtlas(25))
    :return: (n, 3) lasagna Z,X,Y points (slice, column, row) = (AP, ML, DV) in 25 um ARA voxels (not rounded)
    """
    brain_atlas = _check_atlas_25(brain_atlas)
    # IBL atlas indices (ML, AP, DV)
    ixiyiz = brain_atlas.bc.xyz2i(np.atleast_2d(np.asarray(xyz, dtype=float)), round=False, mode='wrap')
    # the IBL atlas AP index runs posterior-wards, the ARA slice index anterior-wards
    ixiyiz[:, 1] = _ARA_SHAPE_MLDVAP[2] - 1 - ixiyiz[:, 1]
    # (ML, AP, DV) = (column, slice, row) -> (slice, column, row)
    return ixiyiz[:, [1, 0, 2]]


def _read_elastix_parameter_file(file):
    """Parse an elastix parameter file into a dict {key: list of str}."""
    pm = {}
    with open(file, 'r', errors='replace') as fid:
        for line in fid:
            line = line.split('//')[0].strip()
            if line.startswith('(') and line.endswith(')'):
                key, _, value = line[1:-1].partition(' ')
                pm[key] = value.replace('"', '').split()
    return pm


def _pm_floats(pm, key):
    return np.array(pm[key], dtype=float)


class _ElastixAffine:
    """elastix AffineTransform: T(x) = A (x - c) + c + t"""

    def __init__(self, pm):
        p = _pm_floats(pm, 'TransformParameters')
        self.A = p[:9].reshape(3, 3)
        self.t = p[9:]
        self.c = _pm_floats(pm, 'CenterOfRotationPoint')

    def __call__(self, x):
        return (x - self.c) @ self.A.T + self.c + self.t

    def inverse(self, y):
        return np.linalg.solve(self.A, (y - self.c - self.t).T).T + self.c


class _ElastixBSpline:
    """
    elastix cubic BSplineTransform: T(x) = x + d(x), with d the cubic B-spline defined by the control point
    coefficients (TransformParameters, one grid per component, column index varying fastest) on the control
    point grid. Points and grid parameters are (column, row, slice).
    As in ITK the displacement is zero where the support of the spline is not fully inside the grid.
    """

    def __init__(self, pm):
        if int(pm.get('BSplineTransformSplineOrder', ['3'])[0]) != 3:
            raise NotImplementedError('Only cubic B-spline transforms are supported')
        self.grid_size = _pm_floats(pm, 'GridSize').astype(int)
        self.grid_origin = _pm_floats(pm, 'GridOrigin')
        self.grid_spacing = _pm_floats(pm, 'GridSpacing')
        # stored with the column index varying fastest: reshapes to [slice, row, column] per component,
        # transposed to [column, row, slice] to be indexed with (column, row, slice) points
        coeffs = _pm_floats(pm, 'TransformParameters').reshape(3, *self.grid_size[::-1])
        self.coeffs = [c.transpose(2, 1, 0) for c in coeffs]

    def __call__(self, x):
        u = (x - self.grid_origin) / self.grid_spacing
        valid = np.all((u >= 1) & (u < self.grid_size - 2), axis=1)
        d = np.zeros_like(x)
        if np.any(valid):
            for k in range(3):
                # with prefilter=False the input array is used directly as the B-spline coefficients
                d[valid, k] = map_coordinates(self.coeffs[k], u[valid].T, order=3, prefilter=False, mode='nearest')
        return x + d


class _Sample2AraTransform:
    """
    brainregister sample2ARA registration (elastix affine followed by bspline, fixed = ARA, moving = sample).
    Calling the transform maps ARA voxels to sample voxels; the inverse is solved numerically.
    Points are in elastix order (mldvap): voxel indices (column, row, slice), = (ML, DV, AP) in ARA space and
    ~ (ML, DV, AP) in sample space. NB: not the IBL (ML, AP, DV) order.
    """

    def __init__(self, transform_dir):
        transform_dir = Path(transform_dir)
        if not transform_dir.is_dir():
            raise FileNotFoundError(f'Transform directory {transform_dir} not found')
        files = sorted({f for pattern in _TRANSFORM_FILE_PATTERNS for f in transform_dir.glob(pattern)})
        found = {}
        for file in files:
            pm = _read_elastix_parameter_file(file)
            if 'Transform' in pm:
                found.setdefault(pm['Transform'][0], []).append((file, pm))
        if sorted(found) != ['AffineTransform', 'BSplineTransform'] or any(len(v) != 1 for v in found.values()):
            files = {k: [f.name for f, _ in v] for k, v in found.items()}
            raise ValueError(
                f'{transform_dir} must contain exactly one affine and one bspline elastix parameter file '
                f'named {" or ".join(_TRANSFORM_FILE_PATTERNS)}, found {files}'
            )
        self.transforms = []
        for kind, cls in [('AffineTransform', _ElastixAffine), ('BSplineTransform', _ElastixBSpline)]:
            file, pm = found[kind][0]
            self._check_parameters(file, pm)
            self.transforms.append(cls(pm))

    @staticmethod
    def _check_parameters(file, pm):
        size = tuple(int(float(s)) for s in pm.get('Size', []))
        if size != _ARA_SHAPE_MLDVAP:
            raise ValueError(
                f'{file}: fixed image size {size} is not the 25 um ARA volume {_ARA_SHAPE_MLDVAP}. '
                'Is this a sample2ARA transform directory (not ARA2sample)?'
            )
        if pm.get('HowToCombineTransforms', ['Compose'])[0] != 'Compose':
            raise NotImplementedError(f'{file}: only "Compose" is supported for HowToCombineTransforms')
        for key, expected in [
            ('Spacing', np.ones(3)),
            ('Origin', np.zeros(3)),
            ('Direction', np.eye(3).ravel()),
            ('GridDirection', np.eye(3).ravel()),
        ]:
            if key in pm and not np.allclose(_pm_floats(pm, key), expected):
                raise NotImplementedError(f'{file}: {key} must be {expected}')

    def __call__(self, mldvap):
        x = np.atleast_2d(np.asarray(mldvap, dtype=float))
        for t in self.transforms:  # elastix Compose: T(x) = T_bspline(T_affine(x))
            x = t(x)
        return x

    def _jacobian(self, x, h=0.5):
        jac = np.zeros((x.shape[0], 3, 3))
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = h
            jac[:, :, j] = (self(x + dx) - self(x - dx)) / (2 * h)
        return jac

    def inverse(self, mldvap, tol=1e-6, max_iter=100):
        """Solve T(x) = mldvap for x with Newton iterations, starting from the inverse of the affine part."""
        y = np.atleast_2d(np.asarray(mldvap, dtype=float))
        x = self.transforms[0].inverse(y)
        todo = np.ones(y.shape[0], dtype=bool)
        for _ in range(max_iter):
            idx = np.flatnonzero(todo)
            r = y[idx] - self(x[idx])
            converged = np.linalg.norm(r, axis=1) < tol
            todo[idx[converged]] = False
            if not np.any(todo):
                break
            idx, r = idx[~converged], r[~converged]
            x[idx] += np.linalg.solve(self._jacobian(x[idx]), r[..., np.newaxis])[..., 0]
        residual = np.linalg.norm(y - self(x), axis=1)
        if np.any(residual > tol):
            raise RuntimeError(
                f'Inverse transform did not converge for {np.sum(residual > tol)} point(s) '
                f'(max residual {residual.max():.3g} voxels)'
            )
        return x


def _warn_outside_ara(mldvap):
    outside = np.any((mldvap < 0) | (mldvap > np.array(_ARA_SHAPE_MLDVAP) - 1), axis=1)
    if np.any(outside):
        _logger.warning(f'{np.sum(outside)} / {outside.size} point(s) fall outside the ARA volume')


def _sample_to_ara_elastix(mldvap, transform_dir):
    ara = _Sample2AraTransform(transform_dir).inverse(mldvap)
    _warn_outside_ara(ara)
    return ara


def _ara_to_sample_elastix(mldvap, transform_dir):
    mldvap = np.atleast_2d(np.asarray(mldvap, dtype=float))
    _warn_outside_ara(mldvap)
    return _Sample2AraTransform(transform_dir)(mldvap)


def _lasagna_to_elastix(apmldv):
    # (slice, column, row) -> (column, row, slice), i.e. (AP, ML, DV) -> (ML, DV, AP)
    return np.atleast_2d(np.asarray(apmldv, dtype=float))[:, [1, 2, 0]]


def _elastix_to_lasagna(mldvap):
    # (column, row, slice) -> (slice, column, row), i.e. (ML, DV, AP) -> (AP, ML, DV)
    return np.atleast_2d(np.asarray(mldvap, dtype=float))[:, [2, 0, 1]]


def lasagna_sample_to_ara(apmldv, transform_dir):
    """
    Transform lasagna points traced in the brainregister sample space to ARA space, using the inverse of the
    sample2ARA registration (consistent with the sample2ARA/STD_*.tif volumes).

    :param apmldv: (n, 3) lasagna Z,X,Y points (slice, column, row) ~ (AP, ML, DV) in sample voxels
        (traced on downsampledStacks_25/ds_*.tif)
    :param transform_dir: directory containing the sample2ARA affine and bspline elastix parameter files
    :return: (n, 3) lasagna Z,X,Y points (slice, column, row) = (AP, ML, DV) in 25 um ARA voxels (not rounded)
    """
    return _elastix_to_lasagna(_sample_to_ara_elastix(_lasagna_to_elastix(apmldv), transform_dir))


def lasagna_ara_to_sample(apmldv, transform_dir):
    """
    Transform lasagna points in ARA space to the brainregister sample space, applying the sample2ARA
    registration directly.

    :param apmldv: (n, 3) lasagna Z,X,Y points (slice, column, row) = (AP, ML, DV) in 25 um ARA voxels
        (e.g. traced on sample2ARA/STD_*.tif)
    :param transform_dir: directory containing the sample2ARA affine and bspline elastix parameter files
    :return: (n, 3) lasagna Z,X,Y points (slice, column, row) ~ (AP, ML, DV) in sample voxels (not rounded)
    """
    return _elastix_to_lasagna(_ara_to_sample_elastix(_lasagna_to_elastix(apmldv), transform_dir))


def ibl_sample_to_ara(mldvap, transform_dir, brain_atlas=None):
    """
    Transform points in the brainregister sample space to IBL coordinates, using the inverse of the
    sample2ARA registration.

    :param mldvap: (n, 3) sample voxel indices in elastix order (column, row, slice) ~ (ML, DV, AP) of
        downsampledStacks_25/ds_*.tif (NB: not the IBL (ML, AP, DV) order)
    :param transform_dir: directory containing the sample2ARA affine and bspline elastix parameter files
    :param brain_atlas: iblatlas.atlas.AllenAtlas at 25 um (default AllenAtlas(25))
    :return: (n, 3) xyz IBL coordinates (m), (ML, AP, DV)
    """
    brain_atlas = _check_atlas_25(brain_atlas)  # check before running the transform
    return lasagna_to_ibl(_elastix_to_lasagna(_sample_to_ara_elastix(mldvap, transform_dir)), brain_atlas)


def ibl_ara_to_sample(xyz, transform_dir, brain_atlas=None):
    """
    Transform IBL coordinates to the brainregister sample space, applying the sample2ARA registration directly.

    :param xyz: (n, 3) IBL coordinates (m), (ML, AP, DV), e.g. Alyx xyz_picks / 1e6
    :param transform_dir: directory containing the sample2ARA affine and bspline elastix parameter files
    :param brain_atlas: iblatlas.atlas.AllenAtlas at 25 um (default AllenAtlas(25))
    :return: (n, 3) sample voxel indices in elastix order (column, row, slice) ~ (ML, DV, AP) of
        downsampledStacks_25/ds_*.tif (NB: not the IBL (ML, AP, DV) order). Index an image read with tifffile
        (indexed [slice, row, column]) with mldvap[:, ::-1].
    """
    return _ara_to_sample_elastix(_lasagna_to_elastix(ibl_to_lasagna(xyz, brain_atlas)), transform_dir)


def transform_lasagna_csv(file_in, file_out, transform_dir, direction='sample_to_ara'):
    """
    Read a lasagna points csv file, transform the points between sample and ARA space with the sample2ARA
    registration and write them to a lasagna points csv file.

    :param file_in: lasagna points csv (rows of Z,X,Y voxels = (slice, column, row) ~ (AP, ML, DV))
    :param file_out: output lasagna points csv
    :param transform_dir: directory containing the sample2ARA affine and bspline elastix parameter files
    :param direction: 'sample_to_ara' or 'ara_to_sample'
    :return: (n, 3) transformed lasagna Z,X,Y points
    """
    funcs = {'sample_to_ara': lasagna_sample_to_ara, 'ara_to_sample': lasagna_ara_to_sample}
    if direction not in funcs:
        raise ValueError(f'direction must be one of {list(funcs)}, got {direction}')
    apmldv = funcs[direction](read_lasagna_csv(file_in), transform_dir)
    write_lasagna_csv(file_out, apmldv)
    return apmldv


def read_lasagna_csv(file):
    """
    Read a lasagna points csv file.

    :param file: lasagna points csv, one point per row as Z,X,Y voxels (slice, column, row) ~ (AP, ML, DV)
    :return: (n, 3) float array of Z,X,Y points; (0, 3) for an empty file
    """
    file = Path(file)
    if file.stat().st_size == 0:
        return np.empty((0, 3))
    apmldv = np.atleast_2d(np.loadtxt(file, delimiter=',', dtype=float))
    if apmldv.shape[1] != 3:
        raise ValueError(f'{file}: expected 3 values (Z,X,Y) per row, got {apmldv.shape[1]}')
    return apmldv


def write_lasagna_csv(file, apmldv, decimals=3):
    """
    Write points to a lasagna points csv file (lasagna rounds the points to whole voxels when loading them).

    :param file: output csv file
    :param apmldv: (n, 3) Z,X,Y points (slice, column, row) ~ (AP, ML, DV)
    :param decimals: number of decimals kept in the file
    """
    apmldv = np.atleast_2d(np.asarray(apmldv, dtype=float))
    if apmldv.shape[1] != 3:
        raise ValueError(f'expected (n, 3) Z,X,Y points, got shape {apmldv.shape}')
    np.savetxt(file, np.round(apmldv, decimals) + 0.0, delimiter=',', fmt='%g')  # + 0.0 avoids writing -0


def get_picked_tracks(histology_path, glob_pattern='*_pts_transformed.csv', brain_atlas=None):
    """
    This outputs reads in the Lasagna output and converts the picked tracks in the IBL coordinates
    :param histology_path: Path object: folder path containing tracks
    :return: xyz coordinates in
    """
    brain_atlas = brain_atlas or atlas.AllenAtlas()
    xyzs = []
    histology_path = Path(histology_path)
    if histology_path.is_file():
        files_track = [histology_path]
    else:
        files_track = list(histology_path.rglob(glob_pattern))
    for file_track in files_track:
        xyzs.append(load_track_csv(file_track, brain_atlas=brain_atlas))
    return {'files': files_track, 'xyz': xyzs}


def get_micro_manipulator_data(subject, one=None, force_extract=False):
    """
    Looks for all ephys sessions for a given subject and get the probe micro-manipulator
    trajectories.
    If probes ALF object not on flat-iron, attempts to perform the extraction from meta-data
    and task settings file.
    """
    if not one:
        one = ONE(cache_rest=None)

    eids, sessions = one.search(subject=subject, task_protocol='ephys', details=True)
    probes = alfio.AlfBunch({})
    for ses in sessions:
        sess_path = Path(ses['local_path'])
        probe = None
        if not force_extract:
            probe = one.load_object(ses['url'], 'probes')
        if not probe:
            _logger.warning(f'Re-extraction probe info for {sess_path}')
            dtypes = ['_iblrig_taskSettings.raw', 'ephysData.raw.meta']
            raw_files = one.load(ses['url'], dataset_types=dtypes, download_only=True)
            if all([rf is None for rf in raw_files]):
                _logger.warning(f'no raw settings files nor ephys data found for {ses["local_path"]}. Skip this session.')
                continue
            extract_probes(sess_path, bin_exists=False)
            probe = alfio.load_object(sess_path.joinpath('alf'), 'probes')
        one.load(ses['url'], dataset_types='channels.localCoordinates', download_only=True)
        # get for each insertion the sites local mapping: if not found assumes checkerboard pattern
        probe['sites_coordinates'] = []
        for prb in probe.description:
            chfile = Path(ses['local_path']).joinpath('alf', prb['label'], 'channels.localCoordinates.npy')
            if chfile.exists():
                probe['sites_coordinates'].append(np.load(chfile))
            else:
                _logger.warning(f'no channel.localCoordinates found for {ses["local_path"]}.Assumes checkerboard pattern')
                th = trace_header(version=1)
                probe['sites_coordinates'].append(np.c_[th['x'], th['y']])
        # put the session information in there
        probe['session'] = [ses] * len(probe.description)
        probes = probes.append(probe)
    return probes


def plot2d_all(trajectories, tracks, brain_atlas=None):
    """
    Plot all tracks on a single 2d slice
    :param trajectories: dictionary output of the Alyx REST query on trajectories
    :param tracks:
    :return:
    """
    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    plt.figure()
    axs = brain_atlas.plot_sslice(brain_atlas.bc.i2x(190), cmap=plt.get_cmap('bone'))
    plt.figure()
    axc = brain_atlas.plot_cslice(brain_atlas.bc.i2y(350))
    for xyz in tracks['xyz']:
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e6, 'b')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e6, 'b')
    for trj in trajectories:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brain_atlas)
        xyz = ins.xyz
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e6, 'r')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e6, 'r')


def plot3d_all(trajectories, tracks, brain_atlas=None):
    """
    Plot all tracks on a single 2d slice
    :param trajectories: dictionary output of the Alyx REST query on trajectories
    :param tracks:
    :return:
    """
    from mayavi import mlab

    brain_atlas = brain_atlas or atlas.AllenAtlas()
    src = mlab.pipeline.scalar_field(brain_atlas.label)
    mlab.pipeline.iso_surface(
        src,
        contours=[
            0.5,
        ],
        opacity=0.3,
    )

    pts = []
    for xyz in tracks['xyz']:
        mlapdv = brain_atlas.bc.xyz2i(xyz)
        pts.append(mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2], line_width=3))

    plt_trj = []
    for trj in trajectories:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brain_atlas)
        mlapdv = brain_atlas.bc.xyz2i(ins.xyz)
        plt = mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2], line_width=3, color=(1.0, 0.0, 1.0))
        plt_trj.append(plt)


def interpolate_along_track(xyz_track, depths):
    """
    Get the coordinates of points along a track according to their distances from the first
    point.
    :param xyz_track: np.array [npoints, 3]. Usually the first point is the deepest
    :param depths: distance from the first point of the track, usually the convention is the
    deepest point is 0 and going up
    :return: xyz_channels
    """
    # from scipy.interpolate import interp1d
    # this is the cumulative distance from the lowest picked point (first)
    distance = np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz_track, axis=0) ** 2, axis=1))])
    xyz_channels = np.zeros((depths.shape[0], 3))
    for m in np.arange(3):
        xyz_channels[:, m] = np.interp(depths, distance, xyz_track[:, m])
        # xyz_channels[:, m] = interp1d(distance, xyz[:, m], kind='cubic')(chdepths / 1e6)
    # plt.figure()
    # plt.plot(xyz_track[:, 0] * 1e6, xyz_track[:, 2] * 1e6, 'k*'), plt.axis('equal')
    # plt.plot(xyz_channels[:, 0] * 1e6, xyz_channels[:, 2] * 1e6, '.'), plt.axis('equal')
    return xyz_channels


def get_brain_regions(xyz, channels_positions=None, brain_atlas=None):
    """
    :param xyz: numpy array of 3D coordinates corresponding to a picked track or a trajectory
    the deepest point is assumed to be the tip.
    :param channels_positions:
    :param brain_atlas:
    :return: brain_regions (associated to each channel),
             insertion (object atlas.Insertion, defining 2 points of entries
             (tip and end of probe))
    """
    """
    this is the depth along the probe (from the first point which is the deepest labeled point)
    Due to the blockiness, depths may not be unique along the track so it has to be prepared
    """

    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    if channels_positions is None:
        geometry = trace_header(version=1)
        channels_positions = np.c_[geometry['x'], geometry['y']]

    xyz = xyz[np.argsort(xyz[:, 2]), :]
    d = atlas.cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
    indsort = np.argsort(d)
    xyz = xyz[indsort, :]
    d = d[indsort]
    iduplicates = np.where(np.diff(d) == 0)[0]
    xyz = np.delete(xyz, iduplicates, axis=0)
    d = np.delete(d, iduplicates, axis=0)

    assert np.all(np.diff(d) > 0), 'Depths should be strictly increasing'

    # Get the probe insertion from the coordinates
    insertion = atlas.Insertion.from_track(xyz, brain_atlas)

    # Interpolate channel positions along the probe depth and get brain locations
    TIP_SIZE_UM = 200
    xyz_channels = interpolate_along_track(xyz, (channels_positions[:, 1] + TIP_SIZE_UM) / 1e6)

    # get the brain regions
    brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
    brain_regions['xyz'] = xyz_channels
    brain_regions['lateral'] = channels_positions[:, 0]
    brain_regions['axial'] = channels_positions[:, 1]
    assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1

    return brain_regions, insertion


def register_chronic_track(chronic_id, picks=None, one=None, overwrite=False, channels=True, brain_atlas=None):
    """
    Register the user picks to a chronic insertion in Alyx.
    Here we update the database in 4 steps
    1) The user picks converted to IBL coordinates will be stored in the json field of the
    corresponding chronic insertion models
    2) All associated probe insertions are identified and the user picks stored in the json field too
    2) The trajectory associated to the chronic insertion computed from the histology track is created or patched
    3) Channel locations are set in the table
    :param chronic_id:
    :param picks:
    :param one:
    :param overwrite:
    :param channels:
    :param brain_atlas:
    :return:
    """
    assert one
    chronic = one.alyx.rest('chronic-insertions', 'list', id=chronic_id)[0]
    for probe_id in chronic['probe_insertion']:
        pid = probe_id['id']
        brain_locations, insertion_histology = register_track(
            pid, picks=picks, one=one, overwrite=overwrite, channels=channels, brain_atlas=brain_atlas
        )

    if picks is None or picks.size == 0:
        hist_qc = base.QC(chronic_id, one=one, endpoint='chronic-insertions')
        hist_qc.update_extended_qc({'tracing_exists': False})
        hist_qc.update('CRITICAL', namespace='tracing')
    else:
        one.alyx.json_field_update(
            endpoint='chronic-insertions', uuid=chronic_id, field_name='json', data={'xyz_picks': np.int32(picks * 1e6).tolist()}
        )
        # Update the insertion qc to register tracing exits
        hist_qc = base.QC(chronic_id, one=one, endpoint='chronic-insertions')
        hist_qc.update_extended_qc({'tracing_exists': True})

    return brain_locations, insertion_histology


def _load_channel_coordinates_from_meta(eid, pname, one):
    """
    Load the channel coordinates from the raw spikeglx ap.meta file of a probe insertion.

    :param eid: session id
    :param pname: probe name
    :param one: ONE instance
    :return: (n_channels, 2) array of lateral, axial positions in um, or None if not found
    """
    dsets = one.list_datasets(eid, '*.ap.meta', collection=f'raw_ephys_data/{pname}')
    if len(dsets) != 1:
        _logger.debug(f'Expected 1 ap.meta file in raw_ephys_data/{pname} for session {eid}, found {len(dsets)}')
        return None
    meta_file = one.load_dataset(eid, dsets[0], download_only=True)
    md = spikeglx.read_meta_data(meta_file)
    # geometry_from_meta falls back to a default geometry if the channel map is missing, we don't want this
    if not any(k in md for k in ['snsShankMap', 'snsGeomMap']):
        _logger.debug(f'No channel map (snsShankMap/snsGeomMap) in {meta_file}')
        return None
    geom = spikeglx.geometry_from_meta(md)
    return np.c_[geom['x'], geom['y']].astype(float)


def _load_channel_coordinates_from_sorting(eid, pname, one):
    """
    Load the channel coordinates (channels.localCoordinates) of the spike sorting of a probe insertion.
    The iblsorter output is preferred over pykilosort.

    :param eid: session id
    :param pname: probe name
    :param one: ONE instance
    :return: (n_channels, 2) array of lateral, axial positions in um, or None if no spike sorting is found
    """
    for sorter in ['iblsorter', 'pykilosort']:
        collection = f'alf/{pname}/{sorter}'
        if len(one.list_datasets(eid, 'channels.localCoordinates.npy', collection=collection)):
            return one.load_dataset(eid, 'channels.localCoordinates.npy', collection=collection)
    return None


def _load_channel_coordinates(probe_id, one):
    """
    Load the channel coordinates of a probe insertion. These are read from the raw spikeglx ap.meta file and
    if that fails from the channels.localCoordinates of the spike sorting.

    :param probe_id: probe insertion id
    :param one: ONE instance
    :return: (n_channels, 2) array of lateral, axial positions in um, or None if neither is found
    """
    try:
        eid, pname = one.pid2eid(probe_id)
    except Exception:
        return None
    for source, loader in [
        ('ap.meta', _load_channel_coordinates_from_meta),
        ('channels.localCoordinates', _load_channel_coordinates_from_sorting),
    ]:
        try:
            chn_coords = loader(eid, pname, one)
        except Exception as err:
            _logger.warning(f'Loading channel coordinates from {source} failed for {probe_id}: {err}')
            continue
        if chn_coords is not None:
            _logger.info(f'Using channel coordinates from {source} for {probe_id}')
            return chn_coords
    return None


def register_track(
    probe_id, picks=None, one=None, overwrite=False, channels=True, brain_atlas=None, endpoint='insertions', chn_coords=None
):
    """
    Register the user picks to a probe in Alyx
    Here we update Alyx models on the database in 3 steps
    1) The user picks converted to IBL coordinates will be stored in the json field of the
    corresponding probe insertion models
    2) The trajectory computed from the histology track is created or patched
    3) Channel locations are set in the table

    :param chn_coords: (n_channels, 2) array of lateral, axial channel positions in um. If None, they are
     read from the raw spikeglx ap.meta file, then from the channels.localCoordinates of the spike sorting
     (iblsorter, then pykilosort). If neither is found, the NP1 geometry is used.
    """
    assert one
    brain_atlas = brain_atlas or atlas.AllenAtlas()
    # 0) if it's an empty track, create a null trajectory and exit
    if picks is None or picks.size == 0:
        tdict = {
            'x': None,
            'y': None,
            'z': None,
            'phi': None,
            'theta': None,
            'depth': None,
            'roll': None,
            'provenance': 'Histology track',
            'coordinate_system': 'IBL-Allen',
        }
        if endpoint == 'chronic-insertions':
            tdict['chronic_insertion'] = probe_id
            tdict['probe_insertion'] = None
        else:
            tdict['probe_insertion'] = probe_id
            tdict['chronic_insertion'] = None

        brain_locations = None
        # Update the insertion qc to CRITICAL
        hist_qc = base.QC(probe_id, one=one, endpoint=endpoint)
        hist_qc.update_extended_qc({'tracing_exists': False})
        hist_qc.update('CRITICAL', namespace='tracing')
        insertion_histology = None
        # Here need to change the track qc to critical and also extended qc to zero
    else:
        if chn_coords is None and endpoint == 'insertions':
            chn_coords = _load_channel_coordinates(probe_id, one)
        if chn_coords is None:
            _logger.warning(f'No channel coordinates found for {probe_id}, using NP1 geometry for histology channels')

        brain_locations, insertion_histology = get_brain_regions(picks, channels_positions=chn_coords, brain_atlas=brain_atlas)
        # 1) update the alyx models, first put the picked points in the insertion json
        one.alyx.json_field_update(
            endpoint=endpoint, uuid=probe_id, field_name='json', data={'xyz_picks': np.int32(picks * 1e6).tolist()}
        )

        # Update the insertion qc to register tracing exits
        hist_qc = base.QC(probe_id, one=one, endpoint=endpoint)
        hist_qc.update_extended_qc({'tracing_exists': True})
        # 2) patch or create the trajectory coming from histology track
        tdict = create_trajectory_dict(probe_id, insertion_histology, provenance='Histology track', endpoint=endpoint)

    alyx_end = 'chronic_insertion' if endpoint == 'chronic-insertions' else 'probe_insertion'
    hist_traj = one.alyx.get(f'/trajectories?&{alyx_end}={probe_id}&provenance=Histology track', clobber=True)
    # if the trajectory exists, remove it, this will cascade delete existing channel locations
    if len(hist_traj):
        if overwrite:
            one.alyx.rest('trajectories', 'delete', id=hist_traj[0]['id'])
        else:
            raise FileExistsError(
                'The session already exists, however overwrite is set to False.If you want to overwrite, set overwrite=True.'
            )
    hist_traj = one.alyx.rest('trajectories', 'create', data=tdict)

    if brain_locations is None:
        return brain_locations, None
    # 3) create channel locations
    if channels:
        channel_dict = create_channel_dict(hist_traj, brain_locations)
        one.alyx.rest('channels', 'create', data=channel_dict)

    return brain_locations, insertion_histology


def register_aligned_track(probe_id, xyz_channels, chn_coords=None, one=None, overwrite=False, channels=True, brain_atlas=None):
    """
    Register ephys aligned trajectory and channel locations to Alyx
    Here we update Alyx models on the database in 2 steps
    1) The trajectory computed from the final electrode channel locations
    2) Channel locations are set to the trajectory
    """
    assert one
    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    if chn_coords is None:
        geometry = trace_header(version=1)
        chn_coords = np.c_[geometry['x'], geometry['y']]

    insertion = atlas.Insertion.from_track(xyz_channels, brain_atlas)
    tdict = create_trajectory_dict(probe_id, insertion, provenance='Ephys aligned histology track')

    hist_traj = one.alyx.rest(
        'trajectories', 'list', probe_insertion=probe_id, provenance='Ephys aligned histology track', no_cache=True
    )
    # if the trajectory exists, remove it, this will cascade delete existing channel locations
    if len(hist_traj):
        if overwrite:
            one.alyx.rest('trajectories', 'delete', id=hist_traj[0]['id'])
        else:
            raise FileExistsError(
                'The session already exists, however overwrite is set to False.If you want to overwrite, set overwrite=True.'
            )
    hist_traj = one.alyx.rest('trajectories', 'create', data=tdict)

    if channels:
        brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
        brain_regions['xyz'] = xyz_channels
        brain_regions['lateral'] = chn_coords[:, 0]
        brain_regions['axial'] = chn_coords[:, 1]
        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = create_channel_dict(hist_traj, brain_regions)
        one.alyx.rest('channels', 'create', data=channel_dict)


def create_trajectory_dict(probe_id, insertion, provenance, endpoint='insertions'):
    """
    Create trajectory dictionary in form to upload to alyx
    :param probe id: unique id of probe insertion
    :type probe_id: string (hexadecimal UUID)
    :param insertion: Insertion object describing entry and tip of trajectory
    :type insertion: object atlas.Insertion
    :param provenance: 'Histology track' or 'Ephys aligned histology track'
    :type provenance: string
    :param endpoint: Alyx endpoint, either 'insertions', or 'chronic-insertions'
    :type endpoint: string
    :return tdict:
    :type tdict: dict
    """
    tdict = {
        'x': insertion.x * 1e6,
        'y': insertion.y * 1e6,
        'z': insertion.z * 1e6,
        'phi': insertion.phi,
        'theta': insertion.theta,
        'depth': insertion.depth * 1e6,
        'roll': insertion.beta,
        'provenance': provenance,
        'coordinate_system': 'IBL-Allen',
    }
    if endpoint == 'chronic-insertions':
        tdict['chronic_insertion'] = probe_id
        tdict['probe_insertion'] = None
    else:
        tdict['probe_insertion'] = probe_id
        tdict['chronic_insertion'] = None

    return tdict


def create_channel_dict(traj, brain_locations):
    """
    Create channel dictionary in form to upload to alyx
    :param traj: alyx trajectory object to attach channel information to
    :type traj: dict
    :param brain_locations: information about location of electrode channels in brain atlas
    :type insertion: Bunch
    :return tdict:
    :type tdict: list of dict
    """
    channel_dict = []
    for i in np.arange(brain_locations.id.size):
        channel_dict.append({
            'x': np.float64(brain_locations.xyz[i, 0] * 1e6),
            'y': np.float64(brain_locations.xyz[i, 1] * 1e6),
            'z': np.float64(brain_locations.xyz[i, 2] * 1e6),
            'axial': np.float64(brain_locations.axial[i]),
            'lateral': np.float64(brain_locations.lateral[i]),
            'brain_region': int(brain_locations.id[i]),
            'trajectory_estimate': traj['id'],
        })

    return channel_dict


def _parse_filename(track_file):
    tmp = track_file.name.split('_')
    inumber = [i for i, s in enumerate(tmp) if s.isdigit and len(s) == 3][-1]
    search_filter = {
        'date': tmp[0],
        'experiment_number': int(tmp[inumber]),
        'name': '_'.join(tmp[inumber + 1 : -1]),
        'subject': '_'.join(tmp[1:inumber]),
    }
    return search_filter


def register_chronic_track_files(path_tracks, one=None, overwrite=False, brain_atlas=None):
    """
    Registers track files for chronic insertions
    :param path_tracks:
    :param one:
    :param overwrite:
    :param brain_atlas:
    :return:
    """

    brain_atlas = brain_atlas or atlas.AllenAtlas()
    glob_pattern = '*_probe*_pts*.csv'
    path_tracks = Path(path_tracks)

    if not path_tracks.is_dir():
        track_files = [path_tracks]
    else:
        track_files = list(path_tracks.rglob(glob_pattern))
        track_files.sort()

    assert path_tracks.exists()
    assert one

    ntracks = len(track_files)
    for ind, track_file in enumerate(track_files):
        # Nomenclature expected:
        # '{yyyy-mm-dd}}_{nickname}_{session_number}_{probe_label}_pts.csv'
        # beware: there may be underscores in the subject nickname

        search_filter = _parse_filename(track_file)
        probe = one.alyx.rest('chronic-insertions', 'list', no_cache=True, **search_filter)
        if len(probe) == 0:
            raise ValueError(
                f'Could not find associated chronic insertion for {search_filter["subject"]},{search_filter["name"]}'
            )
        elif len(probe) == 1:
            probe = probe[0]
        else:
            raise ValueError('Multiple chronic insertions found.')
        chronic_id = probe['id']
        try:
            xyz_picks = load_track_csv(track_file, brain_atlas=brain_atlas)
            register_chronic_track(chronic_id, xyz_picks, one=one, overwrite=overwrite, brain_atlas=brain_atlas)
        except Exception as e:
            _logger.error(str(track_file))
            raise e
        _logger.info(f'{ind + 1}/{ntracks}, {str(track_file)}')


def register_track_files(path_tracks, one=None, overwrite=False, brain_atlas=None):
    """
    :param path_tracks: path to directory containing tracks; also works with a single file name
    :param one:
    :return:
    """
    brain_atlas = brain_atlas or atlas.AllenAtlas()
    glob_pattern = '*_probe*_pts*.csv'
    path_tracks = Path(path_tracks)

    if not path_tracks.is_dir():
        track_files = [path_tracks]
    else:
        track_files = list(path_tracks.rglob(glob_pattern))
        track_files.sort()

    assert path_tracks.exists()
    assert one

    ntracks = len(track_files)
    for ind, track_file in enumerate(track_files):
        # Nomenclature expected:
        # '{yyyy-mm-dd}}_{nickname}_{session_number}_{probe_label}_pts.csv'
        # beware: there may be underscores in the subject nickname

        search_filter = _parse_filename(track_file)
        probe = one.alyx.rest('insertions', 'list', no_cache=True, **search_filter)
        if len(probe) == 0:
            eid = one.search(
                subject=search_filter['subject'], date_range=search_filter['date'], number=search_filter['experiment_number']
            )
            if len(eid) == 0:
                raise Exception(f'No session found {track_file.name}')
            insertion = {'session': eid[0], 'name': search_filter['name']}
            probe = one.alyx.rest('insertions', 'create', data=insertion)
        elif len(probe) == 1:
            probe = probe[0]
        else:
            raise ValueError('Multiple probes found.')
        probe_id = probe['id']
        try:
            xyz_picks = load_track_csv(track_file, brain_atlas=brain_atlas)
            register_track(probe_id, xyz_picks, one=one, overwrite=overwrite, brain_atlas=brain_atlas)
        except Exception as e:
            _logger.error(str(track_file))
            raise e
        _logger.info(f'{ind + 1}/{ntracks}, {str(track_file)}')


def detect_missing_histology_tracks(path_tracks=None, one=None, subject=None, brain_atlas=None):
    """
    Compares the number of probe insertions to the number of registered histology tracks to see if
    there is a discrepancy so that missing tracks can be properly logged in the database
    :param path_tracks: path to track files to be registered
    :param subject: subject nickname for which to detect missing tracks
    """

    brain_atlas = brain_atlas or atlas.AllenAtlas()
    if path_tracks:
        glob_pattern = '*_probe*_pts*.csv'

        path_tracks = Path(path_tracks)

        if not path_tracks.is_dir():
            track_files = [path_tracks]
        else:
            track_files = list(path_tracks.rglob(glob_pattern))
            track_files.sort()

        subjects = []
        for track_file in track_files:
            search_filter = _parse_filename(track_file)
            subjects.append(search_filter['subject'])

        unique_subjects = np.unique(subjects)
    elif not path_tracks and subject:
        unique_subjects = [subject]
    else:
        _logger.warning('Must specifiy either path_tracks or subject argument')
        return

    for subj in unique_subjects:
        insertions = one.alyx.rest('insertions', 'list', subject=subj, no_cache=True)
        trajectories = one.alyx.rest('trajectories', 'list', subject=subj, provenance='Histology track', no_cache=True)
        if len(insertions) != len(trajectories):
            ins_sess = np.array([ins['session'] + ins['name'] for ins in insertions])
            traj_sess = np.array([traj['session']['id'] + traj['probe_name'] for traj in trajectories])
            miss_idx = np.where(np.isin(ins_sess, traj_sess, invert=True))[0]

            for idx in miss_idx:
                info = one.eid2path(ins_sess[idx][:36], query_type='remote').parts
                print(ins_sess[idx][:36])
                msg = (
                    f'Histology tracing missing for {info[-3]}, {info[-2]}, {info[-1]},'
                    f' {ins_sess[idx][36:]}.\nEnter [y]es to register an empty track for '
                    f'this insertion \nEnter [n]o, if tracing for this probe insertion will be '
                    f'conducted at a later date \n>'
                )
                resp = input(msg)
                resp = resp.lower()
                if resp == 'y' or resp == 'yes':
                    _logger.info('Histology track for this probe insertion registered as empty')
                    probe_id = insertions[idx]['id']
                    print(insertions[idx]['session'])
                    print(probe_id)
                    register_track(probe_id, one=one, brain_atlas=brain_atlas)
                else:
                    _logger.info('Histology track for this probe insertion will not be registered')
                    continue


def coverage(trajs, ba=None, dist_fcn=[100, 150]):
    """
    Computes a coverage volume from
    :param trajs: dictionary of trajectories from Alyx rest endpoint (one.alyx.rest...)
    :param ba: iblatlas.atlas.BrainAtlas instance
    :return: 3D np.array the same size as the volume provided in the brain atlas
    """
    # in um. Coverage = 1 below the first value, 0 after the second, cosine taper in between
    ACTIVE_LENGTH_UM = 3.5 * 1e3
    MAX_DIST_UM = dist_fcn[1]  # max distance around the probe to be searched for
    if ba is None:
        ba = atlas.AllenAtlas()

    def crawl_up_from_tip(ins, d):
        return (ins.entry - ins.tip) * (d[:, np.newaxis] / np.linalg.norm(ins.entry - ins.tip)) + ins.tip

    full_coverage = np.zeros(ba.image.shape, dtype=np.float32).flatten()

    for p in np.arange(len(trajs)):
        if len(trajs) > 20:
            if p % 20 == 0:
                print(p / len(trajs))
        traj = trajs[p]

        ins = atlas.Insertion.from_dict(traj)
        # those are the top and bottom coordinates of the active part of the shank extended
        # to maxdist
        d = np.array([ACTIVE_LENGTH_UM + MAX_DIST_UM * np.sqrt(2), -MAX_DIST_UM * np.sqrt(2)]) + TIP_SIZE_UM
        top_bottom = crawl_up_from_tip(ins, d / 1e6)
        # this is the axis that has the biggest deviation. Almost always z
        axis = np.argmax(np.abs(np.diff(top_bottom, axis=0)))
        if axis != 2:
            _logger.warning(f'This works only for 45 degree or vertical tracks so far, skipping {ins}')
            continue
        # sample the active track path along this axis
        tbi = ba.bc.xyz2i(top_bottom)
        nz = tbi[1, axis] - tbi[0, axis] + 1
        ishank = np.round(np.array([np.linspace(tbi[0, i], tbi[1, i], nz) for i in np.arange(3)]).T).astype(np.int32)

        # creates a flattened "column" of candidate volume indices around the track
        # around each sample get an horizontal square slice of nx *2 +1 and ny *2 +1 samples
        # flatten the corresponding xyz indices and  compute the min distance to the track only
        # for those
        nx = int(np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[0]) * np.sqrt(2) / 2)) * 2 + 1
        ny = int(np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[1]) * np.sqrt(2) / 2)) * 2 + 1
        ixyz = np.stack([v.flatten() for v in np.meshgrid(np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), np.arange(nz))]).T
        ixyz[:, 0] = ishank[ixyz[:, 2], 0] + ixyz[:, 0]
        ixyz[:, 1] = ishank[ixyz[:, 2], 1] + ixyz[:, 1]
        ixyz[:, 2] = ishank[ixyz[:, 2], 2]
        # if any, remove indices that lie outside of the volume bounds
        iok = np.logical_and(0 <= ixyz[:, 0], ixyz[:, 0] < ba.bc.nx)
        iok &= np.logical_and(0 <= ixyz[:, 1], ixyz[:, 1] < ba.bc.ny)
        iok &= np.logical_and(0 <= ixyz[:, 2], ixyz[:, 2] < ba.bc.nz)
        ixyz = ixyz[iok, :]
        # get the minimum distance to the trajectory, to which is applied the cosine taper
        xyz = np.c_[ba.bc.xscale[ixyz[:, 0]], ba.bc.yscale[ixyz[:, 1]], ba.bc.zscale[ixyz[:, 2]]]
        sites_bounds = crawl_up_from_tip(ins, (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6)
        mdist = ins.trajectory.mindist(xyz, bounds=sites_bounds)
        coverage = 1 - fcn_cosine(np.array(dist_fcn) / 1e6)(mdist)
        # remap to the coverage volume
        flat_ind = ba._lookup_inds(ixyz)
        full_coverage[flat_ind] += coverage

    full_coverage = full_coverage.reshape(ba.image.shape)
    full_coverage[ba.label == 0] = np.nan
    return full_coverage, np.mean(xyz, 0), flat_ind


def coverage_grid(xyz_channels, spacing=500, ba=None):

    if ba is None:
        ba = atlas.AllenAtlas()

    def _get_scale_and_indices(v, bin, lim):
        _lim = [np.min(lim), np.max(lim)]
        # if bin is a nonzero scalar, this is a bin size: create scale and indices
        if np.isscalar(bin) and bin != 0:
            scale = np.arange(_lim[0], _lim[1] + bin / 2, bin)
            ind = (np.floor((v - _lim[0]) / bin)).astype(np.int64)
            if lim[0] > lim[1]:
                scale = scale[::-1]
                ind = (scale.size - 1) - ind
        # if bin == 0, aggregate over unique values
        else:
            scale, ind = np.unique(v, return_inverse=True)
        return scale, ind

    xlim = ba.bc.xlim
    ylim = ba.bc.ylim
    zlim = ba.bc.zlim

    xscale, xind = _get_scale_and_indices(xyz_channels[:, 0], spacing / 1e6, xlim)
    yscale, yind = _get_scale_and_indices(xyz_channels[:, 1], spacing / 1e6, ylim)
    zscale, zind = _get_scale_and_indices(xyz_channels[:, 2], spacing / 1e6, zlim)

    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny, nz = [xscale.size, yscale.size, zscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind, zind].transpose(), dims=(ny, nx, nz))
    r = np.bincount(ind2d, minlength=nx * ny * nz).reshape(ny, nx, nz)

    # Make things analagous to AllenAtlas
    dxyz = spacing / 1e6 * np.array([1, -1, -1])
    dims2xyz = np.array([1, 0, 2])
    nxyz = np.array(r.shape)[dims2xyz]
    iorigin = atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / spacing

    bc = atlas.BrainCoordinates(nxyz=nxyz, xyz0=(0, 0, 0), dxyz=dxyz)
    bc = atlas.BrainCoordinates(nxyz=nxyz, xyz0=-bc.i2xyz(iorigin), dxyz=dxyz)

    return r, bc

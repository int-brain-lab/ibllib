"""
Classes for manipulating brain atlases, insertions, and coordinates.
"""
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
import logging
import warnings

import numpy as np

import iblatlas.atlas


def deprecated_decorator(function):
    def deprecated_function(*args, **kwargs):
        warning_text = f"{function.__module__}.{function.__name__} is deprecated. " \
                       f"Use iblatlas.{function.__module__.split('.')[-1]}.{function.__name__} instead"
        warnings.warn(warning_text, DeprecationWarning)
        return function(*args, **kwargs)

    return deprecated_function

def deprecated_dataclass(dataclass):
    def deprecated_function():
        warning_text = f"{dataclass.__module__}.{dataclass.__name__} is deprecated. " \
                       f"Use iblatlas.{dataclass.__module__.split('.')[-1]}.{dataclass.__name__} instead"
        warnings.warn(warning_text, DeprecationWarning)
        return dataclass

    return deprecated_function()


ALLEN_CCF_LANDMARKS_MLAPDV_UM = {'bregma': np.array([5739, 5400, 332])}
"""dict: The ML AP DV voxel coordinates of brain landmarks in the Allen atlas."""

PAXINOS_CCF_LANDMARKS_MLAPDV_UM = {'bregma': np.array([5700, 4300 + 160, 330])}
"""dict: The ML AP DV voxel coordinates of brain landmarks in the Franklin & Paxinos atlas."""

S3_BUCKET_IBL = 'ibl-brain-wide-map-public'
"""str: The name of the public IBL S3 bucket containing atlas data."""

_logger = logging.getLogger(__name__)


@deprecated_decorator
def BrainCoordinates(*args, **kwargs):
    return iblatlas.atlas.BrainCoordinates(*args, **kwargs)


@deprecated_decorator
def BrainAtlas(*args, **kwargs):
    return iblatlas.atlas.BrainAtlas(*args, **kwargs)


class Trajectory(iblatlas.atlas.Trajectory):
    """
    3D Trajectory (usually for a linear probe), minimally defined by a vector and a point.

    Examples
    --------
    Instantiate from a best fit from an n by 3 array containing xyz coordinates:

    >>> trj = Trajectory.fit(xyz)
    """
    vector: np.ndarray
    point: np.ndarray

    def __init_sublcall__(self):
        warning_text = f"{dataclass.__module__}.{dataclass.__name__} is deprecated. " \
                       f"Use iblatlas.{dataclass.__module__.split('.')[-1]}.{dataclass.__name__} instead"
        warnings.warn(warning_text, DeprecationWarning)


@deprecated_dataclass
@dataclass
class Insertion(iblatlas.atlas.Insertion):
    """
    Defines an ephys probe insertion in 3D coordinate. IBL conventions.

    To instantiate, use the static methods: `Insertion.from_track` and `Insertion.from_dict`.
    """
    x: float
    y: float
    z: float
    phi: float
    theta: float
    depth: float
    label: str = ''
    beta: float = 0

    @staticmethod
    def from_track(xyzs, brain_atlas=None):
        """
        Define an insersion from one or more trajectory.

        Parameters
        ----------
        xyzs : numpy.array
             An n by 3 array xyz coordinates representing an insertion trajectory.
        brain_atlas : BrainAtlas
            A brain atlas instance, used to attain the point of entry.

        Returns
        -------
        Insertion
        """
        assert brain_atlas, 'Input argument brain_atlas must be defined'
        traj = Trajectory.fit(xyzs)
        # project the deepest point into the vector to get the tip coordinate
        tip = traj.project(xyzs[np.argmin(xyzs[:, 2]), :])
        # get intersection with the brain surface as an entry point
        entry = Insertion.get_brain_entry(traj, brain_atlas)
        # convert to spherical system to store the insertion
        depth, theta, phi = cart2sph(*(entry - tip))
        insertion_dict = {
            'x': entry[0], 'y': entry[1], 'z': entry[2], 'phi': phi, 'theta': theta, 'depth': depth
        }
        return Insertion(**insertion_dict)

    @staticmethod
    def from_dict(d, brain_atlas=None):
        """
        Constructs an Insertion object from the json information stored in probes.description file.

        Parameters
        ----------
        d : dict
            A dictionary containing at least the following keys {'x', 'y', 'z', 'phi', 'theta',
            'depth'}.  The depth and xyz coordinates must be in um.
        brain_atlas : BrainAtlas, default=None
            If provided, disregards the z coordinate and locks the insertion point to the z of the
            brain surface.

        Returns
        -------
        Insertion

        Examples
        --------
        >>> tri = {'x': 544.0, 'y': 1285.0, 'z': 0.0, 'phi': 0.0, 'theta': 5.0, 'depth': 4501.0}
        >>> ins = Insertion.from_dict(tri)
        """
        assert brain_atlas, 'Input argument brain_atlas must be defined'
        z = d['z'] / 1e6
        if not hasattr(brain_atlas, 'top'):
            brain_atlas.compute_surface()
        iy = brain_atlas.bc.y2i(d['y'] / 1e6)
        ix = brain_atlas.bc.x2i(d['x'] / 1e6)
        # Only use the brain surface value as z if it isn't NaN (this happens when the surface touches the edges
        # of the atlas volume
        if not np.isnan(brain_atlas.top[iy, ix]):
            z = brain_atlas.top[iy, ix]
        return Insertion(x=d['x'] / 1e6, y=d['y'] / 1e6, z=z,
                         phi=d['phi'], theta=d['theta'], depth=d['depth'] / 1e6,
                         beta=d.get('beta', 0), label=d.get('label', ''))

    @property
    def trajectory(self):
        """
        Gets the trajectory object matching insertion coordinates
        :return: atlas.Trajectory
        """
        return Trajectory.fit(self.xyz)

    @property
    def xyz(self):
        return np.c_[self.entry, self.tip].transpose()

    @property
    def entry(self):
        return np.array((self.x, self.y, self.z))

    @property
    def tip(self):
        return sph2cart(- self.depth, self.theta, self.phi) + np.array((self.x, self.y, self.z))

    @staticmethod
    def _get_surface_intersection(traj, brain_atlas, surface='top'):
        """
        TODO Document!

        Parameters
        ----------
        traj
        brain_atlas
        surface

        Returns
        -------

        """
        brain_atlas.compute_surface()

        distance = traj.mindist(brain_atlas.srf_xyz)
        dist_sort = np.argsort(distance)
        # In some cases the nearest two intersection points are not the top and bottom of brain
        # So we find all intersection points that fall within one voxel and take the one with
        # highest dV to be entry and lowest dV to be exit
        idx_lim = np.sum(distance[dist_sort] * 1e6 < np.max(brain_atlas.res_um))
        dist_lim = dist_sort[0:idx_lim]
        z_val = brain_atlas.srf_xyz[dist_lim, 2]
        if surface == 'top':
            ma = np.argmax(z_val)
            _xyz = brain_atlas.srf_xyz[dist_lim[ma], :]
            _ixyz = brain_atlas.bc.xyz2i(_xyz)
            _ixyz[brain_atlas.xyz2dims[2]] += 1
        elif surface == 'bottom':
            ma = np.argmin(z_val)
            _xyz = brain_atlas.srf_xyz[dist_lim[ma], :]
            _ixyz = brain_atlas.bc.xyz2i(_xyz)

        xyz = brain_atlas.bc.i2xyz(_ixyz.astype(float))

        return xyz

    @staticmethod
    def get_brain_exit(traj, brain_atlas):
        """
        Given a Trajectory and a BrainAtlas object, computes the brain exit coordinate as the
        intersection of the trajectory and the brain surface (brain_atlas.surface)
        :param brain_atlas:
        :return: 3 element array x,y,z
        """
        # Find point where trajectory intersects with bottom of brain
        return Insertion._get_surface_intersection(traj, brain_atlas, surface='bottom')

    @staticmethod
    def get_brain_entry(traj, brain_atlas):
        """
        Given a Trajectory and a BrainAtlas object, computes the brain entry coordinate as the
        intersection of the trajectory and the brain surface (brain_atlas.surface)
        :param brain_atlas:
        :return: 3 element array x,y,z
        """
        # Find point where trajectory intersects with top of brain
        return Insertion._get_surface_intersection(traj, brain_atlas, surface='top')


@deprecated_decorator
def AllenAtlas(*args, **kwargs):
    return iblatlas.atlas.AllenAtlas(*args, **kwargs)


@deprecated_decorator
def NeedlesAtlas(*args, **kwargs):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system. The Needles atlas defines a stretch along AP
    axis and a squeeze along the DV axis.

    Parameters
    ----------
    res_um : {10, 25, 50} int
        The Atlas resolution in micrometres; one of 10, 25 or 50um.
    **kwargs
        See AllenAtlas.

    Returns
    -------
    AllenAtlas
        An Allen atlas object with MRI atlas scaling applied.

    Notes
    -----
    The scaling was determined by manually transforming the DSURQE atlas [1]_ onto the Allen CCF.
    The DSURQE atlas is an MRI atlas acquired from 40 C57BL/6J mice post-mortem, with 40um
    isometric resolution.  The alignment was performed by Mayo Faulkner.
    The atlas data can be found `here <http://repo.mouseimaging.ca/repo/DSURQE_40micron_nifti/>`__.
    More information on the dataset and segmentation can be found
    `here <http://repo.mouseimaging.ca/repo/DSURQE_40micron/notes_on_DSURQE_atlas>`__.

    References
    ----------
    .. [1] Dorr AE, Lerch JP, Spring S, Kabani N, Henkelman RM (2008). High resolution
       three-dimensional brain atlas using an average magnetic resonance image of 40 adult C57Bl/6J
       mice. Neuroimage 42(1):60-9. [doi 10.1016/j.neuroimage.2008.03.037]
    """

    return iblatlas.atlas.NeedlesAtlas(*args, **kwargs)


@deprecated_decorator
def MRITorontoAtlas(*args, **kwargs):
    """
    The MRI Toronto brain atlas.

    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system. The MRI Toronto atlas defines a stretch along AP
    a squeeze along DV *and* a squeeze along ML. These are based on 12 p65 mice MRIs averaged [1]_.

    Parameters
    ----------
    res_um : {10, 25, 50} int
        The Atlas resolution in micrometres; one of 10, 25 or 50um.
    **kwargs
        See AllenAtlas.

    Returns
    -------
    AllenAtlas
        An Allen atlas object with MRI atlas scaling applied.

    References
    ----------
    .. [1] Qiu, LR, Fernandes, DJ, Szulc-Lerch, KU et al. (2018) Mouse MRI shows brain areas
       relatively larger in males emerge before those larger in females. Nat Commun 9, 2615.
       [doi 10.1038/s41467-018-04921-2]
    """
    return iblatlas.atlas.MRITorontoAtlas(*args, **kwargs)


@deprecated_decorator
def FranklinPaxinosAtlas(*args, **kwargs):
    return iblatlas.atlas.FranklinPaxinosAtlas(*args, **kwargs)

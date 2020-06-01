from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

import pandas as pd
import numpy as np
import nrrd

from brainbox.core import Bunch
from ibllib.io import params
from oneibl.webclient import http_download_file

ALLEN_CCF_LANDMARKS_MLAPDV_UM = {'bregma': np.array([5739, 5400, 332])}


def cart2sph(x, y, z):
    """
    Converts cartesian to spherical Coordinates
    theta: polar angle, phi: azimuth
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x) * 180 / np.pi
    theta = np.zeros_like(r)
    iok = r != 0
    theta[iok] = np.arccos(z[iok] / r[iok]) * 180 / np.pi
    if theta.size == 1:
        theta = np.float(theta)
    return r, theta, phi


def sph2cart(r, theta, phi):
    """
    Converts Spherical to Cartesian coordinates
    theta: polar angle, phi: azimuth
    """
    x = r * np.cos(phi / 180 * np.pi) * np.sin(theta / 180 * np.pi)
    y = r * np.sin(phi / 180 * np.pi) * np.sin(theta / 180 * np.pi)
    z = r * np.cos(theta / 180 * np.pi)
    return x, y, z


class BrainCoordinates:
    """
    Class for mapping and indexing a 3D array to real-world coordinates
    x = ml, right positive
    y = ap, anterior positive
    z = dv, dorsal positive

    The layout of the Atlas dimension is done according to the most used sections so they lay
    contiguous on disk assuming C-ordering: V[iap, iml, idv]

    nxyz: number of elements along each cartesian axis (nx, ny, nz) = (nml, nap, ndv)
    xyz0: coordinates of the element volume[0, 0, 0]] in the coordinate space
    dxyz: spatial interval of the volume along the 3 dimensions
    """

    def __init__(self, nxyz, xyz0=[0, 0, 0], dxyz=[1, 1, 1]):
        if np.isscalar(dxyz):
            dxyz = [dxyz for i in range(3)]
        self.x0, self.y0, self.z0 = list(xyz0)
        self.dx, self.dy, self.dz = list(dxyz)
        self.nx, self.ny, self.nz = list(nxyz)

    @property
    def dxyz(self):
        return np.array([self.dx, self.dy, self.dz])

    @property
    def nxyz(self):
        return np.array([self.nx, self.ny, self.nz])

    """Methods ratios to indice"""
    def r2ix(self, r):
        return int((self.nx - 1) * r)

    def r2iy(self, r):
        return int((self.nz - 1) * r)

    def r2iz(self, r):
        return int((self.nz - 1) * r)

    """Methods distance to indice"""
    @staticmethod
    def _round(i, round=True):
        if round:
            return np.round(i).astype(np.int)
        else:
            return i

    def x2i(self, x, round=True):
        return self._round((x - self.x0) / self.dx, round=round)

    def y2i(self, y, round=True):
        return self._round((y - self.y0) / self.dy, round=round)

    def z2i(self, z, round=True):
        return self._round((z - self.z0) / self.dz, round=round)

    def xyz2i(self, xyz, round=True):
        xyz = np.array(xyz)
        dt = np.int if round else np.float
        out = np.zeros_like(xyz, dtype=dt)
        out[..., 0] = self.x2i(xyz[..., 0], round=round)
        out[..., 1] = self.y2i(xyz[..., 1], round=round)
        out[..., 2] = self.z2i(xyz[..., 2], round=round)
        return out

    """Methods indices to distance"""
    def i2x(self, ind):
        return ind * self.dx + self.x0

    def i2y(self, ind):
        return ind * self.dy + self.y0

    def i2z(self, ind):
        return ind * self.dz + self.z0

    def i2xyz(self, iii):
        iii = np.array(iii)
        out = np.zeros_like(iii)
        out[..., 0] = self.i2x(iii[..., 0])
        out[..., 1] = self.i2y(iii[..., 1])
        out[..., 2] = self.i2z(iii[..., 2])
        return out

    """Methods bounds"""
    @property
    def xlim(self):
        return self.i2x(np.array([0, self.nx - 1]))

    @property
    def ylim(self):
        return self.i2y(np.array([0, self.ny - 1]))

    @property
    def zlim(self):
        return self.i2z(np.array([0, self.nz - 1]))

    def lim(self, axis):
        if axis == 0:
            return self.xlim
        elif axis == 1:
            return self.ylim
        elif axis == 2:
            return self.zlim

    """returns scales"""
    @property
    def xscale(self):
        return self.i2x(np.arange(self.nx))

    @property
    def yscale(self):
        return self.i2y(np.arange(self.ny))

    @property
    def zscale(self):
        return self.i2z(np.arange(self.nz))

    """returns the 3d mgrid used for 3d visualization"""
    @property
    def mgrid(self):
        return np.meshgrid(self.xscale, self.yscale, self.zscale)


class BrainAtlas:
    """
    Objects that holds image, labels and coordinate transforms for a brain Atlas.
    Currently this is designed for the AllenCCF at several resolutions,
    yet this class can be used for other atlases arises.
    """
    def __init__(self, image, label, dxyz, regions, iorigin=[0, 0, 0],
                 dims2xyz=[0, 1, 2], xyz2dims=[0, 1, 2]):
        """
        self.image: image volume (ap, ml, dv)
        self.label: label volume (ap, ml, dv)
        self.bc: atlas.BrainCoordinate object
        self.regions: atlas.BrainRegions object
        self.top: 2d np array (ap, ml) containing the z-coordinate (m) of the surface of the brain
        self.dims2xyz and self.zyz2dims: map image axis order to xyz coordinates order
        """
        self.image = image
        self.label = label
        self.regions = regions
        self.dims2xyz = dims2xyz
        self.xyz2dims = xyz2dims
        assert(np.all(self.dims2xyz[self.xyz2dims] == np.array([0, 1, 2])))
        assert(np.all(self.xyz2dims[self.dims2xyz] == np.array([0, 1, 2])))
        # create the coordinate transform object that maps volume indices to real world coordinates
        nxyz = np.array(self.image.shape)[self.dims2xyz]
        bc = BrainCoordinates(nxyz=nxyz, xyz0=(0, 0, 0), dxyz=dxyz)
        self.bc = BrainCoordinates(nxyz=nxyz, xyz0=- bc.i2xyz(iorigin), dxyz=dxyz)
        """
        Get the volume top surface, this is needed to compute probe insertions intersections
        """
        l0 = self.label == 0
        bottom = np.zeros(self.label.shape[:2])
        top = np.zeros(self.label.shape[:2])
        top[np.all(l0, axis=2)] = np.nan
        bottom[np.all(l0, axis=2)] = np.nan
        iz = 0
        # not very elegant, but fast enough for our purposes
        while True:
            if iz >= l0.shape[2]:
                break
            top[np.bitwise_and(top == 0, ~l0[:, :, iz])] = iz
            ireverse = l0.shape[2] - 1 - iz
            bottom[np.bitwise_and(bottom == 0, ~l0[:, :, ireverse])] = ireverse
            iz += 1
        self.top = self.bc.i2z(top)
        self.bottom = self.bc.i2z(bottom)

    def _lookup(self, xyz):
        """
        Performs a 3D lookup from real world coordinates to the flat indices in the volume
        defined in the BrainCoordinates object
        :param xyz: [n, 3] array of coordinates
        :return: n array of label values
        """
        idims = np.split(self.bc.xyz2i(xyz)[:, self.xyz2dims], [1, 2], axis=-1)
        inds = np.ravel_multi_index(idims, self.bc.nxyz[self.xyz2dims])
        return inds.squeeze()

    def get_labels(self, xyz):
        """
        Performs a 3D lookup from real world coordinates to the volume labels
        :param xyz: [n, 3] array of coordinates
        :return: n array of label values
        """
        return self.label.flat[self._lookup(xyz)]

    def _label2rgb(self, imlabel):
        """
        Converts a slice from the label volume to its RGB equivalent for display
        :param imlabel: 2D np-array containing label ids (slice of the label volume)
        :return: 3D np-array of the slice uint8 rgb values
        """
        if self.regions is None or getattr(self.regions, 'rgb', None) is None:
            return imlabel
        else:  # if the regions exist and have the rgb attribute, do the rgb lookup
            # the lookup is done in pure numpy for speed. This is the ismember matlab fcn
            im_unique, ilabels, iim = np.unique(imlabel, return_index=True, return_inverse=True)
            _, ir_unique, _ = np.intersect1d(self.regions.id, im_unique, return_indices=True)
            return np.reshape(self.regions.rgb[ir_unique[iim], :], (*imlabel.shape, 3))

    def tilted_slice(self, xyz, axis, volume='image'):
        """
        From line coordinates, extracts the tilted plane containing the line from the 3D volume
        :param xyz: np.array: points defining a probe trajectory in 3D space (xyz triplets)
        if more than 2 points are provided will take the best fit
        :param axis:
            0: along ml = sagittal-slice
            1: along ap = coronal-slice
            2: along dv = horizontal-slice
        :param volume: 'image' or 'annotation'
        :return: np.array, abscissa extent (width), ordinate extent (height),
        squeezed axis extent (depth)
        """
        if axis == 0:   # sagittal slice (squeeze/take along ml-axis)
            wdim, hdim, ddim = (1, 2, 0)
        elif axis == 1:  # coronal slice (squeeze/take along ap-axis)
            wdim, hdim, ddim = (0, 2, 1)
        elif axis == 2:  # horizontal slice (squeeze/take along dv-axis)
            wdim, hdim, ddim = (0, 1, 2)
        # get the best fit and find exit points of the volume along squeezed axis
        trj = Trajectory.fit(xyz)
        sub_volume = trj._eval(self.bc.lim(axis=hdim), axis=hdim)
        sub_volume[:, wdim] = self.bc.lim(axis=wdim)
        sub_volume_i = self.bc.xyz2i(sub_volume)
        tile_shape = np.array([np.diff(sub_volume_i[:, hdim])[0] + 1, self.bc.nxyz[wdim]])
        # get indices along each dimension
        indx = np.arange(tile_shape[1])
        indy = np.arange(tile_shape[0])
        inds = np.linspace(*sub_volume_i[:, ddim], tile_shape[0])
        # compute the slice indices and output the slice
        _, INDS = np.meshgrid(indx, np.int64(np.around(inds)))
        INDX, INDY = np.meshgrid(indx, indy)
        indsl = [[INDX, INDY, INDS][i] for i in np.argsort([wdim, hdim, ddim])[self.xyz2dims]]
        if volume.lower() == 'annotation':
            tslice = self._label2rgb(self.label[indsl[0], indsl[1], indsl[2]])
        elif volume.lower() == 'image':
            tslice = self.image[indsl[0], indsl[1], indsl[2]]

        #  get extents with correct convention NB: matplotlib flips the y-axis on imshow !
        width = np.sort(sub_volume[:, wdim])[np.argsort(self.bc.lim(axis=wdim))]
        height = np.flipud(np.sort(sub_volume[:, hdim])[np.argsort(self.bc.lim(axis=hdim))])
        depth = np.flipud(np.sort(sub_volume[:, ddim])[np.argsort(self.bc.lim(axis=ddim))])
        return tslice, width, height, depth

    def plot_tilted_slice(self, xyz, axis, volume='image', cmap=None, ax=None, **kwargs):
        """
        From line coordinates, extracts the tilted plane containing the line from the 3D volume
        :param xyz: np.array: points defining a probe trajectory in 3D space (xyz triplets)
        if more than 2 points are provided will take the best fit
        :param axis:
            0: along ml = sagittal-slice
            1: along ap = coronal-slice
            2: along dv = horizontal-slice
        :param volume: 'image' or 'annotation'
        :return: matplotlib axis
        """
        if axis == 0:
            axis_labels = np.array(['ap (um)', 'dv (um)', 'ml (um)'])
        elif axis == 1:
            axis_labels = np.array(['ml (um)', 'dv (um)', 'ap (um)'])
        elif axis == 2:
            axis_labels = np.array(['ml (um)', 'ap (um)', 'dv (um)'])

        tslice, width, height, depth = self.tilted_slice(xyz, axis, volume=volume)
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        if not ax:
            plt.figure()
            ax = plt.gca()
            ax.axis('equal')
        if not cmap:
            cmap = plt.get_cmap('bone')
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        ax.imshow(tslice, extent=np.r_[width, height], cmap=cmap, **kwargs)
        sec_ax = ax.secondary_yaxis('right', functions=(
                                    lambda x: x * ab[0] + ab[1],
                                    lambda y: (y - ab[1]) / ab[0]))
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        sec_ax.set_ylabel(axis_labels[2])
        return ax

    @staticmethod
    def _plot_slice(im, extent, ax=None, cmap=None, **kwargs):
        if not ax:
            ax = plt.gca()
            ax.axis('equal')
        if not cmap:
            cmap = plt.get_cmap('bone')
        ax.imshow(im, extent=extent, cmap=cmap, **kwargs)
        return ax

    def slice(self, coordinate, axis, volume='image'):
        """
        :param coordinate: float
        :param axis: xyz convention:  0 for ml, 1 for ap, 2
        :param volume: 'image' or 'annotation'
        :return: 2d array or 3d RGB numpy int8 array
        """
        index = self.bc.xyz2i(np.array([coordinate] * 3))[axis]
        if volume == 'annotation':
            im = self.label.take(index, axis=self.xyz2dims[axis])
            return self._label2rgb(im)
        elif volume == 'image':
            return self.image.take(index, axis=self.xyz2dims[axis])

    def plot_cslice(self, ap_coordinate, volume='image', **kwargs):
        """
        Imshow a coronal slice
        :param: ap_coordinate (m)
        :param volume: 'image' or 'annotation'
        :return: ax
        """
        cslice = self.slice(ap_coordinate, axis=1, volume=volume)
        extent = np.r_[self.bc.xlim, np.flip(self.bc.zlim)] * 1e6
        return self._plot_slice(np.swapaxes(cslice, 0, 1), extent=extent, **kwargs)

    def plot_hslice(self, dv_coordinate, volume='image', **kwargs):
        """
        Imshow a horizontal slice
        :param: dv_coordinate (m)
        :param volume: 'image' or 'annotation'
        :return: ax
        """
        hslice = self.slice(dv_coordinate, axis=2, volume=volume)
        extent = np.r_[self.bc.ylim, self.bc.xlim] * 1e6
        return self._plot_slice(np.swapaxes(hslice, 0, 1), extent=extent, **kwargs)

    def plot_sslice(self, ml_coordinate, volume='image', **kwargs):
        """
        Imshow a sagittal slice
        :param: ml_coordinate (m)
        :param volume: 'image' or 'annotation'
        :return: ax
        """
        sslice = self.slice(ml_coordinate, axis=0, volume=volume)
        extent = np.r_[self.bc.ylim, np.flip(self.bc.zlim)] * 1e6
        return self._plot_slice(np.swapaxes(sslice, 0, 1), extent=extent, **kwargs)


@dataclass
class Trajectory:
    """
    3D Trajectory (usually for a linear probe). Minimally defined by a vector and a point.
    instantiate from a best fit from a n by 3 array containing xyz coordinates:
        trj = Trajectory.fit(xyz)
    """
    vector: np.ndarray
    point: np.ndarray

    @staticmethod
    def fit(xyz):
        """
        fits a line to a 3D cloud of points, returns a Trajectory object
        :param xyz: n by 3 numpy array containing cloud of points
        :returns: a Trajectory object
        """
        xyz_mean = np.mean(xyz, axis=0)
        return Trajectory(vector=np.linalg.svd(xyz - xyz_mean)[2][0], point=xyz_mean)

    def eval_x(self, x):
        """
        given an array of x coordinates, returns the xyz array of coordinates along the insertion
        :param x: n by 1 or numpy array containing x-coordinates
        :return: n by 3 numpy array containing xyz-coordinates
        """
        return self._eval(x, axis=0)

    def eval_y(self, y):
        """
        given an array of y coordinates, returns the xyz array of coordinates along the insertion
        :param y: n by 1 or numpy array containing y-coordinates
        :return: n by 3 numpy array containing xyz-coordinates
        """
        return self._eval(y, axis=1)

    def eval_z(self, z):
        """
        given an array of z coordinates, returns the xyz array of coordinates along the insertion
        :param z: n by 1 or numpy array containing z-coordinates
        :return: n by 3 numpy array containing xyz-coordinates
        """
        return self._eval(z, axis=2)

    def project(self, point):
        """
        projects a point onto the trajectory line
        :param point: np.array(x, y, z) coordinates
        :return:
        """
        return (self.point + np.dot(point - self.point, self.vector) /
                np.dot(self.vector, self.vector) * self.vector)

    def _eval(self, c, axis):
        # uses symmetric form of 3d line equation to get xyz coordinates given one coordinate
        if not isinstance(c, np.ndarray):
            c = np.array(c)
        while c.ndim < 2:
            c = c[..., np.newaxis]
        # there are cases where it's impossible to project if a line is // to the axis
        if self.vector[axis] == 0:
            return np.nan * np.zeros((c.shape[0], 3))
        else:
            return (c - self.point[axis]) * self.vector / self.vector[axis] + self.point

    def exit_points(self, bc):
        """
        Given a Trajectory and a BrainCoordinates object, computes the intersection of the
        trajectory with the brain coordinates bounding box
        :param bc: BrainCoordinate objects
        :return: np.ndarray 2 y 3 corresponding to exit points xyz coordinates
        """
        bounds = np.c_[bc.xlim, bc.ylim, bc.zlim]
        epoints = np.r_[self.eval_x(bc.xlim), self.eval_y(bc.ylim), self.eval_z(bc.zlim)]
        epoints = epoints[~np.all(np.isnan(epoints), axis=1)]
        ind = np.all(np.bitwise_and(bounds[0, :] <= epoints, epoints <= bounds[1, :]), axis=1)
        return epoints[ind, :]


@dataclass
class Insertion:
    """
    Defines an ephys probe insertion in 3D coordinate. IBL conventions.
    To instantiate, use the static methods:
    Insertion.from_track
    Insertion.from_dict
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
        :param brain_atlas: None. If provided, disregards the z coordinate and locks the insertion
        point to the z of the brain surface
        :return: Trajectory object
        """
        assert brain_atlas, 'Input argument brain_atlas must be defined'
        traj = Trajectory.fit(xyzs)
        # project the deepest point into the vector to get the tip coordinate
        tip = traj.project(xyzs[np.argmin(xyzs[:, 2]), :])
        # get intersection with the brain surface as an entry point
        entry = Insertion.get_brain_entry(traj, brain_atlas)
        # convert to spherical system to store the insertion
        depth, theta, phi = cart2sph(*(entry - tip))
        insertion_dict = {'x': entry[0], 'y': entry[1], 'z': entry[2],
                          'phi': phi, 'theta': theta, 'depth': depth}
        return Insertion(**insertion_dict)

    @staticmethod
    def from_dict(d, brain_atlas=None):
        """
        Constructs an Insertion object from the json information stored in probes.description file
        :param trj: dictionary containing at least the following keys, in um
           {
            'x': 544.0,
            'y': 1285.0,
            'z': 0.0,
            'phi': 0.0,
            'theta': 5.0,
            'depth': 4501.0
            }
        :param brain_atlas: None. If provided, disregards the z coordinate and locks the insertion
        point to the z of the brain surface
        :return: Trajectory object
        """
        z = d['z'] / 1e6
        if brain_atlas:
            iy = brain_atlas.bc.y2i(d['y'] / 1e6)
            ix = brain_atlas.bc.x2i(d['x'] / 1e6)
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
    def _get_surface_intersection(traj, brain_atlas, surface, z=0):
        """
        Given a Trajectory and a BrainAtlas object, computes the intersection of the trajectory
        and a surface (usually brain_atlas.top)
        :param brain_atlas:
        :param z: init position for the lookup
        :return: 3 element array x,y,z
        """
        # do a recursive look-up of the brain surface along the trajectory, 5 is more than enough
        for m in range(5):
            xyz = traj.eval_z(z)[0]
            iy = brain_atlas.bc.y2i(xyz[1])
            ix = brain_atlas.bc.x2i(xyz[0])
            z = surface[iy, ix]
        return xyz

    @staticmethod
    def get_brain_exit(traj, brain_atlas):
        """
        Given a Trajectory and a BrainAtlas object, computes the brain entry coordinate as the
        intersection of the trajectory and the brain surface (brain_atlas.top)
        :param brain_atlas:
        :return: 3 element array x,y,z
        """
        # do a recursive look-up of the brain surface along the trajectory, 5 is more than enough
        return Insertion._get_surface_intersection(traj, brain_atlas,
                                                   brain_atlas.bottom, z=brain_atlas.bc.zlim[-1])

    @staticmethod
    def get_brain_entry(traj, brain_atlas):
        """
        Given a Trajectory and a BrainAtlas object, computes the brain entry coordinate as the
        intersection of the trajectory and the brain surface (brain_atlas.top)
        :param brain_atlas:
        :return: 3 element array x,y,z
        """
        # do a recursive look-up of the brain surface along the trajectory, 5 is more than enough
        return Insertion._get_surface_intersection(traj, brain_atlas, brain_atlas.top, z=0)


@dataclass
class BrainRegions:
    """
    self.id: contains label ids found in the BrainCoordinate.label volume
    self.name: list/tuple of brain region names
    self.acronym: list/tuple of brain region acronyms
    """
    id: np.ndarray
    name: np.object
    acronym: np.object
    rgb: np.uint8

    def get(self, ids) -> Bunch:
        """
        Get a bunch of the name/id
        """
        uid, uind = np.unique(ids, return_inverse=True)
        a, iself, _ = np.intersect1d(self.id, uid, assume_unique=False, return_indices=True)
        return Bunch(id=self.id[iself[uind]],
                     name=self.name[iself[uind]],
                     acronym=self.acronym[iself[uind]],
                     rgb=self.rgb[iself[uind]])


class AllenAtlas(BrainAtlas):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system
    """

    def __init__(self, res_um=25, par=None, scaling=np.array([1, 1, 1]), mock=False,
                 hist_path=None):
        """
        :param res_um: 10, 25 or 50 um
        :param par: dictionary of parameters to override systems ones
        :param scaling:
        :param mock:
        :return: atlas.BrainAtlas
        """
        par = params.read('one_params')
        FILE_REGIONS = str(Path(__file__).parent.joinpath('allen_structure_tree.csv'))
        FLAT_IRON_ATLAS_REL_PATH = Path('histology', 'ATLAS', 'Needles', 'Allen')
        if mock:
            image, label = [np.zeros((528, 456, 320), dtype=np.bool) for _ in range(2)]
        elif hist_path:
            path_atlas = Path(par.CACHE_DIR).joinpath(FLAT_IRON_ATLAS_REL_PATH)
            file_label = path_atlas.joinpath(f'annotation_{res_um}.nrrd')
            if not file_label.exists():
                _download_atlas_flatiron(file_label, FLAT_IRON_ATLAS_REL_PATH, par)
            image, _ = nrrd.read(hist_path, index_order='C')  # dv, ml, ap
            label, _ = nrrd.read(file_label, index_order='C')  # dv, ml, ap
            label = np.swapaxes(np.swapaxes(label, 2, 0), 1, 2)  # label[iap, iml, idv]
            image = np.swapaxes(np.swapaxes(image, 2, 0), 1, 2)  # image[iap, iml, idv]
            # Make sure histology image has the same dimensions as CCF
            assert (image.shape == label.shape)
        else:
            path_atlas = Path(par.CACHE_DIR).joinpath(FLAT_IRON_ATLAS_REL_PATH)
            file_image = path_atlas.joinpath(f'average_template_{res_um}.nrrd')
            file_label = path_atlas.joinpath(f'annotation_{res_um}.nrrd')
            if not file_image.exists():
                _download_atlas_flatiron(file_image, FLAT_IRON_ATLAS_REL_PATH, par)
            if not file_label.exists():
                _download_atlas_flatiron(file_label, FLAT_IRON_ATLAS_REL_PATH, par)
            image, _ = nrrd.read(file_image, index_order='C')  # dv, ml, ap
            label, _ = nrrd.read(file_label, index_order='C')  # dv, ml, ap
            label = np.swapaxes(np.swapaxes(label, 2, 0), 1, 2)  # label[iap, iml, idv]
            image = np.swapaxes(np.swapaxes(image, 2, 0), 1, 2)  # image[iap, iml, idv]
        # resulting volumes origin: x right, y front, z top
        regions = _regions_from_allen_csv(FILE_REGIONS)
        xyz2dims = np.array([1, 0, 2])
        dims2xyz = np.array([1, 0, 2])
        dxyz = res_um * 1e-6 * np.array([1, -1, -1]) * scaling
        # we use Bregma as the origin
        ibregma = (ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / res_um)
        self.res_um = res_um
        super().__init__(image, label, dxyz, regions, ibregma,
                         dims2xyz=dims2xyz, xyz2dims=xyz2dims)

    def xyz2ccf(self, xyz):
        """
        Converts coordinates to the CCF coordinates, which is assumed to be the cube indices
        times the spacing so far.
        :param xyz:
        :return: mlapdv coordinates in um, origin is the front left top corner of the data volume
        """
        return self.bc.xyz2i(xyz) * np.float(self.res_um)


def NeedlesAtlas(*args, **kwargs):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system. The Needles atlas defines a stretch along AP
    axis and a sqeeze along the DV axis.
    :param res_um: 10, 25 or 50 um
    :return: atlas.BrainAtlas
    """
    DV_SCALE = 0.952  # multiplicative factor on DV dimension, determined from MRI->CCF transform
    AP_SCALE = 1.087  # multiplicative factor on AP dimension
    kwargs['scaling'] = np.array([1, AP_SCALE, DV_SCALE])
    return AllenAtlas(*args, **kwargs)


def _download_atlas_flatiron(file_image, FLAT_IRON_ATLAS_REL_PATH, par):
    file_image.parent.mkdir(exist_ok=True, parents=True)
    url = (par.HTTP_DATA_SERVER + '/' +
           '/'.join(FLAT_IRON_ATLAS_REL_PATH.parts) + '/' + file_image.name)
    http_download_file(url, cache_dir=Path(par.CACHE_DIR).joinpath(FLAT_IRON_ATLAS_REL_PATH),
                       username=par.HTTP_DATA_SERVER_LOGIN,
                       password=par.HTTP_DATA_SERVER_PWD)


def _regions_from_allen_csv(csv_file):
    """
    Reads csv file containing the ALlen Ontology and instantiates a BrainRegions object
    :param csv_file:
    :return: BrainRegions object
    """
    df_regions = pd.read_csv(csv_file)
    # converts colors to RGB uint8 array
    c = np.uint32(df_regions.color_hex_triplet.map(
        lambda x: int(x, 16) if isinstance(x, str) else 256 ** 3 - 1))
    c = np.flip(np.reshape(c.view(np.uint8), (df_regions.id.size, 4))[:, :3], 1)
    # creates the BrainRegion instance
    return BrainRegions(id=df_regions.id.values,
                        name=df_regions.name.values,
                        acronym=df_regions.acronym.values,
                        rgb=c)

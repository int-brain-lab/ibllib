from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

import pandas as pd
import numpy as np
import nrrd

from brainbox.core import Bunch
from ibllib.io import params


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
    Currently this is the designted for the AllenCCF at several resolutions,
    yet this class could be extended/subclassed in the future if the need for other atlases arises.
    """
    def __init__(self, image, label, regions, dxyz, iorigin=[0, 0, 0],
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
        Get the volume top surface
        """
        l0 = self.label == 0
        s = np.zeros(self.label.shape[:2])
        s[np.all(l0, axis=2)] = np.nan
        iz = 0
        # not very elegant, but fast enough for our purposes
        while True:
            if iz >= l0.shape[2]:
                break
            inds = np.bitwise_and(s == 0, ~l0[:, :, iz])
            s[inds] = iz
            iz += 1
        self.top = self.bc.i2z(s)

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

    def _tilted_slice(self, linepts, sxdim=0, sydim=1, ssdim=1):
        """
        Get a slice from the volume, tilted around 1 rotation axis
        :param linepts: 2 points defining a probe trajectory. This trajectory is projected onto the
        sxdim=0. The extracted slice corresponds to the plane orthogonal to the sxdim=0 plane
        passing by the projected trajectory.
        :param sxdim: = 0  coordinate system dimension corresponding to slice abscissa
         (this direction is the rotation axis for tilt)
        :param sydim: = 2  coordinate system dimension corresponding to slice ordinate
        :param: ssdim: = 1  squeezed dimension

        For a tilted coronal slice (default), sxdim=0, sydim=2, ssdim=1
        For a tilted sagittal slice, sxdim=1, sydim=2, ssdim=0
        """
        tilt_line = linepts.copy()
        tilt_line[:, sxdim] = 0
        tilt_line_i = self.bc.xyz2i(tilt_line)
        tile_shape = np.array([np.diff(tilt_line_i[:, sydim])[0] + 1, self.bc.nxyz[sxdim]])
        indx = np.arange(tile_shape[1])
        indy = np.arange(tile_shape[0])
        inds = np.linspace(*tilt_line_i[:, ssdim], tile_shape[0])
        _, INDS = np.meshgrid(indx, np.int64(np.around(inds)))
        INDX, INDY = np.meshgrid(indx, indy)
        inds = [[INDX, INDY, INDS][i] for i in np.argsort([sxdim, sydim, ssdim])[self.xyz2dims]]
        return self.image[inds[0], inds[1], inds[2]]

    @staticmethod
    def _plot_slice(im, extent, ax=None, cmap=None, **kwargs):
        if not ax:
            ax = plt.gca()
            ax.axis('equal')
        if not cmap:
            cmap = plt.get_cmap('bone')
        ax.imshow(im, extent=extent, cmap=cmap, **kwargs)
        return ax

    def plot_cslice(self, ap_coordinate, volume='image', **kwargs):
        """
        Imshow a coronal slice
        :param: ap_coordinate (mm)
        :param: ax
        """
        vol = self.label if volume == 'annotation' else self.image
        return self._plot_slice(vol[self.bc.y2i(ap_coordinate / 1e3), :, :].transpose(),
                                extent=np.r_[self.bc.xlim * 1e3, np.flip(self.bc.zlim) * 1e3],
                                **kwargs)

    def plot_hslice(self, dv_coordinate, volume='image', **kwargs):
        """
        Imshow a horizontal slice
        :param: dv_coordinate (mm)
        :param: ax
        """
        vol = self.label if volume == 'annotation' else self.image
        return self._plot_slice(vol[:, :, self.bc.z2i(dv_coordinate / 1e3)].transpose(),
                                extent=np.r_[self.bc.ylim * 1e3, self.bc.xlim * 1e3],
                                **kwargs)

    def plot_sslice(self, ml_coordinate, volume='image', **kwargs):
        """
        Imshow a sagittal slice
        :param: ml_coordinate (mm)
        :param: ax
        """
        vol = self.label if volume == 'annotation' else self.image
        return self._plot_slice(vol[:, self.bc.x2i(ml_coordinate / 1e3), :].transpose(),
                                extent=np.r_[self.bc.ylim * 1e3, np.flip(self.bc.zlim) * 1e3],
                                **kwargs)


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
        return self._eval(x, dim=0)

    def eval_y(self, y):
        """
        given an array of y coordinates, returns the xyz array of coordinates along the insertion
        :param y: n by 1 or numpy array containing y-coordinates
        :return: n by 3 numpy array containing xyz-coordinates
        """
        return self._eval(y, dim=1)

    def eval_z(self, z):
        """
        given an array of z coordinates, returns the xyz array of coordinates along the insertion
        :param z: n by 1 or numpy array containing z-coordinates
        :return: n by 3 numpy array containing xyz-coordinates
        """
        return self._eval(z, dim=2)

    def _eval(self, c, dim):
        # uses symmetric form of 3d line equation to get xyz coordinates given one coordinate
        if not isinstance(c, np.ndarray):
            c = np.array(c)
        while c.ndim < 2:
            c = c[..., np.newaxis]
        # there are cases where it's impossible to project if a line is // to the axis
        if self.vector[dim] == 0:
            return np.nan * np.zeros((c.shape[0], 3))
        else:
            return (c - self.point[dim]) * self.vector / self.vector[dim] + self.point

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
    label: str
    x: float
    y: float
    z: float
    phi: float
    theta: float
    depth: float
    beta: float

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
            ix = brain_atlas.bc.y2i(d['x'] / 1e6)
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

    def get(self, ids) -> Bunch:
        """
        Get a bunch of the name/id
        """
        uid, uind = np.unique(ids, return_inverse=True)
        a, iself, _ = np.intersect1d(self.id, uid, assume_unique=False, return_indices=True)
        return Bunch(id=self.id[iself[uind]], name=self.name[iself[uind]],
                     acronym=self.acronym[iself[uind]])


def AllenAtlas(res_um=25, par=None):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system
    :param res_um: 25 or 50 um
    :return: atlas.BrainAtlas
    """
    if par is None:
        # Bregma indices for the 10um Allen Brain Atlas, mlapdv
        pdefault = {
            'PATH_ATLAS': '/datadisk/BrainAtlas/ATLASES/Allen/',
            'FILE_REGIONS': str(Path(__file__).parent.joinpath('allen_structure_tree.csv')),
            'INDICES_BREGMA': list(np.array([1140 - (570 + 3.9), 540, 0 + 33.2]))
        }
        par = params.read('ibl_histology', default=pdefault)
        if not Path(par.PATH_ATLAS).exists():
            raise NotImplementedError("Atlas doesn't exist ! Mock option not implemented yet")
            # TODO: mock atlas to get only the coordinate framework
            pass
        params.write('ibl_histology', par)
    else:
        par = Bunch(par)
    # file_image = Path(path_atlas).joinpath(f'ara_nissl_{res_um}.nrrd')
    file_image = Path(par.PATH_ATLAS).joinpath(f'average_template_{res_um}.nrrd')
    file_label = Path(par.PATH_ATLAS).joinpath(f'annotation_{res_um}.nrrd')
    image, header = nrrd.read(file_image, index_order='C')  # dv, ml, ap
    image = np.swapaxes(np.swapaxes(image, 2, 0), 1, 2)  # image[iap, iml, idv]
    label, header = nrrd.read(file_label, index_order='C')  # dv, ml, ap
    label = np.swapaxes(np.swapaxes(label, 2, 0), 1, 2)  # label[iap, iml, idv]
    df_regions = pd.read_csv(par.FILE_REGIONS)
    regions = BrainRegions(id=df_regions.id.values,
                           name=df_regions.name.values,
                           acronym=df_regions.acronym.values)
    xyz2dims = np.array([1, 0, 2])
    dims2xyz = np.array([1, 0, 2])
    dxyz = res_um * 1e-6 * np.array([1, -1, -1])
    ibregma = (np.array(par.INDICES_BREGMA) * 10 / res_um)
    return BrainAtlas(image, label, regions, dxyz, ibregma, dims2xyz=dims2xyz, xyz2dims=xyz2dims)

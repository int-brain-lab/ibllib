from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

import pandas as pd
import numpy as np
import nrrd

from brainbox.core import Bunch


def cart2sph(x, y, z):
    """ Converts cartesian to spherical Coordinates"""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x) * 180 / np.pi
    phi = np.zeros_like(r)
    iok = r != 0
    phi[iok] = np.arccos(z[iok] / r[iok]) * 180 / np.pi
    return r, theta, phi


def sph2cart(r, theta, phi):
    """ Converts Spherical to Cartesian coordinates"""
    x = r * np.cos(theta / 180 * np.pi) * np.sin(phi / 180 * np.pi)
    y = r * np.sin(theta / 180 * np.pi) * np.sin(phi / 180 * np.pi)
    z = r * np.cos(phi / 180 * np.pi)
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
        out[..., 1] = self.x2i(xyz[..., 1], round=round)
        out[..., 2] = self.x2i(xyz[..., 2], round=round)
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


class BrainAtlas:
    """
    Objects that holds image, labels and coordinate transforms for a brain Atlas.
    Currently this is the AllenCCF only at several resolutions, this class could be extended
    subclassed in the future if the need for other atlases arises.
    """
    def __init__(self, image, label, regions, dxyz, iorigin=[0, 0, 0],
                 dims2xyz=[0, 1, 2], xyz2dims=[0, 1, 2]):
        """
        self.image: image volume (ap, ml, dv)
        self.label: label volume (ap, ml, dv)
        self.bc: atlas.BrainCoordinate object
        self.regions: atlas.BrainRegions object
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

    def plot_cslice(self, ap_coordinate, ax=None):
        """
        Imshow a coronal slice
        :param: ap_coordinate (mm)
        """
        if not ax:
            ax = plt.gca()
        plt.imshow(self.image[self.bc.y2i(ap_coordinate / 1e3), :, :].transpose(),
                   extent=np.r_[self.bc.xlim * 1e3, np.flip(self.bc.zlim) * 1e3],
                   cmap=plt.get_cmap('seismic'))
        return ax

    def plot_hslice(self, dv_coordinate, ax=None):
        """
        Imshow a horizontal slice
        :param: dv_coordinate (mm)
        """
        if not ax:
            ax = plt.gca()
        plt.imshow(self.image[:, :, self.bc.z2i(dv_coordinate / 1e3)].transpose(),
                   extent=np.r_[self.bc.ylim * 1e3, self.bc.xlim * 1e3],
                   cmap=plt.get_cmap('seismic'))
        return ax

    def plot_sslice(self, ml_coordinate, ax=None):
        """
        Imshow a sagittal slice
        :param: ml_coordinate (mm)
        """
        if not ax:
            ax = plt.gca()
        plt.imshow(self.image[:, self.bc.x2i(ml_coordinate / 1e3), :].transpose(),
                   extent=np.r_[self.bc.ylim * 1e3, np.flip(self.bc.zlim) * 1e3],
                   cmap=plt.get_cmap('seismic'))
        return ax


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
        _, iself, _ = np.intersect1d(self.id, uid, assume_unique=True, return_indices=True)
        return Bunch(id=self.id[iself[uind]], name=self.name[iself[uind]],
                     acronym=self.acronym[iself[uind]])


def AllenAtlas(res_um=25):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system
    :param res_um: 25 or 50 um
    :return: atlas.BrainAtlas
    """
    # Bregma indices for the 10um Allen Brain Atlas, mlapdv
    INDICES_BREGMA = np.array([1140 - (570 + 3.9), 540, 0 + 33.2])
    PATH_ATLAS = '/datadisk/BrainAtlas/ATLASES/Allen/'
    FILE_REGIONS = Path(__file__).parent.joinpath('allen_structure_tree.csv')
    # file_image = Path(path_atlas).joinpath(f'ara_nissl_{res_um}.nrrd')
    file_image = Path(PATH_ATLAS).joinpath(f'average_template_{res_um}.nrrd')
    file_label = Path(PATH_ATLAS).joinpath(f'annotation_{res_um}.nrrd')
    image, header = nrrd.read(file_image, index_order='C')  # dv, ml, ap
    image = np.swapaxes(np.swapaxes(image, 2, 0), 1, 2)  # image[iap, iml, idv]
    label, header = nrrd.read(file_label, index_order='C')  # dv, ml, ap
    label = np.swapaxes(np.swapaxes(label, 2, 0), 1, 2)  # label[iap, iml, idv]
    df_regions = pd.read_csv(FILE_REGIONS)
    regions = BrainRegions(id=df_regions.id.values,
                           name=df_regions.name.values,
                           acronym=df_regions.acronym.values)
    xyz2dims = np.array([1, 0, 2])
    dims2xyz = np.array([1, 0, 2])
    dxyz = res_um * 1e-6 * np.array([1, -1, -1])
    ibregma = (INDICES_BREGMA * 10 / res_um)
    return BrainAtlas(image, label, regions, dxyz, ibregma, dims2xyz=dims2xyz, xyz2dims=xyz2dims)

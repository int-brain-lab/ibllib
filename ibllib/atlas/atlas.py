from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
from pathlib import Path, PurePosixPath
import numpy as np
import nrrd

from one.webclient import http_download_file
import one.params
import one.remote.aws as aws

from iblutil.numerical import ismember
from ibllib.atlas.regions import BrainRegions, FranklinPaxinosRegions


_logger = logging.getLogger(__name__)
ALLEN_CCF_LANDMARKS_MLAPDV_UM = {'bregma': np.array([5739, 5400, 332])}
PAXINOS_CCF_LANDMARKS_MLAPDV_UM = {'bregma': np.array([5700, 4300 + 160, 330])}

S3_BUCKET_IBL = 'ibl-brain-wide-map-public'


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
        theta = float(theta)
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
        nanval = 0
        if round:
            ii = np.array(np.round(i)).astype(int)
            ii[np.isnan(i)] = nanval
            return ii
        else:
            return i

    def x2i(self, x, round=True, mode='raise'):
        i = np.asarray(self._round((x - self.x0) / self.dx, round=round))
        if np.any(i < 0) or np.any(i >= self.nx):
            if mode == 'clip':
                i[i < 0] = 0
                i[i >= self.nx] = self.nx - 1
            elif mode == 'raise':
                raise ValueError("At least one x value lies outside of the atlas volume.")
            elif mode == 'wrap':
                pass
        return i

    def y2i(self, y, round=True, mode='raise'):
        i = np.asarray(self._round((y - self.y0) / self.dy, round=round))
        if np.any(i < 0) or np.any(i >= self.ny):
            if mode == 'clip':
                i[i < 0] = 0
                i[i >= self.ny] = self.ny - 1
            elif mode == 'raise':
                raise ValueError("At least one y value lies outside of the atlas volume.")
            elif mode == 'wrap':
                pass
        return i

    def z2i(self, z, round=True, mode='raise'):
        i = np.asarray(self._round((z - self.z0) / self.dz, round=round))
        if np.any(i < 0) or np.any(i >= self.nz):
            if mode == 'clip':
                i[i < 0] = 0
                i[i >= self.nz] = self.nz - 1
            elif mode == 'raise':
                raise ValueError("At least one z value lies outside of the atlas volume.")
            elif mode == 'wrap':
                pass
        return i

    def xyz2i(self, xyz, round=True, mode='raise'):
        """
        :param mode: {‘raise’, 'clip', 'wrap'} determines what to do when determined index lies outside the atlas volume
                     'raise' will raise a ValueError
                     'clip' will replace the index with the closest index inside the volume
                     'wrap' will wrap around to the other side of the volume. This is only here for legacy reasons
        """
        xyz = np.array(xyz)
        dt = int if round else float
        out = np.zeros_like(xyz, dtype=dt)
        out[..., 0] = self.x2i(xyz[..., 0], round=round, mode=mode)
        out[..., 1] = self.y2i(xyz[..., 1], round=round, mode=mode)
        out[..., 2] = self.z2i(xyz[..., 2], round=round, mode=mode)
        return out

    """Methods indices to distance"""
    def i2x(self, ind):
        return ind * self.dx + self.x0

    def i2y(self, ind):
        return ind * self.dy + self.y0

    def i2z(self, ind):
        return ind * self.dz + self.z0

    def i2xyz(self, iii):
        iii = np.array(iii, dtype=float)
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
        assert np.all(self.dims2xyz[self.xyz2dims] == np.array([0, 1, 2]))
        assert np.all(self.xyz2dims[self.dims2xyz] == np.array([0, 1, 2]))
        # create the coordinate transform object that maps volume indices to real world coordinates
        nxyz = np.array(self.image.shape)[self.dims2xyz]
        bc = BrainCoordinates(nxyz=nxyz, xyz0=(0, 0, 0), dxyz=dxyz)
        self.bc = BrainCoordinates(nxyz=nxyz, xyz0=-bc.i2xyz(iorigin), dxyz=dxyz)

        self.surface = None
        self.boundary = None

    @staticmethod
    def _get_cache_dir():
        par = one.params.get(silent=True)
        path_atlas = Path(par.CACHE_DIR).joinpath(PurePosixPath('histology', 'ATLAS', 'Needles', 'Allen', 'flatmaps'))
        return path_atlas

    def compute_surface(self):
        """
        Get the volume top, bottom, left and right surfaces, and from these the outer surface of
        the image volume. This is needed to compute probe insertions intersections.

        NOTE: In places where the top or bottom surface touch the top or bottom of the atlas volume, the surface
        will be set to np.nan. If you encounter issues working with these surfaces check if this might be the cause.
        """
        if self.surface is None:  # only compute if it hasn't already been computed
            axz = self.xyz2dims[2]  # this is the dv axis
            _surface = (self.label == 0).astype(np.int8) * 2
            l0 = np.diff(_surface, axis=axz, append=2)
            _top = np.argmax(l0 == -2, axis=axz).astype(float)
            _top[_top == 0] = np.nan
            _bottom = self.bc.nz - np.argmax(np.flip(l0, axis=axz) == 2, axis=axz).astype(float)
            _bottom[_bottom == self.bc.nz] = np.nan
            self.top = self.bc.i2z(_top + 1)
            self.bottom = self.bc.i2z(_bottom - 1)
            self.surface = np.diff(_surface, axis=self.xyz2dims[0], append=2) + l0
            idx_srf = np.where(self.surface != 0)
            self.surface[idx_srf] = 1
            self.srf_xyz = self.bc.i2xyz(np.c_[idx_srf[self.xyz2dims[0]], idx_srf[self.xyz2dims[1]],
                                               idx_srf[self.xyz2dims[2]]].astype(float))

    def _lookup_inds(self, ixyz, mode='raise'):
        """
        Performs a 3D lookup from volume indices ixyz to the image volume
        :param ixyz: [n, 3] array of indices in the mlapdv order
        :return: n array of flat indices
        """
        idims = np.split(ixyz[..., self.xyz2dims], [1, 2], axis=-1)
        inds = np.ravel_multi_index(idims, self.bc.nxyz[self.xyz2dims], mode=mode)
        return inds.squeeze()

    def _lookup(self, xyz, mode='raise'):
        """
        Performs a 3D lookup from real world coordinates to the flat indices in the volume
        defined in the BrainCoordinates object
        :param xyz: [n, 3] array of coordinates
        :return: n array of flat indices
        """
        return self._lookup_inds(self.bc.xyz2i(xyz, mode=mode), mode=mode)

    def get_labels(self, xyz, mapping=None, radius_um=None, mode='raise'):
        """
        Performs a 3D lookup from real world coordinates to the volume labels
        and return the regions ids according to the mapping
        :param xyz: [n, 3] array of coordinates
        :param mapping: brain region mapping (defaults to original Allen mapping)
        :param radius_um: if not null, returns a regions ids array and an array of proportion
         of regions in a sphere of size radius around the coordinates.
        :return: n array of region ids
        """
        mapping = mapping or self.regions.default_mapping

        if radius_um:
            nrx = int(np.ceil(radius_um / abs(self.bc.dx) / 1e6))
            nry = int(np.ceil(radius_um / abs(self.bc.dy) / 1e6))
            nrz = int(np.ceil(radius_um / abs(self.bc.dz) / 1e6))
            nr = [nrx, nry, nrz]
            iii = self.bc.xyz2i(xyz)
            # computing the cube radius and indices is more complicated as volume indices are not
            # necessariy in ml, ap, dv order so the indices order is dynamic
            rcube = np.meshgrid(*tuple((np.arange(
                -nr[i], nr[i] + 1) * self.bc.dxyz[i]) ** 2 for i in self.xyz2dims))
            rcube = np.sqrt(rcube[0] + rcube[1], rcube[2]) * 1e6
            icube = tuple(slice(-nr[i] + iii[i], nr[i] + iii[i] + 1) for i in self.xyz2dims)
            cube = self.regions.mappings[mapping][self.label[icube]]
            ilabs, counts = np.unique(cube[rcube <= radius_um], return_counts=True)
            return self.regions.id[ilabs], counts / np.sum(counts)
        else:
            regions_indices = self._get_mapping(mapping=mapping)[self.label.flat[self._lookup(xyz, mode=mode)]]
            return self.regions.id[regions_indices]

    def _get_mapping(self, mapping=None):
        """
        Safe way to get mappings if nothing defined in regions.
        A mapping transforms from the full allen brain Atlas ids to the remapped ids
        new_ids = ids[mapping]
        """
        mapping = mapping or self.regions.default_mapping
        if hasattr(self.regions, 'mappings'):
            return self.regions.mappings[mapping]
        else:
            return np.arange(self.regions.id.size)

    def _label2rgb(self, imlabel):
        """
        Converts a slice from the label volume to its RGB equivalent for display
        :param imlabel: 2D np-array containing label ids (slice of the label volume)
        :return: 3D np-array of the slice uint8 rgb values
        """
        if getattr(self.regions, 'rgb', None) is None:
            return self.regions.id[imlabel]
        else:  # if the regions exist and have the rgb attribute, do the rgb lookup
            return self.regions.rgb[imlabel]

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
        if isinstance(volume, np.ndarray):
            tslice = volume[indsl[0], indsl[1], indsl[2]]
        elif volume.lower() == 'annotation':
            tslice = self._label2rgb(self.label[indsl[0], indsl[1], indsl[2]])
        elif volume.lower() == 'image':
            tslice = self.image[indsl[0], indsl[1], indsl[2]]
        elif volume.lower() == 'surface':
            tslice = self.surface[indsl[0], indsl[1], indsl[2]]

        #  get extents with correct convention NB: matplotlib flips the y-axis on imshow !
        width = np.sort(sub_volume[:, wdim])[np.argsort(self.bc.lim(axis=wdim))]
        height = np.flipud(np.sort(sub_volume[:, hdim])[np.argsort(self.bc.lim(axis=hdim))])
        depth = np.flipud(np.sort(sub_volume[:, ddim])[np.argsort(self.bc.lim(axis=ddim))])
        return tslice, width, height, depth

    def plot_tilted_slice(self, xyz, axis, volume='image', cmap=None, ax=None, sec_ax=False, **kwargs):
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
        if sec_ax:
            return ax, sec_ax
        else:
            return ax

    @staticmethod
    def _plot_slice(im, extent, ax=None, cmap=None, volume=None, **kwargs):
        if not ax:
            ax = plt.gca()
            ax.axis('equal')
        if not cmap:
            cmap = plt.get_cmap('bone')

        if volume == 'boundary':
            imb = np.zeros((*im.shape[:2], 4), dtype=np.uint8)
            imb[im == 1] = np.array([0, 0, 0, 255])
            im = imb

        ax.imshow(im, extent=extent, cmap=cmap, **kwargs)
        return ax

    def extent(self, axis):
        """
        :param axis: direction along which the volume is stacked:
         (2 = z for horizontal slice)
         (1 = y for coronal slice)
         (0 = x for sagittal slice)
        :return:
        """

        if axis == 0:
            extent = np.r_[self.bc.ylim, np.flip(self.bc.zlim)] * 1e6
        elif axis == 1:
            extent = np.r_[self.bc.xlim, np.flip(self.bc.zlim)] * 1e6
        elif axis == 2:
            extent = np.r_[self.bc.xlim, np.flip(self.bc.ylim)] * 1e6
        return extent

    def slice(self, coordinate, axis, volume='image', mode='raise', region_values=None,
              mapping=None, bc=None):
        """
        Get slice through atlas

        :param coordinate: coordinate to slice in metres, float
        :param axis: xyz convention:  0 for ml, 1 for ap, 2 for dv
            - 0: sagittal slice (along ml axis)
            - 1: coronal slice (along ap axis)
            - 2: horizontal slice (along dv axis)
        :param volume:
            - 'image' - allen image volume
            - 'annotation' - allen annotation volume
            - 'surface' - outer surface of mesh
            - 'boundary' - outline of boundaries between all regions
            - 'volume' - custom volume, must pass in volume of shape ba.image.shape as regions_value argument
            - 'value' - custom value per allen region, must pass in array of shape ba.regions.id as regions_value argument
        :param mode: error mode for out of bounds coordinates
            -   'raise' raise an error
            -   'clip' gets the first or last index
        :param region_values: custom values to plot
            - if volume='volume', region_values must have shape ba.image.shape
            - if volume='value', region_values must have shape ba.regions.id
        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :return: 2d array or 3d RGB numpy int8 array
        """
        if axis == 0:
            index = self.bc.x2i(np.array(coordinate), mode=mode)
        elif axis == 1:
            index = self.bc.y2i(np.array(coordinate), mode=mode)
        elif axis == 2:
            index = self.bc.z2i(np.array(coordinate), mode=mode)

        # np.take is 50 thousand times slower than straight slicing !
        def _take(vol, ind, axis):
            if mode == 'clip':
                ind = np.minimum(np.maximum(ind, 0), vol.shape[axis] - 1)
            if axis == 0:
                return vol[ind, :, :]
            elif axis == 1:
                return vol[:, ind, :]
            elif axis == 2:
                return vol[:, :, ind]

        def _take_remap(vol, ind, axis, mapping):
            # For the labels, remap the regions indices according to the mapping
            return self._get_mapping(mapping=mapping)[_take(vol, ind, axis)]

        if isinstance(volume, np.ndarray):
            return _take(volume, index, axis=self.xyz2dims[axis])
        elif volume in 'annotation':
            iregion = _take_remap(self.label, index, self.xyz2dims[axis], mapping)
            return self._label2rgb(iregion)
        elif volume == 'image':
            return _take(self.image, index, axis=self.xyz2dims[axis])
        elif volume == 'value':
            return region_values[_take_remap(self.label, index, self.xyz2dims[axis], mapping)]
        elif volume == 'image':
            return _take(self.image, index, axis=self.xyz2dims[axis])
        elif volume in ['surface', 'edges']:
            self.compute_surface()
            return _take(self.surface, index, axis=self.xyz2dims[axis])
        elif volume == 'boundary':
            iregion = _take_remap(self.label, index, self.xyz2dims[axis], mapping)
            return self.compute_boundaries(iregion)

        elif volume == 'volume':
            if bc is not None:
                index = bc.xyz2i(np.array([coordinate] * 3))[axis]
            return _take(region_values, index, axis=self.xyz2dims[axis])

    def compute_boundaries(self, values):
        """
        Compute the boundaries between regions on slice
        :param values:
        :return:
        """
        boundary = np.abs(np.diff(values, axis=0, prepend=0))
        boundary = boundary + np.abs(np.diff(values, axis=1, prepend=0))
        boundary = boundary + np.abs(np.diff(values, axis=1, append=0))
        boundary = boundary + np.abs(np.diff(values, axis=0, append=0))

        boundary[boundary != 0] = 1

        return boundary

    def plot_slices(self, xyz, *args, **kwargs):
        """
        From a single coordinate, plots the 3 slices that intersect at this point in a single
        matplotlib figure
        :param xyz: mlapdv coordinate in m
        :param args: arguments to be forwarded to plot slices
        :param kwargs: keyword arguments to be forwarded to plot slices
        :return: 2 by 2 array of axes
        """
        fig, axs = plt.subplots(2, 2)
        self.plot_cslice(xyz[1], *args, ax=axs[0, 0], **kwargs)
        self.plot_sslice(xyz[0], *args, ax=axs[0, 1], **kwargs)
        self.plot_hslice(xyz[2], *args, ax=axs[1, 0], **kwargs)
        xyz_um = xyz * 1e6
        axs[0, 0].plot(xyz_um[0], xyz_um[2], 'g*')
        axs[0, 1].plot(xyz_um[1], xyz_um[2], 'g*')
        axs[1, 0].plot(xyz_um[0], xyz_um[1], 'g*')
        return axs

    def plot_cslice(self, ap_coordinate, volume='image', mapping=None, region_values=None, **kwargs):
        """
        Plot coronal slice through atlas at given ap_coordinate

        :param: ap_coordinate (m)
        :param volume:
            - 'image' - allen image volume
            - 'annotation' - allen annotation volume
            - 'surface' - outer surface of mesh
            - 'boundary' - outline of boundaries between all regions
            - 'volume' - custom volume, must pass in volume of shape ba.image.shape as regions_value argument
            - 'value' - custom value per allen region, must pass in array of shape ba.regions.id as regions_value argument
        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :param region_values: custom values to plot
            - if volume='volume', region_values must have shape ba.image.shape
            - if volume='value', region_values must have shape ba.regions.id
        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :param **kwargs: matplotlib.pyplot.imshow kwarg arguments
        :return: matplotlib ax object
        """

        cslice = self.slice(ap_coordinate, axis=1, volume=volume, mapping=mapping, region_values=region_values)
        return self._plot_slice(np.moveaxis(cslice, 0, 1), extent=self.extent(axis=1), volume=volume, **kwargs)

    def plot_hslice(self, dv_coordinate, volume='image', mapping=None, region_values=None, **kwargs):
        """
        Plot horizontal slice through atlas at given dv_coordinate

        :param: dv_coordinate (m)
        :param volume:
            - 'image' - allen image volume
            - 'annotation' - allen annotation volume
            - 'surface' - outer surface of mesh
            - 'boundary' - outline of boundaries between all regions
            - 'volume' - custom volume, must pass in volume of shape ba.image.shape as regions_value argument
            - 'value' - custom value per allen region, must pass in array of shape ba.regions.id as regions_value argument
        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :param region_values: custom values to plot
            - if volume='volume', region_values must have shape ba.image.shape
            - if volume='value', region_values must have shape ba.regions.id
        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :param **kwargs: matplotlib.pyplot.imshow kwarg arguments
        :return: matplotlib ax object
        """

        hslice = self.slice(dv_coordinate, axis=2, volume=volume, mapping=mapping, region_values=region_values)
        return self._plot_slice(hslice, extent=self.extent(axis=2), volume=volume, **kwargs)

    def plot_sslice(self, ml_coordinate, volume='image', mapping=None, region_values=None, **kwargs):
        """
        Plot sagittal slice through atlas at given ml_coordinate

        :param: ml_coordinate (m)
        :param volume:
            - 'image' - allen image volume
            - 'annotation' - allen annotation volume
            - 'surface' - outer surface of mesh
            - 'boundary' - outline of boundaries between all regions
            - 'volume' - custom volume, must pass in volume of shape ba.image.shape as regions_value argument
            - 'value' - custom value per allen region, must pass in array of shape ba.regions.id as regions_value argument
        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :param region_values: custom values to plot
            - if volume='volume', region_values must have shape ba.image.shape
            - if volume='value', region_values must have shape ba.regions.id
        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :param **kwargs: matplotlib.pyplot.imshow kwarg arguments
        :return: matplotlib ax object
        """

        sslice = self.slice(ml_coordinate, axis=0, volume=volume, mapping=mapping, region_values=region_values)
        return self._plot_slice(np.swapaxes(sslice, 0, 1), extent=self.extent(axis=0), volume=volume, **kwargs)

    def plot_top(self, volume='annotation', mapping=None, region_values=None, ax=None, **kwargs):
        """
        Plot top view of atlas
        :param volume:
            - 'image' - allen image volume
            - 'annotation' - allen annotation volume
            - 'boundary' - outline of boundaries between all regions
            - 'volume' - custom volume, must pass in volume of shape ba.image.shape as regions_value argument
            - 'value' - custom value per allen region, must pass in array of shape ba.regions.id as regions_value argument

        :param mapping: mapping to use. Options can be found using ba.regions.mappings.keys()
        :param region_values:
        :param ax:
        :param kwargs:
        :return:
        """

        self.compute_surface()
        ix, iy = np.meshgrid(np.arange(self.bc.nx), np.arange(self.bc.ny))
        iz = self.bc.z2i(self.top)
        inds = self._lookup_inds(np.stack((ix, iy, iz), axis=-1))

        regions = self._get_mapping(mapping=mapping)[self.label.flat[inds]]

        if volume == 'annotation':
            im = self._label2rgb(regions)
        elif volume == 'image':
            im = self.top
        elif volume == 'value':
            im = region_values[regions]
        elif volume == 'volume':
            im = np.zeros((iz.shape))
            for x in range(im.shape[0]):
                for y in range(im.shape[1]):
                    im[x, y] = region_values[x, y, iz[x, y]]
        elif volume == 'boundary':
            im = self.compute_boundaries(regions)

        return self._plot_slice(im, self.extent(axis=2), ax=ax, volume=volume, **kwargs)


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
        # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        if point.ndim == 1:
            return self.project(point[np.newaxis])[0]
        return (self.point + np.dot(point[:, np.newaxis] - self.point, self.vector) /
                np.dot(self.vector, self.vector) * self.vector)

    def mindist(self, xyz, bounds=None):
        """
        Computes the minimum distance to the trajectory line for one or a set of points.
        If bounds are provided, computes the minimum distance to the segment instead of an
        infinite line.
        :param xyz: [..., 3]
        :param bounds: defaults to None.  np.array [2, 3]: segment boundaries, inf line if None
        :return: minimum distance [...]
        """
        proj = self.project(xyz)
        d = np.sqrt(np.sum((proj - xyz) ** 2, axis=-1))
        if bounds is not None:
            # project the boundaries and the points along the traj
            b = np.dot(bounds, self.vector)
            ob = np.argsort(b)
            p = np.dot(xyz[:, np.newaxis], self.vector).squeeze()
            # for points below and above boundaries, compute cartesian distance to the boundary
            imin = p < np.min(b)
            d[imin] = np.sqrt(np.sum((xyz[imin, :] - bounds[ob[0], :]) ** 2, axis=-1))
            imax = p > np.max(b)
            d[imax] = np.sqrt(np.sum((xyz[imax, :] - bounds[ob[1], :]) ** 2, axis=-1))
        return d

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


class AllenAtlas(BrainAtlas):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system
    """

    def __init__(self, res_um=25, scaling=np.array([1, 1, 1]), mock=False, hist_path=None):
        """
        :param res_um: 10, 25 or 50 um
        :param scaling: scale factor along ml, ap, dv for squeeze and stretch ([1, 1, 1])
        :param mock: for testing purpose
        :param hist_path
        :return: atlas.BrainAtlas
        """

        par = one.params.get(silent=True)
        FLAT_IRON_ATLAS_REL_PATH = PurePosixPath('histology', 'ATLAS', 'Needles', 'Allen')
        LUT_VERSION = "v01"  # version 01 is the lateralized version
        regions = BrainRegions()
        xyz2dims = np.array([1, 0, 2])  # this is the c-contiguous ordering
        dims2xyz = np.array([1, 0, 2])
        # we use Bregma as the origin
        self.res_um = res_um
        ibregma = (ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / self.res_um)
        dxyz = self.res_um * 1e-6 * np.array([1, -1, -1]) * scaling
        if mock:
            image, label = [np.zeros((528, 456, 320), dtype=np.int16) for _ in range(2)]
            label[:, :, 100:105] = 1327  # lookup index for retina, id 304325711 (no id 1327)
        else:
            path_atlas = Path(par.CACHE_DIR).joinpath(FLAT_IRON_ATLAS_REL_PATH)
            file_image = hist_path or path_atlas.joinpath(f'average_template_{res_um}.nrrd')
            # get the image volume
            if not file_image.exists():
                _download_atlas_allen(file_image, FLAT_IRON_ATLAS_REL_PATH, par)
            # get the remapped label volume
            file_label = path_atlas.joinpath(f'annotation_{res_um}.nrrd')
            if not file_label.exists():
                _download_atlas_allen(file_label, FLAT_IRON_ATLAS_REL_PATH, par)
            file_label_remap = path_atlas.joinpath(f'annotation_{res_um}_lut_{LUT_VERSION}.npz')
            if not file_label_remap.exists():
                label = self._read_volume(file_label).astype(dtype=np.int32)
                _logger.info("computing brain atlas annotations lookup table")
                # lateralize atlas: for this the regions of the left hemisphere have primary
                # keys opposite to to the normal ones
                lateral = np.zeros(label.shape[xyz2dims[0]])
                lateral[int(np.floor(ibregma[0]))] = 1
                lateral = np.sign(np.cumsum(lateral)[np.newaxis, :, np.newaxis] - 0.5)
                label = label * lateral.astype(np.int32)
                # the 10 um atlas is too big to fit in memory so work by chunks instead
                if res_um == 10:
                    first, ncols = (0, 10)
                    while True:
                        last = np.minimum(first + ncols, label.shape[-1])
                        _logger.info(f"Computing... {last} on {label.shape[-1]}")
                        _, im = ismember(label[:, :, first:last], regions.id)
                        label[:, :, first:last] = np.reshape(im, label[:, :, first:last].shape)
                        if last == label.shape[-1]:
                            break
                        first += ncols
                    label = label.astype(dtype=np.uint16)
                    _logger.info("Saving npz, this can take a long time")
                else:
                    _, im = ismember(label, regions.id)
                    label = np.reshape(im.astype(np.uint16), label.shape)
                np.savez_compressed(file_label_remap, label)
                _logger.info(f"Cached remapping file {file_label_remap} ...")
            # loads the files
            label = self._read_volume(file_label_remap)
            image = self._read_volume(file_image)

        super().__init__(image, label, dxyz, regions, ibregma,
                         dims2xyz=dims2xyz, xyz2dims=xyz2dims)

    @staticmethod
    def _read_volume(file_volume):
        if file_volume.suffix == '.nrrd':
            volume, _ = nrrd.read(file_volume, index_order='C')  # ml, dv, ap
            # we want the coronal slice to be the most contiguous
            volume = np.transpose(volume, (2, 0, 1))  # image[iap, iml, idv]
        elif file_volume.suffix == '.npz':
            volume = np.load(file_volume)['arr_0']
        return volume

    def xyz2ccf(self, xyz, ccf_order='mlapdv', mode='raise'):
        """
        Converts coordinates to the CCF coordinates, which is assumed to be the cube indices
        times the spacing.
        :param xyz: mlapdv coordinates in meters, origin Bregma
        :param ccf_order: order that you want values returned 'mlapdv' (ibl) or 'apdvml'
        (Allen mcc vertices)
        :param mode: {‘raise’, 'clip', 'wrap'} determines what to do when determined index lies outside the atlas volume
                     'raise' will raise a ValueError
                     'clip' will replace the index with the closest index inside the volume
                     'wrap' will wrap around to the other side of the volume. This is only here for legacy reasons
        :return: coordinates in CCF space um, origin is the front left top corner of the data
        volume, order determined by ccf_order
        """
        ordre = self._ccf_order(ccf_order)
        ccf = self.bc.xyz2i(xyz, round=False, mode=mode) * float(self.res_um)
        return ccf[..., ordre]

    def ccf2xyz(self, ccf, ccf_order='mlapdv'):
        """
        Converts coordinates from the CCF coordinates, which is assumed to be the cube indices
        times the spacing.
        :param ccf coordinates in CCF space in um, origin is the front left top corner of the data
        volume
        :param ccf_order: order of ccf coordinates given 'mlapdv' (ibl) or 'apdvml'
        (Allen mcc vertices)
        :return: xyz: mlapdv coordinates in m, origin Bregma
        """
        ordre = self._ccf_order(ccf_order, reverse=True)
        return self.bc.i2xyz((ccf[..., ordre] / float(self.res_um)))

    @staticmethod
    def _ccf_order(ccf_order, reverse=False):
        """
        Returns the mapping to go from CCF coordinates order to the brain atlas xyz
        :param ccf_order: 'mlapdv' or 'apdvml'
        :param reverse: defaults to False.
            If False, returns from CCF to brain atlas
            If True, returns from brain atlas to CCF
        :return:
        """
        if ccf_order == 'mlapdv':
            return [0, 1, 2]
        elif ccf_order == 'apdvml':
            if reverse:
                return [2, 0, 1]
            else:
                return [1, 2, 0]
        else:
            ValueError("ccf_order needs to be either 'mlapdv' or 'apdvml'")

    def compute_regions_volume(self):
        """
        Sums the number of voxels in the labels volume for each region.
        Then compute volumes for all of the levels of hierarchy in cubic mm.
        :return:
        """
        nr = self.regions.id.shape[0]
        count = np.bincount(self.label.flatten(), minlength=nr)
        self.regions.compute_hierarchy()
        self.regions.volume = np.zeros_like(count)
        for i in np.arange(nr):
            if count[i] == 0:
                continue
            self.regions.volume[np.unique(self.regions.hierarchy[:, i])] += count[i]
        self.regions.volume = self.regions.volume * (self.res_um / 1e3) ** 3


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


def MRITorontoAtlas(*args, **kwargs):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system. The MRI Toronto atlas defines a stretch along AP
    a squeeze along DV *and* a squeeze along ML. These are based on 12 p65 mice MRIs averaged.
    See: https://www.nature.com/articles/s41467-018-04921-2 DB has access to the dataset.
    :param res_um: 10, 25 or 50 um
    :return: atlas.BrainAtlas
    """
    ML_SCALE = 0.952
    DV_SCALE = 0.885  # multiplicative factor on DV dimension, determined from MRI->CCF transform
    AP_SCALE = 1.031  # multiplicative factor on AP dimension
    kwargs['scaling'] = np.array([ML_SCALE, AP_SCALE, DV_SCALE])
    return AllenAtlas(*args, **kwargs)


def _download_atlas_allen(file_image, FLAT_IRON_ATLAS_REL_PATH, par):
    """
    © 2015 Allen Institute for Brain Science. Allen Mouse Brain Atlas (2015)
    with region annotations (2017).
    Available from: http://download.alleninstitute.org/informatics-archive/current-release/
    mouse_ccf/annotation/
    See Allen Mouse Common Coordinate Framework Technical White Paper for details
    http://help.brain-map.org/download/attachments/8323525/
    Mouse_Common_Coordinate_Framework.pdf?version=3&modificationDate=1508178848279&api=v2
    """

    file_image.parent.mkdir(exist_ok=True, parents=True)

    template_url = ('http://download.alleninstitute.org/informatics-archive/'
                    'current-release/mouse_ccf/average_template')
    annotation_url = ('http://download.alleninstitute.org/informatics-archive/'
                      'current-release/mouse_ccf/annotation/ccf_2017')

    if file_image.name.split('_')[0] == 'average':
        url = template_url + '/' + file_image.name
    elif file_image.name.split('_')[0] == 'annotation':
        url = annotation_url + '/' + file_image.name
    else:
        raise ValueError('Unrecognized file image')

    cache_dir = Path(par.CACHE_DIR).joinpath(FLAT_IRON_ATLAS_REL_PATH)
    return http_download_file(url, target_dir=cache_dir)


class FlatMap(AllenAtlas):

    def __init__(self, flatmap='dorsal_cortex', res_um=25):
        """
        Avaiable flatmaps are currently 'dorsal_cortex', 'circles' and 'pyramid'
        :param flatmap:
        :param res_um:
        """
        super().__init__(res_um=res_um)
        self.name = flatmap
        if flatmap == 'dorsal_cortex':
            self._get_flatmap_from_file()
        elif flatmap == 'circles':
            from ibllib.atlas.flatmaps import circles
            if res_um != 25:
                raise NotImplementedError('Pyramid circles not implemented for resolution other than 25um')
            self.flatmap, self.ml_scale, self.ap_scale = circles(N=5, atlas=self, display='flat')
        elif flatmap == 'pyramid':
            from ibllib.atlas.flatmaps import circles
            if res_um != 25:
                raise NotImplementedError('Pyramid circles not implemented for resolution other than 25um')
            self.flatmap, self.ml_scale, self.ap_scale = circles(N=5, atlas=self, display='pyramid')

    def _get_flatmap_from_file(self):
        # gets the file in the ONE cache for the flatmap name in the property, downloads it if needed
        file_flatmap = self._get_cache_dir().joinpath(f'{self.name}_{self.res_um}.nrrd')
        if not file_flatmap.exists():
            file_flatmap.parent.mkdir(exist_ok=True, parents=True)
            aws.s3_download_file(f'atlas/{file_flatmap.name}', file_flatmap)
        self.flatmap, _ = nrrd.read(file_flatmap)

    def plot_flatmap(self, depth=0, volume='annotation', mapping='Allen', region_values=None, ax=None, **kwargs):
        """
        Displays the 2D image corresponding to the flatmap. If there are several depths, by default it
        will display the first one
        :param depth: index of the depth to display in the flatmap volume (the last dimension)
        :param volume:
        :param mapping:
        :param region_values:
        :param ax:
        :param kwargs:
        :return:
        """
        if self.flatmap.ndim == 3:
            inds = np.int32(self.flatmap[:, :, depth])
        else:
            inds = np.int32(self.flatmap[:, :])
        regions = self._get_mapping(mapping=mapping)[self.label.flat[inds]]
        if volume == 'annotation':
            im = self._label2rgb(regions)
        elif volume == 'value':
            im = region_values[regions]
        elif volume == 'boundary':
            im = self.compute_boundaries(regions)
        elif volume == 'image':
            im = self.image.flat[inds]
        if not ax:
            ax = plt.gca()

        return self._plot_slice(im, self.extent_flmap(), ax=ax, volume=volume, **kwargs)

    def extent_flmap(self):
        extent = np.r_[0, self.flatmap.shape[1], 0, self.flatmap.shape[0]]
        return extent


class FranklinPaxinosAtlas(BrainAtlas):
    """
    Instantiates an atlas.BrainAtlas corresponding to the Allen CCF at the given resolution
    using the IBL Bregma and coordinate system
    """

    def __init__(self, res_um=np.array([10, 100, 10]), scaling=np.array([1, 1, 1]), mock=False, hist_path=None):
        """
        :param res_um: 10, 25 or 50 um
        :param scaling: scale factor along ml, ap, dv for squeeze and stretch ([1, 1, 1])
        :param mock: for testing purpose
        :param hist_path
        :return: atlas.BrainAtlas
        """
        # TODO interpolate?
        par = one.params.get(silent=True)
        FLAT_IRON_ATLAS_REL_PATH = PurePosixPath('histology', 'ATLAS', 'Needles', 'FranklinPaxinos')
        LUT_VERSION = "v01"  # version 01 is the lateralized version
        regions = FranklinPaxinosRegions()
        xyz2dims = np.array([1, 0, 2])  # this is the c-contiguous ordering
        dims2xyz = np.array([1, 0, 2])
        # we use Bregma as the origin
        self.res_um = res_um
        ibregma = (PAXINOS_CCF_LANDMARKS_MLAPDV_UM['bregma'] / self.res_um)
        dxyz = self.res_um * 1e-6 * np.array([1, -1, -1]) * scaling
        if mock:
            image, label = [np.zeros((528, 456, 320), dtype=np.int16) for _ in range(2)]
            label[:, :, 100:105] = 1327  # lookup index for retina, id 304325711 (no id 1327)
        else:
            path_atlas = Path(par.CACHE_DIR).joinpath(FLAT_IRON_ATLAS_REL_PATH)
            file_image = hist_path or path_atlas.joinpath(f'average_template_{res_um[0]}_{res_um[1]}_{res_um[2]}.npz')
            # # get the image volume
            if not file_image.exists():
                path_atlas.mkdir(exist_ok=True, parents=True)
                aws.s3_download_file(f'atlas/FranklinPaxinos/{file_image.name}', str(file_image))
            # # get the remapped label volume
            file_label = path_atlas.joinpath(f'annotation_{res_um[0]}_{res_um[1]}_{res_um[2]}.npz')
            if not file_label.exists():
                path_atlas.mkdir(exist_ok=True, parents=True)
                aws.s3_download_file(f'atlas/FranklinPaxinos/{file_label.name}', str(file_label))

            file_label_remap = path_atlas.joinpath(f'annotation_{res_um[0]}_{res_um[1]}_{res_um[2]}_lut_{LUT_VERSION}.npz')

            if not file_label_remap.exists():
                label = self._read_volume(file_label).astype(dtype=np.int32)
                _logger.info("computing brain atlas annotations lookup table")
                # lateralize atlas: for this the regions of the left hemisphere have primary
                # keys opposite to to the normal ones
                lateral = np.zeros(label.shape[xyz2dims[0]])
                lateral[int(np.floor(ibregma[0]))] = 1
                lateral = np.sign(np.cumsum(lateral)[np.newaxis, :, np.newaxis] - 0.5)
                label = label * lateral.astype(np.int32)
                _, im = ismember(label, regions.id)
                label = np.reshape(im.astype(np.uint16), label.shape)
                np.savez_compressed(file_label_remap, label)
                _logger.info(f"Cached remapping file {file_label_remap} ...")
            # loads the files
            label = self._read_volume(file_label_remap)
            image = self._read_volume(file_image)

        super().__init__(image, label, dxyz, regions, ibregma,
                         dims2xyz=dims2xyz, xyz2dims=xyz2dims)

    @staticmethod
    def _read_volume(file_volume):
        if file_volume.suffix == '.nrrd':
            volume, _ = nrrd.read(file_volume, index_order='C')  # ml, dv, ap
            # we want the coronal slice to be the most contiguous
            volume = np.transpose(volume, (2, 0, 1))  # image[iap, iml, idv]
        elif file_volume.suffix == '.npz':
            volume = np.load(file_volume)['arr_0']
        return volume

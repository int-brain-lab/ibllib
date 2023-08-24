"""Techniques to project the brain volume onto 2D images for visualisation purposes."""
from functools import lru_cache
import logging
import json

import nrrd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from iblutil.util import Bunch
from iblutil.io.hashfile import md5
import one.remote.aws as aws

from ibllib.atlas.atlas import AllenAtlas


_logger = logging.getLogger(__name__)


class FlatMap(AllenAtlas):
    """The Allen Atlas flatmap.

    FIXME Document! How are these flatmaps determined? Are they related to the Swansan atlas or is
     that something else?
    """

    def __init__(self, flatmap='dorsal_cortex', res_um=25):
        """
        Available flatmaps are currently 'dorsal_cortex', 'circles' and 'pyramid'
        :param flatmap:
        :param res_um:
        """
        super().__init__(res_um=res_um)
        self.name = flatmap
        if flatmap == 'dorsal_cortex':
            self._get_flatmap_from_file()
        elif flatmap == 'circles':
            if res_um != 25:
                raise NotImplementedError('Pyramid circles not implemented for resolution other than 25um')
            self.flatmap, self.ml_scale, self.ap_scale = circles(N=5, atlas=self, display='flat')
        elif flatmap == 'pyramid':
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
        Displays the 2D image corresponding to the flatmap.

        If there are several depths, by default it will display the first one.

        Parameters
        ----------
        depth : int
            Index of the depth to display in the flatmap volume (the last dimension).
        volume : {'image', 'annotation', 'boundary', 'value'}
            - 'image' - Allen image volume.
            - 'annotation' - Allen annotation volume.
            - 'boundary' - outline of boundaries between all regions.
            - 'volume' - custom volume, must pass in volume of shape BrainAtlas.image.shape as
               regions_value argument.
        mapping : str, default='Allen'
            The brain region mapping to use.
        region_values : numpy.array
            An array the shape of the brain atlas image containing custom region values. Used when
            `volume` value is 'volume'.
        ax : matplotlib.pyplot.Axes, optional
            A set of axes to plot to.
        **kwargs
            See matplotlib.pyplot.imshow.

        Returns
        -------
        matplotlib.pyplot.Axes
            The plotted image axes.
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
        else:
            raise ValueError(f'Volume type "{volume}" not supported')
        if not ax:
            ax = plt.gca()

        return self._plot_slice(im, self.extent_flmap(), ax=ax, volume=volume, **kwargs)

    def extent_flmap(self):
        """
        Returns the boundary coordinates of the flat map.

        Returns
        -------
        numpy.array
            The bounding coordinates of the flat map image, specified as (left, right, bottom, top).
        """
        extent = np.r_[0, self.flatmap.shape[1], 0, self.flatmap.shape[0]]
        return extent


@lru_cache(maxsize=1, typed=False)
def circles(N=5, atlas=None, display='flat'):
    """
    :param N: number of circles
    :param atlas: brain atlas at 25 m
    :param display: "flat" or "pyramid"
    :return: 2D map of indices, ap_coordinate, ml_coordinate
    """
    atlas = atlas if atlas else AllenAtlas()

    sz = np.array([])
    level = np.array([])
    for k in np.arange(N):
        nlast = 2000  # 25 um for 5mm diameter
        n = int((k + 1) * nlast / N)
        r = .4 * (k + 1) / N
        theta = (np.linspace(0, 2 * np.pi, n) + np.pi / 2)
        sz = np.r_[sz, r * np.exp(1j * theta)]
        level = np.r_[level, theta * 0 + k]

    atlas.compute_surface()
    iy, ix = np.where(~np.isnan(atlas.top))
    centroid = np.array([np.mean(iy), np.mean(ix)])
    xlim = np.array([np.min(ix), np.max(ix)])
    ylim = np.array([np.min(iy), np.max(iy)])

    s = Bunch(
        x=np.real(sz) * np.diff(xlim) + centroid[1],
        y=np.imag(sz) * np.diff(ylim) + centroid[0],
        level=level,
        distance=level * 0,
    )

    # compute the overall linear distance for each circle
    d0 = 0
    for lev in np.unique(s['level']):
        ind = s['level'] == lev
        diff = np.abs(np.diff(s['x'][ind] + 1j * s['y'][ind]))
        s['distance'][ind] = np.cumsum(np.r_[0, diff]) + d0
        d0 = s['distance'][ind][-1]

    fcn = interp1d(s['distance'], s['x'] + 1j * s['y'], fill_value='extrap')
    d = np.arange(0, np.ceil(s['distance'][-1]))

    s_ = Bunch({
        'x': np.real(fcn(d)),
        'y': np.imag(fcn(d)),
        'level': interp1d(s['distance'], level, kind='nearest')(d),
        'distance': d
    })

    if display == 'flat':
        ih = np.arange(atlas.bc.nz)
        iw = np.arange(s_['distance'].size)
        image_map = np.zeros((ih.size, iw.size), dtype=np.int32)
        iw, ih = np.meshgrid(iw, ih)
        # i2d = np.ravel_multi_index((ih[:], iw[:]), image_map.shape)
        iml, _ = np.meshgrid(np.round(s_.x).astype(np.int32), np.arange(atlas.bc.nz))
        iap, idv = np.meshgrid(np.round(s_.y).astype(np.int32), np.arange(atlas.bc.nz))
        i3d = atlas._lookup_inds(np.c_[iml.flat, iap.flat, idv.flat])
        i3d = np.reshape(i3d, [atlas.bc.nz, s_['x'].size])
        image_map[ih, iw] = i3d

    elif display == 'pyramid':
        for i in np.flipud(np.arange(N)):
            ind = s_['level'] == i
            dtot = s_['distance'][ind]
            dtot = dtot - np.mean(dtot)
            if i == N - 1:
                ipx = np.arange(np.floor(dtot[0]), np.ceil(dtot[-1]) + 1)
                nh = atlas.bc.nz * N
                X0 = int(ipx[-1])
                image_map = np.zeros((nh, ipx.size), dtype=np.int32)

            iw = np.arange(np.sum(ind))
            iw = np.int32(iw - np.mean(iw) + X0)
            ih = atlas.bc.nz * i + np.arange(atlas.bc.nz)

            iw, ih = np.meshgrid(iw, ih)
            iml, _ = np.meshgrid(np.round(s_.x[ind]).astype(np.int32), np.arange(atlas.bc.nz))
            iap, idv = np.meshgrid(np.round(s_.y[ind]).astype(np.int32), np.arange(atlas.bc.nz))
            i3d = atlas._lookup_inds(np.c_[iml.flat, iap.flat, idv.flat])
            i3d = np.reshape(i3d, [atlas.bc.nz, s_['x'][ind].size])
            image_map[ih, iw] = i3d
    x, y = (atlas.bc.i2x(s.x), atlas.bc.i2y(s.y))
    return image_map, x, y
    # if display == 'flat':
    #     fig, ax = plt.subplots(2, 1, figsize=(16, 5))
    # elif display == 'pyramid':
    #     fig, ax = plt.subplots(1, 2, figsize=(14, 12))
    # ax[0].imshow(ba._label2rgb(ba.label.flat[image_map]), origin='upper')
    # ax[1].imshow(ba.top)
    # ax[1].plot(centroid[1], centroid[0], '*')
    # ax[1].plot(s.x, s.y)


def swanson(filename="swanson2allen.npz"):
    """
    FIXME Document! Which publication to reference? Are these specifically for flat maps?
     Shouldn't this be made into an Atlas class with a mapping or scaling applied?

    Parameters
    ----------
    filename

    Returns
    -------

    """
    # filename could be "swanson2allen_original.npz", or "swanson2allen.npz" for remapped indices to match
    # existing labels in the brain atlas
    OLD_MD5 = [
        'bb0554ecc704dd4b540151ab57f73822',  # version 2022-05-02 (remapped)
        '7722c1307cf9a6f291ad7632e5dcc88b',  # version 2022-05-09 (removed wolf pixels and 2 artefact regions)
    ]
    npz_file = AllenAtlas._get_cache_dir().joinpath(filename)
    if not npz_file.exists() or md5(npz_file) in OLD_MD5:
        npz_file.parent.mkdir(exist_ok=True, parents=True)
        _logger.info(f'downloading swanson image from {aws.S3_BUCKET_IBL} s3 bucket...')
        aws.s3_download_file(f'atlas/{npz_file.name}', npz_file)
    s2a = np.load(npz_file)['swanson2allen']  # inds contains regions ids
    return s2a


def swanson_json(filename="swansonpaths.json", remap=True):
    """
    Vectorized version of the swanson bitmap file. The vectorized version was generated from swanson() using matlab
    contour to find the paths for each region. The paths for each region were then simplified using the
    Ramer Douglas Peucker algorithm https://rdp.readthedocs.io/en/latest/

    Parameters
    ----------
    filename
    remap

    Returns
    -------

    """
    OLD_MD5 = ['97ccca2b675b28ba9b15ca8af5ba4111',  # errored map with FOTU and CUL4, 5 mixed up
               '56daa7022b5e03080d8623814cda6f38',  # old md5 of swanson json without CENT and PTLp
               # and CUL4 split (on s3 called swansonpaths_56daa.json)
               'f848783954883c606ca390ceda9e37d2']

    json_file = AllenAtlas._get_cache_dir().joinpath(filename)
    if not json_file.exists() or md5(json_file) in OLD_MD5:
        json_file.parent.mkdir(exist_ok=True, parents=True)
        _logger.info(f'downloading swanson paths from {aws.S3_BUCKET_IBL} s3 bucket...')
        aws.s3_download_file(f'atlas/{json_file.name}', json_file, overwrite=True)

    with open(json_file) as f:
        sw_json = json.load(f)

    # The swanson contains regions that are children of regions contained within the Allen
    # annotation volume. Here we remap these regions to the parent that is contained with the
    # annotation volume
    if remap:
        id_map = {391: [392, 393, 394, 395, 396],
                  474: [483, 487],
                  536: [537, 541],
                  601: [602, 603, 604, 608],
                  622: [624, 625, 626, 627, 628, 629, 630, 631, 632, 634, 635, 636, 637, 638],
                  686: [687, 688, 689],
                  708: [709, 710],
                  721: [723, 724, 726, 727, 729, 730, 731],
                  740: [741, 742, 743],
                  758: [759, 760, 761, 762],
                  771: [772, 773],
                  777: [778, 779, 780],
                  788: [789, 790, 791, 792],
                  835: [836, 837, 838],
                  891: [894, 895, 896, 897, 898, 900, 901, 902],
                  926: [927, 928],
                  949: [950, 951, 952, 953, 954],
                  957: [958, 959, 960, 961, 962],
                  999: [1000, 1001],
                  578: [579, 580]}

        rev_map = {}
        for k, vals in id_map.items():
            for v in vals:
                rev_map[v] = k

        for sw in sw_json:
            sw['thisID'] = rev_map.get(sw['thisID'], sw['thisID'])

    return sw_json


@lru_cache(maxsize=None)
def _swanson_labels_positions(thres=20000):
    """
    Computes label positions to overlay on the Swanson flatmap.

    Parameters
    ----------
    thres : int, default=20000
        The number of pixels above which a region is labeled.

    Returns
    -------
    dict of str
        A map of brain acronym to a tuple of x y coordinates.
    """
    s2a = swanson()
    iw, ih = np.meshgrid(np.arange(s2a.shape[1]), np.arange(s2a.shape[0]))
    # compute the center of mass of all regions (fast enough to do on the fly)
    bc = np.maximum(1, np.bincount(s2a.flatten()))
    cmw = np.bincount(s2a.flatten(), weights=iw.flatten()) / bc
    cmh = np.bincount(s2a.flatten(), weights=ih.flatten()) / bc
    bc[0] = 1

    NWH, NWW = (200, 600)
    h, w = s2a.shape
    labels = {}
    for ilabel in np.where(bc > thres)[0]:
        x, y = (cmw[ilabel], cmh[ilabel])
        # the polygon is convex and the label is outside. Dammit !!!
        if s2a[int(y), int(x)] != ilabel:
            # find the nearest point to the center of mass
            ih, iw = np.where(s2a == ilabel)
            iimin = np.argmin(np.abs((x - iw) + 1j * (y - ih)))
            # get the center of mass of a window around this point
            sh = np.arange(np.maximum(0, ih[iimin] - NWH), np.minimum(ih[iimin] + NWH, h))
            sw = np.arange(np.maximum(0, iw[iimin] - NWW), np.minimum(iw[iimin] + NWW, w))
            roi = s2a[sh][:, sw] == ilabel
            roi = roi / np.sum(roi)
            # ax.plot(x, y, 'k+')
            # ax.plot(iw[iimin], ih[iimin], '*k')
            x = sw[np.searchsorted(np.cumsum(np.sum(roi, axis=0)), .5) - 1]
            y = sh[np.searchsorted(np.cumsum(np.sum(roi, axis=1)), .5) - 1]
            # ax.plot(x, y, 'r+')
        labels[ilabel] = (x, y)
    return labels

"""
Module that hold techniques to project the brain volume onto 2D images for visualisation purposes
"""
from functools import lru_cache
import logging
import json

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm

from iblutil.numerical import ismember
from iblutil.util import Bunch
from iblutil.io.hashfile import md5
import one.remote.aws as aws

from ibllib.atlas.atlas import AllenAtlas, BrainRegions
from ibllib.atlas.plots import plot_polygon, plot_polygon_with_hole, coords_for_poly_hole


_logger = logging.getLogger(__name__)


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


def swanson_json(filename="swansonpaths.json"):

    OLD_MD5 = ['f848783954883c606ca390ceda9e37d2']

    json_file = AllenAtlas._get_cache_dir().joinpath(filename)
    if not json_file.exists() or md5(json_file) in OLD_MD5:
        json_file.parent.mkdir(exist_ok=True, parents=True)
        _logger.info(f'downloading swanson paths from {aws.S3_BUCKET_IBL} s3 bucket...')
        aws.s3_download_file(f'atlas/{json_file.name}', json_file)

    with open(json_file) as f:
        sw_json = json.load(f)

    return sw_json


def plot_swanson_vector(acronyms=None, values=None, ax=None, hemisphere=None, br=None, orientation='landscape',
                        empty_color='silver', vmin=None, vmax=None, cmap='cividis', **kwargs):

    br = BrainRegions() if br is None else br
    br.compute_hierarchy()

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_axis_off()

    if acronyms is not None:
        ibr, vals = br.propagate_down(acronyms, values)
        colormap = cm.get_cmap(cmap)
        vmin = vmin or np.nanmin(vals)
        vmax = vmax or np.nanmax(vals)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        rgba_color = colormap(norm(vals), bytes=True)

    sw = swanson()
    sw_json = swanson_json()

    for i, reg in enumerate(sw_json):

        if acronyms is None:
            color = br.rgba[br.mappings['Swanson'][reg['thisID']]] / 255
        else:
            idx = np.where(ibr == reg['thisID'])[0]
            color = rgba_color[idx[0]] / 255 if len(idx) > 0 else empty_color

        coords = reg['coordsReg']

        if reg['hole']:
            vertices, codes = coords_for_poly_hole(coords)
            if orientation == 'portrait':
                vertices[:, [0, 1]] = vertices[:, [1, 0]]
                plot_polygon_with_hole(ax, vertices, codes, color, **kwargs)
                if hemisphere is not None:
                    color_inv = color if hemisphere == 'mirror' else empty_color
                    vertices_inv = np.copy(vertices)
                    vertices_inv[:, 0] = -1 * vertices_inv[:, 0] + (sw.shape[0] * 2)
                    plot_polygon_with_hole(ax, vertices_inv, codes, color_inv, **kwargs)
            else:
                plot_polygon_with_hole(ax, vertices, codes, color, **kwargs)
                if hemisphere is not None:
                    color_inv = color if hemisphere == 'mirror' else empty_color
                    vertices_inv = np.copy(vertices)
                    vertices_inv[:, 1] = -1 * vertices_inv[:, 1] + (sw.shape[0] * 2)
                    plot_polygon_with_hole(ax, vertices_inv, codes, color_inv, **kwargs)
        else:
            coords = [coords] if type(coords) == dict else coords
            for c in coords:

                if orientation == 'portrait':
                    xy = np.c_[c['y'], c['x']]
                    plot_polygon(ax, xy, color, **kwargs)
                    if hemisphere is not None:
                        color_inv = color if hemisphere == 'mirror' else empty_color
                        xy_inv = np.copy(xy)
                        xy_inv[:, 0] = -1 * xy_inv[:, 0] + (sw.shape[0] * 2)
                        plot_polygon(ax, xy_inv, color_inv, **kwargs)
                else:
                    xy = np.c_[c['x'], c['y']]
                    plot_polygon(ax, xy, color, **kwargs)
                    if hemisphere is not None:
                        color_inv = color if hemisphere == 'mirror' else empty_color
                        xy_inv = np.copy(xy)
                        xy_inv[:, 1] = -1 * xy_inv[:, 1] + (sw.shape[0] * 2)
                        plot_polygon(ax, xy_inv, color_inv, **kwargs)

    if orientation == 'portrait':
        ax.set_ylim(0, sw.shape[1])
        if hemisphere is None:
            ax.set_xlim(0, sw.shape[0])
        else:
            ax.set_xlim(0, 2 * sw.shape[0])
    else:
        ax.set_xlim(0, sw.shape[1])
        if hemisphere is None:
            ax.set_ylim(0, sw.shape[0])
        else:
            ax.set_ylim(0, 2 * sw.shape[0])

    def format_coord(x, y):
        ind = sw[int(y), int(x)]
        ancestors = br.ancestors(br.id[ind])['acronym']
        return f'sw-{ind}, x={x:1.4f}, y={y:1.4f}, aid={br.id[ind]}-{br.acronym[ind]} \n {ancestors}'

    ax.format_coord = format_coord

    ax.invert_yaxis()
    ax.set_aspect('equal')


def plot_swanson(acronyms=None, values=None, ax=None, hemisphere=None, br=None,
                 orientation='landscape', annotate=False, empty_color='silver', **kwargs):
    """
    Displays the 2D image corresponding to the swanson flatmap.
    This case is different from the others in the sense that only a region maps to another regions, there
    is no correspondency from the spatial 3D coordinates.
    :param acronyms:
    :param values:
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both' or 'mirror'
    :param br: ibllib.atlas.BrainRegions object
    :param ax: matplotlib axis object to plot onto
    :param orientation: 'landscape' (default) or 'portrait'
    :param annotate: (False) if True, labels regions with acronyms
    :param empty_color: (grey) matplotlib color code or rgb_a int8 tuple defining the filling
     of brain regions not provided. Defaults to 'silver'
    :param kwargs: arguments for imshow
    :return:
    """
    mapping = 'Swanson'
    br = BrainRegions() if br is None else br
    br.compute_hierarchy()
    s2a = swanson()
    # both hemishpere
    if hemisphere == 'both':
        _s2a = s2a + np.sum(br.id > 0)
        _s2a[s2a == 0] = 0
        _s2a[s2a == 1] = 1
        s2a = np.r_[s2a, np.flipud(_s2a)]
        mapping = 'Swanson-lr'
    elif hemisphere == 'mirror':
        s2a = np.r_[s2a, np.flipud(s2a)]
    if orientation == 'portrait':
        s2a = np.transpose(s2a)
    if acronyms is None:
        regions = br.mappings[mapping][s2a]
        im = br.rgba[regions]
        iswan = None
    else:
        ibr, vals = br.propagate_down(acronyms, values)
        # we now have the mapped regions and aggregated values, map values onto swanson map
        iswan, iv = ismember(s2a, ibr)
        im = np.zeros_like(s2a, dtype=np.float32)
        im[iswan] = vals[iv]
        im[~iswan] = np.nan
    if not ax:
        ax = plt.gca()
        ax.set_axis_off()  # unless provided we don't need scales here
    ax.imshow(im, **kwargs)
    # overlay the boundaries if value plot
    imb = np.zeros((*s2a.shape[:2], 4), dtype=np.uint8)
    # fill in the empty regions with the blank regions colours if necessary
    if iswan is not None:
        imb[~iswan] = (np.array(matplotlib.colors.to_rgba(empty_color)) * 255).astype('uint8')
    imb[s2a == 0] = 255
    # imb[s2a == 1] = np.array([167, 169, 172, 255])
    imb[s2a == 1] = np.array([0, 0, 0, 255])
    ax.imshow(imb)
    if annotate:
        annotate_swanson(ax=ax, orientation=orientation, br=br)

    # provides the mean to see the region on axis
    def format_coord(x, y):
        ind = s2a[int(y), int(x)]
        ancestors = br.ancestors(br.id[ind])['acronym']
        return f'sw-{ind}, x={x:1.4f}, y={y:1.4f}, aid={br.id[ind]}-{br.acronym[ind]} \n {ancestors}'

    ax.format_coord = format_coord
    return ax


@lru_cache(maxsize=None)
def _swanson_labels_positions():
    """
    This functions computes label positions to overlay on the Swanson flatmap
    :return: dictionary where keys are acronyms
    """
    NPIX_THRESH = 20000  # number of pixels above which region is labeled
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
    for ilabel in np.where(bc > NPIX_THRESH)[0]:
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


def annotate_swanson(ax, acronyms=None, orientation='landscape', br=None, **kwargs):
    """
    Display annotations on the flatmap
    :param ax:
    :param acronyms: (None) list or np.array of acronyms or allen region ids. If None plot all.
    :param orientation:
    :param br: BrainRegions object
    :param kwargs: arguments for the annotate function
    :return:
    """
    br = br or BrainRegions()
    if acronyms is None:
        indices = np.arange(br.id.size)
    else:  # tech debt: here in fact we should remap and compute labels for hierarchical regions
        aids = br.parse_acronyms_argument(acronyms)
        _, indices, _ = np.intersect1d(br.id, br.remap(aids, 'Swanson-lr'), return_indices=True)
    labels = _swanson_labels_positions()
    for ilabel in labels:
        # do not display uwanted labels
        if ilabel not in indices:
            continue
        # rotate the labels if the dislay is in portrait mode
        xy = reversed(labels[ilabel]) if orientation == 'portrait' else labels[ilabel]
        ax.annotate(br.acronym[ilabel], xy=xy, ha='center', va='center', **kwargs)

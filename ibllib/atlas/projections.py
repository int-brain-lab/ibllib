"""
Module that hold techniques to project the brain volume onto 2D images for visualisation purposes
"""
from functools import lru_cache
from ibllib.atlas import AllenAtlas
import numpy as np
from brainbox.core import Bunch
from scipy.interpolate import interp1d


@lru_cache(maxsize=1, typed=False)
def circles(N=5, atlas=None, display='flat'):
    """

    :param N:
    :param atlas:
    :param display: "flat" or "pyramid"
    :return:
    """
    atlas = atlas if atlas else AllenAtlas()

    sz = np.array([])
    level = np.array([])
    for k in np.arange(N):
        nlast = 2000  # 25 um for 5mm diameter
        n = int((k + 1) * nlast / N)
        print(n, k)
        r = .4 * (k + 1) / N
        theta = np.linspace(0, 2 * np.pi, n) - np.pi / 2
        sz = np.r_[sz, r * np.exp(1j * theta)]
        level = np.r_[level, theta * 0 + k]

    iy, ix = np.where(~np.isnan(atlas.top))
    centroid = np.array([np.mean(iy), np.mean(ix)])
    xlim = np.array([np.min(ix), np.max(ix)])
    ylim = np.array([np.min(iy), np.max(iy)])

    s = Bunch(
        x=np.real(sz) * np.diff(xlim) + centroid[1],
        y=np.imag(sz) * np.diff(ylim) + centroid[0]
    )
    s['distance'] = np.r_[0, np.cumsum(np.abs(np.diff(s['x'] + 1j * s['y'])))]

    fcn = interp1d(s['distance'], s['x'] + 1j * s['y'])

    d = np.arange(0, np.ceil(s['distance'][-1]))

    s_ = Bunch({
        'x': np.real(fcn(d)),
        'y': np.imag(fcn(d)),
        'level': interp1d(s['distance'], level, kind='nearest')(d)
    })
    s_['distance'] = np.r_[0, np.cumsum(np.abs(np.diff(s_['x'] + 1j * s_['y'])))]

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
    return image_map
    # if display == 'flat':
    #     fig, ax = plt.subplots(2, 1, figsize=(16, 5))
    # elif display == 'pyramid':
    #     fig, ax = plt.subplots(1, 2, figsize=(14, 12))
    # ax[0].imshow(ba._label2rgb(ba.label.flat[image_map]), origin='upper')
    # ax[1].imshow(ba.top)
    # ax[1].plot(centroid[1], centroid[0], '*')
    # ax[1].plot(s.x, s.y)

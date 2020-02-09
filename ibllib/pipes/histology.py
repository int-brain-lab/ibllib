from pathlib import Path
import inspect
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ibllib.ephys.neuropixel import SITES_COORDINATES
from oneibl.one import ONE
import alf.io
import ibllib.atlas as atlas
from ibllib.ephys.spikes import probes_description as extract_probes

_logger = logging.getLogger('ibllib')
atlas_params = {
    'PATH_ATLAS': str('/datadisk/BrainAtlas/ATLASES/Allen'),
    'FILE_REGIONS':
        str(Path(inspect.getfile(atlas.AllenAtlas)).parent.joinpath('allen_structure_tree.csv')),
    'INDICES_BREGMA': list(np.array([1140 - (570 + 3.9), 540, 0 + 33.2]))
}
# origin Allen left, front, up
brat = atlas.AllenAtlas(res_um=25, par=atlas_params)


def get_picked_tracks(histology_path, glob_pattern="*_pts_transformed.csv"):
    """
    This outputs reads in the Lasagna output and converts the picked tracks in the IBL coordinates
    :param histology_path:
    :return:
    """
    xyzs = []
    files_track = list(histology_path.rglob(glob_pattern))
    for file_track in files_track:
        print(file_track)
        # apmldv in the histology file is flipped along x and y directions
        ixiyiz = np.loadtxt(file_track, delimiter=',')[:, [1, 0, 2]]
        ixiyiz[:, 1] = 527 - ixiyiz[:, 1]
        ixiyiz = ixiyiz[np.argsort(ixiyiz[:, 2]), :]
        xyz = brat.bc.i2xyz(ixiyiz)
        # xyz[:, 0] = - xyz[:, 0]
        xyzs.append(xyz)

    return {'files': files_track, 'xyz': xyzs}


def get_micro_manipulator_data(subject, one=None, force_extract=False):
    """
    Looks for all ephys sessions for a given subject and get the probe micro-manipulator
    trajectories.
    If probes ALF object not on flat-iron, attempts to perform the extraction from meta-data
    and task settings file.
    """
    if not one:
        one = ONE()

    eids, sessions = one.search(subject=subject, task_protocol='ephys', details=True)
    dtypes = ['probes.description', 'probes.trajectory', ]
    probes = alf.io.AlfBunch({})
    for ses in sessions:
        sess_path = Path(ses['local_path'])
        probe = None
        if not force_extract:
            probe = one.load_object(ses['url'], 'probes')
        if not probe:
            _logger.warning(f"Re-extraction probe info for {sess_path}")
            dtypes = ['_iblrig_taskSettings.raw', 'ephysData.raw.meta']
            raw_files = one.load(ses['url'], dataset_types=dtypes, download_only=True)
            if all([rf is None for rf in raw_files]):
                _logger.warning(f"no raw settings files nor ephys data found for"
                                f" {ses['local_path']}. Skip this session.")
                continue
            extract_probes(sess_path, bin_exists=False)
            probe = alf.io.load_object(sess_path.joinpath('alf'), 'probes')
        one.load(ses['url'], dataset_types='channels.localCoordinates', download_only=True)
        # get for each insertion the sites local mapping: if not found assumes checkerboard pattern
        probe['sites_coordinates'] = []
        for prb in probe.description:
            chfile = Path(ses['local_path']).joinpath('alf', prb['label'],
                                                      'channels.localCoordinates.npy')
            if chfile.exists():
                probe['sites_coordinates'].append(np.load(chfile))
            else:
                _logger.warning(f"no channel.localCoordinates found for {ses['local_path']}."
                                f"Assumes checkerboard pattern")
                probe['sites_coordinates'].append(SITES_COORDINATES)
        # put the session information in there
        probe['session'] = [ses] * len(probe.description)
        probes = probes.append(probe)
    return probes


def plot_merged_result(probes, tracks, index=None):
    pass
    # fig, ax = plt.subplots(1, 2)
    # # plot the atlas image
    # brat.plot_cslice(np.mean(xyz[:, 1]) * 1e3, ax=ax[0])
    # brat.plot_sslice(np.mean(xyz[:, 0]) * 1e3, ax=ax[1])
    # # plot the full tracks
    # ax[0].plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3)
    # ax[1].plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e3)
    # # plot the sites
    # ax[0].plot(xyz_channels[:, 0] * 1e3, xyz_channels[:, 2] * 1e3, 'y*')
    # ax[1].plot(xyz_channels[:, 1] * 1e3, xyz_channels[:, 2] * 1e3, 'y*')


def plot2d_all(probes, tracks):
    """
    Plot all tracks on a single 2d slice
    :param probes:
    :param tracks:
    :return:
    """
    plt.figure()
    axs = brat.plot_sslice(brat.bc.i2x(190) * 1e3, cmap=plt.get_cmap('bone'))
    plt.figure()
    axc = brat.plot_cslice(brat.bc.i2y(350) * 1e3)
    for xyz in tracks['xyz']:
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3, 'b')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e3, 'b')
    for trj in probes['trajectory']:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brat)
        xyz = ins.xyz
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3, 'r')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e3, 'r')


def plot3d_all(probes, tracks):
    """
    Plot all tracks on a single 2d slice
    :param probes:
    :param tracks:
    :return:
    """
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(brat.label)
    mlab.pipeline.iso_surface(src, contours=[0.5, ], opacity=0.3)

    pts = []
    for xyz in tracks['xyz']:
        mlapdv = brat.bc.xyz2i(xyz)
        pts.append(mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2], line_width=3))

    plt_trj = []
    for trj in probes['trajectory']:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brat)
        mlapdv = brat.bc.xyz2i(ins.xyz)
        plt = mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2],
                          line_width=3, color=(1., 0., 1.))
        plt_trj.append(plt)


def get_brain_regions(xyz, channels=SITES_COORDINATES, display=False):
    """
    :param xyz:
    :param channels:
    :param DISPLAY:
    :return:
    """

    """
    this is the depth along the probe (from the first point which is the deepest labeled point)
    Due to the blockiness, depths may not be unique along the track so it has to be prepared
    """
    d = atlas.cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
    ind_depths = np.argsort(d)
    d = np.sort(d)
    iid = np.where(np.diff(d) >= 0)[0]
    ind_depths = ind_depths[iid]
    d = d[iid]

    """
    Interpolate channel positions along the probe depth and get brain locations
    """
    xyz_channels = np.zeros((channels.sitePositions.shape[0], 3))
    for m in np.arange(3):
        xyz_channels[:, m] = np.interp(channels.sitePositions[:, 1] / 1e6,
                                       d[ind_depths], xyz[ind_depths, m])
    brain_regions = brat.regions.get(brat.get_labels(xyz_channels))

    """
    Get the best linear fit probe trajectory using points cloud
    """
    track = atlas.Trajectory.fit(xyz)

    return brain_regions, track


def extract_brain_regions(session_path, display=False):
    # #################### edit those #############################
    file_track = Path(f'/datadisk/scratch/electrode_01_fit.csv')
    ses_path = Path('/datadisk/Data/Subjects/KS005/2019-08-29/001')
    # ##############################################################

    alf_path = ses_path.joinpath('alf')
    channels = alf.io.load_object(alf_path, 'channels')
    # probes = alf.io.load_object(alf_path, 'probes')  # this will be needed to compare planned
    DISPLAY = True

    brain_regions, track_fit = get_brain_regions(file_track, channels, display=DISPLAY)
    pd.DataFrame.from_dict(brain_regions).to_csv()

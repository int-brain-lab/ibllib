import matplotlib.pyplot as plt
import inspect

from pathlib import Path
import numpy as np
import pandas as pd

import alf.io
from ibllib.atlas import AllenAtlas, cart2sph, Trajectory

atlas_params = {
    'PATH_ATLAS': str('/datadisk/BrainAtlas/ATLASES/Allen'),
    'FILE_REGIONS':
        str(Path(inspect.getfile(AllenAtlas)).parent.joinpath('allen_structure_tree.csv')),
    'INDICES_BREGMA': list(np.array([1140 - (570 + 3.9), 540, 0 + 33.2]))
}
# origin Allen left, front, up
self = AllenAtlas(res_um=25, par=atlas_params)


def get_brain_regions(file_track, channels, display=False):
    """
    :param file_track:
    :param channels:
    :param DISPLAY:
    :return:
    """
    ixiyiz = np.loadtxt(file_track, delimiter=',')[:, [1, 0, 2]]  # apmldv in the histology file
    ixiyiz[:, 1] = 527 - ixiyiz[:, 1]  # one axis is swapped compared to binary file
    xyz = self.bc.i2xyz(ixiyiz)

    """
    this is the depth along the probe (from the first point which is the deepest labeled point)
    Due to the blockiness, depths may not be unique along the track so it has to be prepared
    """
    d = cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
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
    brain_regions = self.regions.get(self.get_labels(xyz_channels))

    """
    Get the best linear fit probe trajectory using points cloud
    """
    track = Trajectory.fit(xyz)

    if display:
        fig, ax = plt.subplots(1, 2)
        # plot the atlas image
        self.plot_cslice(np.mean(xyz[:, 1]) * 1e3, ax=ax[0])
        self.plot_sslice(np.mean(xyz[:, 0]) * 1e3, ax=ax[1])
        # plot the full tracks
        ax[0].plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3)
        ax[1].plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e3)
        # plot the sites
        ax[0].plot(xyz_channels[:, 0] * 1e3, xyz_channels[:, 2] * 1e3, 'y*')
        ax[1].plot(xyz_channels[:, 1] * 1e3, xyz_channels[:, 2] * 1e3, 'y*')

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

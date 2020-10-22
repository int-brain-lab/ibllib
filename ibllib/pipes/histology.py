from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np

from ibllib.ephys.neuropixel import SITES_COORDINATES
from oneibl.one import ONE
import alf.io
import ibllib.atlas as atlas
from ibllib.ephys.spikes import probes_description as extract_probes
from ibllib.dsp.utils import fcn_cosine
from ibllib.ephys.neuropixel import TIP_SIZE_UM
from ibllib.qc import base


_logger = logging.getLogger('ibllib')

# origin Allen left, front, up
brain_atlas = atlas.AllenAtlas(res_um=25)


def load_track_csv(file_track):
    """
    Loads a lasagna track and convert to IBL-ALlen coordinate framework
    :param file_track:
    :return: xyz
    """
    # apmldv in the histology file is flipped along y direction
    file_track = Path(file_track)
    if file_track.stat().st_size == 0:
        return np.array([])
    ixiyiz = np.loadtxt(file_track, delimiter=',')[:, [1, 0, 2]]
    ixiyiz[:, 1] = 527 - ixiyiz[:, 1]
    ixiyiz = ixiyiz[np.argsort(ixiyiz[:, 2]), :]
    xyz = brain_atlas.bc.i2xyz(ixiyiz)
    # xyz[:, 0] = - xyz[:, 0]
    return xyz


def get_picked_tracks(histology_path, glob_pattern="*_pts_transformed.csv"):
    """
    This outputs reads in the Lasagna output and converts the picked tracks in the IBL coordinates
    :param histology_path: Path object: folder path containing tracks
    :return: xyz coordinates in
    """
    xyzs = []
    histology_path = Path(histology_path)
    if histology_path.is_file():
        files_track = [histology_path]
    else:
        files_track = list(histology_path.rglob(glob_pattern))
    for file_track in files_track:
        xyzs.append(load_track_csv(file_track))
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


def plot2d_all(trajectories, tracks):
    """
    Plot all tracks on a single 2d slice
    :param trajectories: dictionary output of the Alyx REST query on trajectories
    :param tracks:
    :return:
    """
    plt.figure()
    axs = brain_atlas.plot_sslice(brain_atlas.bc.i2x(190), cmap=plt.get_cmap('bone'))
    plt.figure()
    axc = brain_atlas.plot_cslice(brain_atlas.bc.i2y(350))
    for xyz in tracks['xyz']:
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e6, 'b')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e6, 'b')
    for trj in trajectories:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brain_atlas)
        xyz = ins.xyz
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e6, 'r')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e6, 'r')


def plot3d_all(trajectories, tracks):
    """
    Plot all tracks on a single 2d slice
    :param trajectories: dictionary output of the Alyx REST query on trajectories
    :param tracks:
    :return:
    """
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(brain_atlas.label)
    mlab.pipeline.iso_surface(src, contours=[0.5, ], opacity=0.3)

    pts = []
    for xyz in tracks['xyz']:
        mlapdv = brain_atlas.bc.xyz2i(xyz)
        pts.append(mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2], line_width=3))

    plt_trj = []
    for trj in trajectories:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brain_atlas)
        mlapdv = brain_atlas.bc.xyz2i(ins.xyz)
        plt = mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2],
                          line_width=3, color=(1., 0., 1.))
        plt_trj.append(plt)


def interpolate_along_track(xyz_track, depths):
    """
    Get the coordinates of points along a track according to their distances from the first
    point.
    :param xyz_track: np.array [npoints, 3]. Usually the first point is the deepest
    :param depths: distance from the first point of the track, usually the convention is the
    deepest point is 0 and going up
    :return: xyz_channels
    """
    # from scipy.interpolate import interp1d
    # this is the cumulative distance from the lowest picked point (first)
    distance = np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz_track, axis=0) ** 2, axis=1))])
    xyz_channels = np.zeros((depths.shape[0], 3))
    for m in np.arange(3):
        xyz_channels[:, m] = np.interp(depths, distance, xyz_track[:, m])
        # xyz_channels[:, m] = interp1d(distance, xyz[:, m], kind='cubic')(chdepths / 1e6)
    # plt.figure()
    # plt.plot(xyz_track[:, 0] * 1e6, xyz_track[:, 2] * 1e6, 'k*'), plt.axis('equal')
    # plt.plot(xyz_channels[:, 0] * 1e6, xyz_channels[:, 2] * 1e6, '.'), plt.axis('equal')
    return xyz_channels


def get_brain_regions(xyz, channels_positions=SITES_COORDINATES, brain_atlas=brain_atlas):
    """
    :param xyz: numpy array of 3D coordinates corresponding to a picked track or a trajectory
    the deepest point is assumed to be the tip.
    :param channels_positions:
    :param brain_atlas:
    :return: brain_regions (associated to each channel),
             insertion (object atlas.Insertion, defining 2 points of entries
             (tip and end of probe))
    """
    """
    this is the depth along the probe (from the first point which is the deepest labeled point)
    Due to the blockiness, depths may not be unique along the track so it has to be prepared
    """
    xyz = xyz[np.argsort(xyz[:, 2]), :]
    d = atlas.cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
    indsort = np.argsort(d)
    xyz = xyz[indsort, :]
    d = d[indsort]
    iduplicates = np.where(np.diff(d) == 0)[0]
    xyz = np.delete(xyz, iduplicates, axis=0)
    d = np.delete(d, iduplicates, axis=0)

    assert np.all(np.diff(d) > 0), "Depths should be strictly increasing"

    # Get the probe insertion from the coordinates
    insertion = atlas.Insertion.from_track(xyz, brain_atlas)

    # Interpolate channel positions along the probe depth and get brain locations
    TIP_SIZE_UM = 200
    xyz_channels = interpolate_along_track(xyz, (channels_positions[:, 1] + TIP_SIZE_UM) / 1e6)

    # get the brain regions
    brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
    brain_regions['xyz'] = xyz_channels
    brain_regions['lateral'] = channels_positions[:, 0]
    brain_regions['axial'] = channels_positions[:, 1]
    assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1

    return brain_regions, insertion


def register_track(probe_id, picks=None, one=None, overwrite=False, channels=True):
    """
    Register the user picks to a probe in Alyx
    Here we update Alyx models on the database in 3 steps
    1) The user picks converted to IBL coordinates will be stored in the json field of the
    corresponding probe insertion models
    2) The trajectory computed from the histology track is created or patched
    3) Channel locations are set in the table
    """
    assert one
    # 0) if it's an empty track, create a null trajectory and exit
    if picks is None or picks.size == 0:
        tdict = {'probe_insertion': probe_id,
                 'x': None, 'y': None, 'z': None,
                 'phi': None, 'theta': None, 'depth': None, 'roll': None,
                 'provenance': 'Histology track',
                 'coordinate_system': 'IBL-Allen',
                 }
        brain_locations = None
        # Update the insertion qc to CRITICAL
        hist_qc = base.QC(probe_id, one=one, endpoint='insertions')
        hist_qc.update_extended_qc({'tracing_exists': False})
        hist_qc.update('CRITICAL', namespace='tracing')

        # Here need to change the track qc to critical and also extended qc to zero
    else:
        brain_locations, insertion_histology = get_brain_regions(picks)
        # 1) update the alyx models, first put the picked points in the insertion json
        one.alyx.json_field_update(endpoint='insertions', uuid=probe_id, field_name='json',
                                   data={'xyz_picks': np.int32(picks * 1e6).tolist()})

        # Update the insertion qc to register tracing exits
        hist_qc = base.QC(probe_id, one=one, endpoint='insertions')
        hist_qc.update_extended_qc({'tracing_exists': True})
        # 2) patch or create the trajectory coming from histology track
        tdict = create_trajectory_dict(probe_id, insertion_histology, provenance='Histology track')

    hist_traj = one.alyx.rest('trajectories', 'list',
                              probe_insertion=probe_id,
                              provenance='Histology track')
    # if the trajectory exists, remove it, this will cascade delete existing channel locations
    if len(hist_traj):
        if overwrite:
            one.alyx.rest('trajectories', 'delete', id=hist_traj[0]['id'])
        else:
            raise FileExistsError('The session already exists, however overwrite is set to False.'
                                  'If you want to overwrite, set overwrite=True.')
    hist_traj = one.alyx.rest('trajectories', 'create', data=tdict)

    if brain_locations is None:
        return brain_locations, None
    # 3) create channel locations
    if channels:
        channel_dict = create_channel_dict(hist_traj, brain_locations)
        one.alyx.rest('channels', 'create', data=channel_dict)

    return brain_locations, insertion_histology


def register_aligned_track(probe_id, xyz_channels, chn_coords=None, one=None, overwrite=False,
                           channels=True):
    """
    Register ephys aligned trajectory and channel locations to Alyx
    Here we update Alyx models on the database in 2 steps
    1) The trajectory computed from the final electrode channel locations
    2) Channel locations are set to the trajectory
    """
    assert one
    if not np.any(chn_coords):
        chn_coords = SITES_COORDINATES

    insertion = atlas.Insertion.from_track(xyz_channels, brain_atlas)
    tdict = create_trajectory_dict(probe_id, insertion, provenance='Ephys aligned histology track')

    hist_traj = one.alyx.rest('trajectories', 'list',
                              probe_insertion=probe_id,
                              provenance='Ephys aligned histology track')
    # if the trajectory exists, remove it, this will cascade delete existing channel locations
    if len(hist_traj):
        if overwrite:
            one.alyx.rest('trajectories', 'delete', id=hist_traj[0]['id'])
        else:
            raise FileExistsError('The session already exists, however overwrite is set to False.'
                                  'If you want to overwrite, set overwrite=True.')
    hist_traj = one.alyx.rest('trajectories', 'create', data=tdict)

    if channels:
        brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
        brain_regions['xyz'] = xyz_channels
        brain_regions['lateral'] = chn_coords[:, 0]
        brain_regions['axial'] = chn_coords[:, 1]
        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        channel_dict = create_channel_dict(hist_traj, brain_regions)
        one.alyx.rest('channels', 'create', data=channel_dict)


def create_trajectory_dict(probe_id, insertion, provenance):
    """
    Create trajectory dictionary in form to upload to alyx
    :param probe id: unique id of probe insertion
    :type probe_id: string (hexadecimal UUID)
    :param insertion: Insertion object describing entry and tip of trajectory
    :type insertion: object atlas.Insertion
    :param provenance: 'Histology track' or 'Ephys aligned histology track'
    :type provenance: string
    :return tdict:
    :type tdict: dict
    """
    tdict = {'probe_insertion': probe_id,
             'x': insertion.x * 1e6,
             'y': insertion.y * 1e6,
             'z': insertion.z * 1e6,
             'phi': insertion.phi,
             'theta': insertion.theta,
             'depth': insertion.depth * 1e6,
             'roll': insertion.beta,
             'provenance': provenance,
             'coordinate_system': 'IBL-Allen',
             }

    return tdict


def create_channel_dict(traj, brain_locations):
    """
    Create channel dictionary in form to upload to alyx
    :param traj: alyx trajectory object to attach channel information to
    :type traj: dict
    :param brain_locations: information about location of electrode channels in brain atlas
    :type insertion: Bunch
    :return tdict:
    :type tdict: list of dict
    """
    channel_dict = []
    for i in np.arange(brain_locations.id.size):
        channel_dict.append({
            'x': brain_locations.xyz[i, 0] * 1e6,
            'y': brain_locations.xyz[i, 1] * 1e6,
            'z': brain_locations.xyz[i, 2] * 1e6,
            'axial': brain_locations.axial[i],
            'lateral': brain_locations.lateral[i],
            'brain_region': int(brain_locations.id[i]),
            'trajectory_estimate': traj['id']
        })

    return channel_dict


def _parse_filename(track_file):
    tmp = track_file.name.split('_')
    inumber = [i for i, s in enumerate(tmp) if s.isdigit and len(s) == 3][-1]
    search_filter = {'date': tmp[0], 'experiment_number': int(tmp[inumber]),
                     'name': '_'.join(tmp[inumber + 1:- 1]),
                     'subject': '_'.join(tmp[1:inumber])}
    return search_filter


def register_track_files(path_tracks, one=None, overwrite=False):
    """
    :param path_tracks: path to directory containing tracks; also works with a single file name
    :param one:
    :return:
    """
    glob_pattern = "*_probe*_pts*.csv"

    path_tracks = Path(path_tracks)

    if not path_tracks.is_dir():
        track_files = [path_tracks]
    else:
        track_files = list(path_tracks.rglob(glob_pattern))
        track_files.sort()

    assert path_tracks.exists()
    assert one

    ntracks = len(track_files)
    for ind, track_file in enumerate(track_files):
        # Nomenclature expected:
        # '{yyyy-mm-dd}}_{nickname}_{session_number}_{probe_label}_pts.csv'
        # beware: there may be underscores in the subject nickname

        search_filter = _parse_filename(track_file)
        probe = one.alyx.rest('insertions', 'list', **search_filter)
        if len(probe) == 0:
            eid = one.search(subject=search_filter['subject'], date_range=search_filter['date'],
                             number=search_filter['experiment_number'])
            if len(eid) == 0:
                raise Exception(f"No session found {track_file.name}")
            insertion = {'session': eid[0],
                         'name': search_filter['name']}
            probe = one.alyx.rest('insertions', 'create', data=insertion)
        elif len(probe) == 1:
            probe = probe[0]
        else:
            raise ValueError("Multiple probes found.")
        probe_id = probe['id']
        try:
            xyz_picks = load_track_csv(track_file)
            register_track(probe_id, xyz_picks, one=one, overwrite=overwrite)
        except Exception as e:
            _logger.error(str(track_file))
            raise e
        _logger.info(f"{ind + 1}/{ntracks}, {str(track_file)}")


def detect_missing_histology_tracks(path_tracks=None, one=None, subject=None):
    """
    Compares the number of probe insertions to the number of registered histology tracks to see if
    there is a discrepancy so that missing tracks can be properly logged in the database
    :param path_tracks: path to track files to be registered
    :param subject: subject nickname for which to detect missing tracks
    """

    if path_tracks:
        glob_pattern = "*_probe*_pts*.csv"

        path_tracks = Path(path_tracks)

        if not path_tracks.is_dir():
            track_files = [path_tracks]
        else:
            track_files = list(path_tracks.rglob(glob_pattern))
            track_files.sort()

        subjects = []
        for track_file in track_files:
            search_filter = _parse_filename(track_file)
            subjects.append(search_filter['subject'])

        unique_subjects = np.unique(subjects)
    elif not path_tracks and subject:
        unique_subjects = [subject]
    else:
        _logger.warning('Must specifiy either path_tracks or subject argument')
        return

    for subj in unique_subjects:
        insertions = one.alyx.rest('insertions', 'list', subject=subj)
        trajectories = one.alyx.rest('trajectories', 'list', subject=subj,
                                     provenance='Histology track')
        if len(insertions) != len(trajectories):
            ins_sess = np.array([ins['session'] + ins['name'] for ins in insertions])
            traj_sess = np.array([traj['session']['id'] + traj['probe_name']
                                  for traj in trajectories])
            miss_idx = np.where(np.isin(ins_sess, traj_sess, invert=True))[0]

            for idx in miss_idx:

                info = one.path_from_eid(ins_sess[idx][:36]).parts
                print(ins_sess[idx][:36])
                msg = f"Histology tracing missing for {info[-3]}, {info[-2]}, {info[-1]}," \
                      f" {ins_sess[idx][36:]}.\nEnter [y]es to register an empty track for " \
                      f"this insertion \nEnter [n]o, if tracing for this probe insertion will be "\
                      f"conducted at a later date \n>"
                resp = input(msg)
                resp = resp.lower()
                if resp == 'y' or resp == 'yes':
                    _logger.info('Histology track for this probe insertion registered as empty')
                    probe_id = insertions[idx]['id']
                    print(insertions[idx]['session'])
                    print(probe_id)
                    register_track(probe_id, one=one)
                else:
                    _logger.info('Histology track for this probe insertion will not be registered')
                    continue


def coverage(trajs, ba=None):
    """
    Computes a coverage volume from
    :param trajs: dictionary of trajectories from Alyx rest endpoint (one.alyx.rest...)
    :param ba: ibllib.atlas.BrainAtlas instance
    :return: 3D np.array the same size as the volume provided in the brain atlas
    """
    # in um. Coverage = 1 below the first value, 0 after the second, cosine taper in between
    DIST_FCN = [100, 150]
    ACTIVE_LENGTH_UM = 3.5 * 1e3
    MAX_DIST_UM = DIST_FCN[1]  # max distance around the probe to be searched for
    if ba is None:
        ba = atlas.AllenAtlas()

    def crawl_up_from_tip(ins, d):
        return (ins.entry - ins.tip) * (d[:, np.newaxis] /
                                        np.linalg.norm(ins.entry - ins.tip)) + ins.tip

    full_coverage = np.zeros(ba.image.shape, dtype=np.float32).flatten()

    for p in np.arange(len(trajs)):
        if p % 20 == 0:
            print(p / len(trajs))
        traj = trajs[p]

        ins = atlas.Insertion.from_dict(traj)
        # those are the top and bottom coordinates of the active part of the shank extended
        # to maxdist
        d = (np.array([ACTIVE_LENGTH_UM + MAX_DIST_UM * np.sqrt(2),
                       -MAX_DIST_UM * np.sqrt(2)]) + TIP_SIZE_UM)
        top_bottom = crawl_up_from_tip(ins, d / 1e6)
        # this is the axis that has the biggest deviation. Almost always z
        axis = np.argmax(np.abs(np.diff(top_bottom, axis=0)))
        if axis != 2:
            _logger.warning(f"This works only for 45 degree or vertical tracks so far, skipping"
                            f" {ins}")
            continue
        # sample the active track path along this axis
        tbi = ba.bc.xyz2i(top_bottom)
        nz = tbi[1, axis] - tbi[0, axis] + 1
        ishank = np.round(np.array(
            [np.linspace(tbi[0, i], tbi[1, i], nz) for i in np.arange(3)]).T).astype(np.int32)

        # creates a flattened "column" of candidate volume indices around the track
        # around each sample get an horizontal square slice of nx *2 +1 and ny *2 +1 samples
        # flatten the corresponding xyz indices and  compute the min distance to the track only
        # for those
        nx = int(np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[0]) * np.sqrt(2) / 2)) * 2 + 1
        ny = int(np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[1]) * np.sqrt(2) / 2)) * 2 + 1
        ixyz = np.stack([v.flatten() for v in np.meshgrid(
            np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), np.arange(nz))]).T
        ixyz[:, 0] = ishank[ixyz[:, 2], 0] + ixyz[:, 0]
        ixyz[:, 1] = ishank[ixyz[:, 2], 1] + ixyz[:, 1]
        ixyz[:, 2] = ishank[ixyz[:, 2], 2]
        # if any, remove indices that lie outside of the volume bounds
        iok = np.logical_and(0 <= ixyz[:, 0], ixyz[:, 0] < ba.bc.nx)
        iok &= np.logical_and(0 <= ixyz[:, 1], ixyz[:, 1] < ba.bc.ny)
        iok &= np.logical_and(0 <= ixyz[:, 2], ixyz[:, 2] < ba.bc.nz)
        ixyz = ixyz[iok, :]
        # get the minimum distance to the trajectory, to which is applied the cosine taper
        xyz = np.c_[ba.bc.xscale[ixyz[:, 0]], ba.bc.yscale[ixyz[:, 1]], ba.bc.zscale[ixyz[:, 2]]]
        sites_bounds = crawl_up_from_tip(
            ins, (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6)
        mdist = ins.trajectory.mindist(xyz, bounds=sites_bounds)
        coverage = 1 - fcn_cosine(np.array(DIST_FCN) / 1e6)(mdist)
        # remap to the coverage volume
        full_coverage[ba._lookup_inds(ixyz)] += coverage

    full_coverage = full_coverage.reshape(ba.image.shape)
    full_coverage[ba.label == 0] = np.nan
    return full_coverage

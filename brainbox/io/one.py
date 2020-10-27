from pathlib import Path
import logging
import numpy as np

import alf.io
from ibllib.io import spikeglx
from ibllib.atlas import regions_from_allen_csv
from ibllib.io.extractors.training_wheel import extract_wheel_moves, extract_first_movement_times
from oneibl.one import ONE

from brainbox.core import Bunch

logger = logging.getLogger('ibllib')


def load_lfp(eid, one=None, dataset_types=None):
    """
    From an eid, hits the Alyx database and downloads the standard set of datasets
    needed for LFP
    :param eid:
    :param dataset_types: additional dataset types to add to the list
    :return: spikeglx.Reader
    """
    if dataset_types is None:
        dataset_types = []
    dtypes = dataset_types + ['ephysData.raw.lf', 'ephysData.raw.meta', 'ephysData.raw.ch']
    one.load(eid, dataset_types=dtypes, download_only=True)
    session_path = one.path_from_eid(eid)

    efiles = [ef for ef in spikeglx.glob_ephys_files(session_path, bin_exists=False)
              if ef.get('lf', None)]
    return [spikeglx.Reader(ef['lf']) for ef in efiles]


def load_channel_locations(eid, one=None, probe=None, aligned=False):
    """
    From an eid, get brain locations from Alyx database
    analysis.
    :param eid: session eid or dictionary returned by one.alyx.rest('sessions', 'read', id=eid)
    :param dataset_types: additional spikes/clusters objects to add to the standard list
    :return: channels
    """
    if isinstance(eid, dict):
        ses = eid
        eid = ses['url'][-36:]

    one = one or ONE()

    # When a specific probe has been requested
    if isinstance(probe, str):
        insertions = one.alyx.rest('insertions', 'read', session=eid, name=probe)
        labels = [probe]
        tracing = insertions['json']['extended_qc']['tracing_exists'] or False
        resolved = insertions['json']['extended_qc']['alignment_resolved'] or False
        counts = insertions['json']['extended_qc']['alignment_count'] or 0
        probe_id = insertions['id']
    # No specific probe specified, load any that is available
    # Need to catch for the case where we have two of the same probe insertions
    else:
        insertions = one.alyx.rest('insertions', 'list', session=eid)
        labels = [ins['name'] for ins in insertions]
        tracing = [ins['json']['extended_qc']['tracing_exists'] or False for ins in insertions]
        resolved = [ins['json']['extended_qc']['alignment_resolved'] or False
                    for ins in insertions]
        counts = [ins['json']['extended_qc']['alignment_count'] or 0 for ins in insertions]
        probe_id = [ins['id'] for ins in insertions]

    channels = Bunch({})
    r = regions_from_allen_csv()
    for label, trace, resol, count, id in zip(labels, tracing, resolved, counts, probe_id):
        if trace:
            if resol:
                logger.info(f'Channel locations for {label} have been resolved. '
                            f'Channel and cluster locations obtained from ephys aligned histology '
                            f'track.')
                # download the data
                chans = one.load_object(eid, 'channels', collection=f'alf/{label}')
                channels[label] = Bunch({
                    'atlas_id': chans['brainLocationIds_ccf_2017'],
                    'acronym':  r.get(chans['brainLocationIds_ccf_2017'])['acronym'],
                    'x':  chans['mlapdv'][:, 0] / 1e6,
                    'y':  chans['mlapdv'][:, 1] / 1e6,
                    'z':  chans['mlapdv'][:, 2] / 1e6,
                    'axial_um': chans['localCoordinates'][:, 1],
                    'lateral_um': chans['localCoordinates'][:, 0]
                })
            elif count > 0 and aligned:
                logger.info(f'Channel locations for {label} have not been '
                            f'resolved. However, alignment flag set to True so channel and cluster'
                            f' locations will be obtained from latest available ephys aligned '
                            f'histology track.')
                # get the latest user aligned channels
                traj_id = one.alyx.rest('trajectories', 'list', session=eid, probe_name=label,
                                        provenance='Ephys aligned histology track')[0]['id']
                chans = one.alyx.rest('channels', 'list', trajectory_estimate=traj_id)

                channels[label] = Bunch({
                    'atlas_id': np.array([ch['brain_region']['id'] for ch in chans]),
                    'x': np.array([ch['x'] for ch in chans]) / 1e6,
                    'y': np.array([ch['y'] for ch in chans]) / 1e6,
                    'z': np.array([ch['z'] for ch in chans]) / 1e6,
                    'axial_um': np.array([ch['axial'] for ch in chans]),
                    'lateral_um': np.array([ch['lateral'] for ch in chans])
                })
                channels[label]['acronym'] = r.get(channels[label]['atlas_id'])['acronym']
            else:
                logger.info(f'Channel locations for {label} have not been resolved. '
                            f'Channel and cluster locations obtained from histology track.')
                # get the channels from histology tracing
                traj_id = one.alyx.rest('trajectories', 'list', session=eid, probe_name=label,
                                        provenance = 'Histology track')[0]['id']
                chans = one.alyx.rest('channels', 'list', trajectory_estimate=traj_id)

                channels[label] = Bunch({
                    'atlas_id': np.array([ch['brain_region'] for ch in chans]),
                    'x': np.array([ch['x'] for ch in chans]) / 1e6,
                    'y': np.array([ch['y'] for ch in chans]) / 1e6,
                    'z': np.array([ch['z'] for ch in chans]) / 1e6,
                    'axial_um': np.array([ch['axial'] for ch in chans]),
                    'lateral_um': np.array([ch['lateral'] for ch in chans])
                })
                channels[label]['acronym'] = r.get(channels[label]['atlas_id'])['acronym']
        else:
            logger.warning(f'Histology tracing for this probe does not exist. Channels for {label}'
                           f'will return an empty dict')

    return channels


def load_ephys_session(eid, one=None, dataset_types=None):
    """
    From an eid, hits the Alyx database and downloads a standard default set of dataset types
    From a local session Path (pathlib.Path), loads a standard default set of dataset types
     to perform analysis:
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'spikes.clusters',
        'spikes.times',
        'probes.description'
    :param eid: experiment UUID or pathlib.Path of the local session
    :param one: one instance
    :param dataset_types: additional spikes/clusters objects to add to the standard default list
    :return: spikes, clusters, trials (dict of bunch, 1 bunch per probe)
    """
    assert one
    spikes, clusters = load_spike_sorting(eid, one=one, dataset_types=dataset_types)
    if isinstance(eid, Path):
        trials = alf.io.load_object(eid.joinpath('alf'), object='trials')
    else:
        trials = one.load_object(eid, obj='trials')

    return spikes, clusters, trials


def load_spike_sorting(eid, one=None, probe=None, dataset_types=None):
    """
    From an eid, hits the Alyx database and downloads a standard default set of dataset types
    From a local session Path (pathlib.Path), loads a standard default set of dataset types
     to perform analysis:
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'spikes.clusters',
        'spikes.times',
        'probes.description'
    :param eid: experiment UUID or pathlib.Path of the local session
    :param one:
    :param dataset_types: additional spikes/clusters objects to add to the standard default list
    :return: spikes, clusters (dict of bunch, 1 bunch per probe)
    """
    if isinstance(eid, Path):
        return _load_spike_sorting_local(eid)
    one = one or ONE()

    # This is a first draft, no safeguard, no error handling and a draft dataset list.
    session_path = one.path_from_eid(eid)

    # here should try loadlocal before hitting database
    if not session_path:
        print("no session path")
        return (None, None), 'no session path'

    dtypes_default = [
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'spikes.clusters',
        'spikes.times',
        'probes.description'
    ]
    if dataset_types is None:
        dtypes = dtypes_default
    else:
        #  Append extra optional DS
        dtypes = list(set(dataset_types + dtypes_default))

    one.load(eid, dataset_types=dtypes, download_only=True)
    return _load_spike_sorting_local(session_path)


def _load_spike_sorting_local(session_path):
    # gets clusters and spikes from a local session folder
    try:
        probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')
    except FileNotFoundError:
        print("no probes")
        return (None, None), 'no probes'
    spikes = Bunch({})
    clusters = Bunch({})
    for i, _ in enumerate(probes['description']):
        probe_path = session_path.joinpath('alf', probes['description'][i]['label'])
        try:
            cluster = alf.io.load_object(probe_path, object='clusters')
            spike = alf.io.load_object(probe_path, object='spikes')
        except FileNotFoundError:
            print("one probe missing")
            return (None, None), "one probe missing"
        label = probes['description'][i]['label']
        clusters[label] = cluster
        spikes[label] = spike

    return spikes, clusters


def merge_clusters_channels(dic_clus, channels, keys_to_add_extra=None):
    '''
    Takes (default and any extra) values in given keys from channels and assign them to clusters.
    If channels does not contain any data, the new keys are added to clusters but left empty.
    :param dic_clus: dict of bunch, 1 bunch per probe, containing cluster information
    :param channels: dict of bunch, 1 bunch per probe, containing channels information
    :param keys_to_add_extra: Any extra keys contained in channels (will be added to default
    ['acronym', 'atlas_id'])
    :return: clusters (dict of bunch, 1 bunch per probe), with new keys values.
    '''
    probe_labels = list(channels.keys())  # Convert dict_keys into list
    keys_to_add_default = ['acronym', 'atlas_id']

    if keys_to_add_extra is None:
        keys_to_add = keys_to_add_default
    else:
        #  Append extra optional keys
        keys_to_add = list(set(keys_to_add_extra + keys_to_add_default))

    for i_p in range(0, len(probe_labels)):
        clu_ch = dic_clus[probe_labels[i_p]]['channels']

        for i_k in range(0, len(keys_to_add)):
            key = keys_to_add[i_k]
            assert key in channels[probe_labels[i_p]].keys()  # Check key is in channels
            ch_key = channels[probe_labels[i_p]][key]

            if max(clu_ch) < len(ch_key):  # Check length as will use clu_ch as index
                dic_clus[probe_labels[i_p]][key] = ch_key[clu_ch]
            else:
                print(f'Channels in probe {probe_labels[i_p]} does not have'
                      f' the right element number compared to cluster.'
                      f' Data in new cluster key {key} is thus returned empty.')
                dic_clus[probe_labels[i_p]][key] = []

    return dic_clus


def load_spike_sorting_with_channel(eid, one=None, dataset_types=None, aligned=False):
    '''
    For a given eid, get spikes, clusters and channels information, and merges clusters
    and channels information before returning all three variables.
    :param eid:
    :param one:
    :return: spikes, clusters, channels (dict of bunch, 1 bunch per probe)
    '''
    # --- Get spikes and clusters data
    dic_spk_bunch, dic_clus = load_spike_sorting(eid, one=one, dataset_types=dataset_types)

    # -- Get brain regions and assign to clusters
    channels = load_channel_locations(eid, one=one, aligned=aligned)
    dic_clus = merge_clusters_channels(dic_clus, channels, keys_to_add_extra=None)

    return dic_spk_bunch, dic_clus, channels


def load_wheel_reaction_times(eid, one=None):
    """
    Return the calculated reaction times for session.  Reaction times are defined as the time
    between the go cue (onset tone) and the onset of the first substantial wheel movement.   A
    movement is considered sufficiently large if its peak amplitude is at least 1/3rd of the
    distance to threshold (~0.1 radians).

    Negative times mean the onset of the movement occurred before the go cue.  Nans may occur if
    there was no detected movement withing the period, or when the goCue_times or feedback_times
    are nan.

    Parameters
    ----------
    eid : str
        Session UUID
    one : oneibl.ONE
        An instance of ONE for loading data.  If None a new one is instantiated using the defaults.

    Returns
    ----------
    array-like
        reaction times
    """
    if one is None:
        one = ONE()

    trials = one.load_object(eid, 'trials')
    # If already extracted, load and return
    if trials and 'firstMovement_times' in trials:
        return trials['firstMovement_times'] - trials['goCue_times']
    # Otherwise load wheelMoves object and calculate
    moves = one.load_object(eid, 'wheelMoves')
    # Re-extract wheel moves if necessary
    if not moves or 'peakAmplitude' not in moves:
        wheel = one.load_object(eid, 'wheel')
        moves = extract_wheel_moves(wheel['timestamps'], wheel['position'])
    assert trials and moves, 'unable to load trials and wheelMoves data'
    firstMove_times, is_final_movement, ids = extract_first_movement_times(moves, trials)
    return firstMove_times - trials['goCue_times']

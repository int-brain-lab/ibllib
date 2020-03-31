import numpy as np

import alf.io
from ibllib.io import spikeglx
from oneibl.one import ONE

from brainbox.core import Bunch


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


def load_channel_locations(eid, one=None, probe=None):
    """
    From an eid, get brain locations from Alyx database
    analysis.
    :param eid: session eid or dictionary returned by one.alyx.rest('sessions', 'read', id=eid)
    :param dataset_types: additional spikes/clusters objects to add to the standard list
    :return:
    """
    if isinstance(eid, dict):
        ses = eid
    else:
        # need to query alyx. Make sure we have a one client before we hit the endpoint
        if not one:
            one = ONE()
        ses = one.alyx.rest('sessions', 'read', id=eid)
    if isinstance(probe, str):
        probe = [probe]
    labels = probe if probe else [pi['name'] for pi in ses['probe_insertion']]
    channels = Bunch({})
    for label in labels:
        i = [i for i, pi in enumerate(ses['probe_insertion']) if pi['name'] == label]
        if len(i) == 0:
            continue
        trajs = ses['probe_insertion'][i[0]]['trajectory_estimate']
        if not trajs:
            continue
        # the trajectories are ordered within the serializer: histology processed, histology,
        # micro manipulator, plannes so the first is always the desired one
        traj = trajs[0]
        channels[label] = Bunch({
            'atlas_id': np.array([ch['brain_region']['id'] for ch in traj['channels']]),
            'acronym': np.array([ch['brain_region']['acronym'] for ch in traj['channels']]),
            'x': np.array([ch['x'] for ch in traj['channels']]) / 1e6,
            'y': np.array([ch['y'] for ch in traj['channels']]) / 1e6,
            'z': np.array([ch['z'] for ch in traj['channels']]) / 1e6,
            'axial_um': np.array([ch['axial'] for ch in traj['channels']]),
            'lateral_um': np.array([ch['lateral'] for ch in traj['channels']])
        })
    return channels


def load_ephys_session(eid, one=None, dataset_types=None):
    spikes, clusters = load_spike_sorting(eid, one=None, dataset_types=None)
    trials = one.load_object(eid, obj='trials')

    return spikes, clusters, trials


def load_spike_sorting(eid, one=None, dataset_types=None):
    """
    From an eid, hits the Alyx database and downloads a standard set of dataset types to perform
    analysis.
    :param eid:
    :param dataset_types: additional spikes/clusters objects to add to the standard list
    :return:
    """
    if not one:
        one = ONE()
    # This is a first draft, no safeguard, no error handling and a draft dataset list.
    session_path = one.path_from_eid(eid)
    if not session_path:
        print("no session path")
        return (None, None), 'no session path'

    dtypes = [
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'spikes.clusters',
        'spikes.times',
        'probes.description',
    ]
    if dataset_types:
        dtypes = list(set(dataset_types + dtypes))

    _ = one.load(eid, dataset_types=dtypes, download_only=True)
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
        except FileNotFoundError:
            print("one probe missing")
            return (None, None), "one probe missing"
        spike = alf.io.load_object(probe_path, object='spikes')
        label = probes['description'][i]['label']
        clusters[label] = cluster
        spikes[label] = spike

    return spikes, clusters

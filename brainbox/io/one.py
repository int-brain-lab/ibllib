"""Functions for loading IBL ephys and trial data using the Open Neurophysiology Environment."""
import logging
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from one.api import ONE

from iblutil.util import Bunch

from ibllib.io import spikeglx
from ibllib.io.extractors.training_wheel import extract_wheel_moves, extract_first_movement_times
from ibllib.ephys.neuropixel import SITES_COORDINATES, TIP_SIZE_UM, trace_header
from ibllib.atlas import atlas
from ibllib.atlas import AllenAtlas
from ibllib.pipes import histology
from ibllib.pipes.ephys_alignment import EphysAlignment

from brainbox.core import TimeSeries
from brainbox.processing import sync

_logger = logging.getLogger('ibllib')


SPIKES_ATTRIBUTES = ['clusters', 'times']
CLUSTERS_ATTRIBUTES = ['channels', 'depths', 'metrics']


def load_lfp(eid, one=None, dataset_types=None, **kwargs):
    """
    TODO Verify works
    From an eid, hits the Alyx database and downloads the standard set of datasets
    needed for LFP
    :param eid:
    :param dataset_types: additional dataset types to add to the list
    :param open: if True, spikeglx readers are opened
    :return: spikeglx.Reader
    """
    if dataset_types is None:
        dataset_types = []
    dtypes = dataset_types + ['*ephysData.raw.lf*', '*ephysData.raw.meta*', '*ephysData.raw.ch*']
    [one.load_dataset(eid, dset, download_only=True) for dset in dtypes]
    session_path = one.eid2path(eid)

    efiles = [ef for ef in spikeglx.glob_ephys_files(session_path, bin_exists=False)
              if ef.get('lf', None)]
    return [spikeglx.Reader(ef['lf'], **kwargs) for ef in efiles]


def _collection_filter_from_args(probe, spike_sorter=None):
    collection = f'alf/{probe}/{spike_sorter}'
    collection = collection.replace('None', '*')
    collection = collection.replace('/*', '*')
    collection = collection[:-1] if collection.endswith('/') else collection
    return collection


def _get_spike_sorting_collection(collections, pname):
    """
    Filters a list or array of collections to get the relevant spike sorting dataset
    if there is a pykilosort, load it
    """
    #
    collection = next(filter(lambda c: c == f'alf/{pname}/pykilosort', collections), None)
    # otherwise, prefers the shortest
    collection = collection or next(iter(sorted(filter(lambda c: f'alf/{pname}' in c, collections), key=len)), None)
    _logger.debug(f"selecting: {collection} to load amongst candidates: {collections}")
    return collection


def _channels_alyx2bunch(chans):
    channels = Bunch({
        'atlas_id': np.array([ch['brain_region'] for ch in chans]),
        'x': np.array([ch['x'] for ch in chans]) / 1e6,
        'y': np.array([ch['y'] for ch in chans]) / 1e6,
        'z': np.array([ch['z'] for ch in chans]) / 1e6,
        'axial_um': np.array([ch['axial'] for ch in chans]),
        'lateral_um': np.array([ch['lateral'] for ch in chans])
    })
    return channels


def _channels_traj2bunch(xyz_chans, brain_atlas):
    brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_chans))
    channels = {
        'x': xyz_chans[:, 0],
        'y': xyz_chans[:, 1],
        'z': xyz_chans[:, 2],
        'acronym': brain_regions['acronym'],
        'atlas_id': brain_regions['id']
    }

    return channels


def _channels_bunch2alf(channels):
    channels_ = {
        'mlapdv': np.c_[channels['x'], channels['y'], channels['z']] * 1e6,
        'brainLocationIds_ccf_2017': channels['atlas_id'],
        'localCoordinates': np.c_[channels['lateral_um'], channels['axial_um']]}
    return channels_


def _channels_alf2bunch(channels, brain_regions=None):
    # reformat the dictionary according to the standard that comes out of Alyx
    channels_ = {
        'x': channels['mlapdv'][:, 0].astype(np.float64) / 1e6,
        'y': channels['mlapdv'][:, 1].astype(np.float64) / 1e6,
        'z': channels['mlapdv'][:, 2].astype(np.float64) / 1e6,
        'acronym': None,
        'atlas_id': channels['brainLocationIds_ccf_2017'],
        'axial_um': channels['localCoordinates'][:, 1],
        'lateral_um': channels['localCoordinates'][:, 0],
    }
    if brain_regions:
        channels_['acronym'] = brain_regions.get(channels_['atlas_id'])['acronym']
    return channels_


def _load_spike_sorting(eid, one=None, collection=None, revision=None, return_channels=True, dataset_types=None,
                        brain_regions=None):
    """
    Generic function to load spike sorting according data using ONE.

    Will try to load one spike sorting for any probe present for the eid matching the collection
    For each probe it will load a spike sorting:
        - if there is one version: loads this one
        - if there are several versions: loads pykilosort, if not found the shortest collection (alf/probeXX)

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : one.api.OneAlyx
        An instance of ONE (may be in 'local' mode)
    collection : str
        collection filter word - accepts wildcards - can be a combination of spike sorter and
        probe.  See `ALF documentation`_ for details.
    revision : str
        A particular revision return (defaults to latest revision).  See `ALF documentation`_ for
        details.
    return_channels : bool
        Defaults to False otherwise loads channels from disk (takes longer)

    .. _ALF documentation: https://one.internationalbrainlab.org/alf_intro.html#optional-components

    Returns
    -------
    spikes : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains bunch(es) of spike data for the provided
        session and spike sorter, with keys ('clusters', 'times')
    clusters : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains bunch(es) of cluster data, with keys
        ('channels', 'depths', 'metrics')
    channels : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains channel locations with keys ('acronym',
        'atlas_id', 'x', 'y', 'z').  Only returned when return_channels is True.  Atlas IDs
        non-lateralized.
    """
    one = one or ONE()
    # enumerate probes and load according to the name
    collections = one.list_collections(eid, filename='spikes*', collection=collection, revision=revision)
    if len(collections) == 0:
        _logger.warning(f"eid {eid}: no collection found with collection filter: {collection}, revision: {revision}")
    pnames = list(set([c.split('/')[1] for c in collections]))
    spikes, clusters, channels = ({} for _ in range(3))

    spike_attributes, cluster_attributes = _get_attributes(dataset_types)

    for pname in pnames:
        probe_collection = _get_spike_sorting_collection(collections, pname)
        spikes[pname] = one.load_object(eid, collection=probe_collection, obj='spikes',
                                        attribute=spike_attributes)
        clusters[pname] = one.load_object(eid, collection=probe_collection, obj='clusters',
                                          attribute=cluster_attributes)
    if return_channels:
        channels = _load_channels_locations_from_disk(
            eid, collection=collection, one=one, revision=revision, brain_regions=brain_regions)
        return spikes, clusters, channels
    else:
        return spikes, clusters


def _get_attributes(dataset_types):
    if dataset_types is None:
        return SPIKES_ATTRIBUTES, CLUSTERS_ATTRIBUTES
    else:
        spike_attributes = [sp.split('.')[1] for sp in dataset_types if 'spikes.' in sp]
        cluster_attributes = [cl.split('.')[1] for cl in dataset_types if 'clusters.' in cl]
        spike_attributes = list(set(SPIKES_ATTRIBUTES + spike_attributes))
        cluster_attributes = list(set(CLUSTERS_ATTRIBUTES + cluster_attributes))
        return spike_attributes, cluster_attributes


def _load_channels_locations_from_disk(eid, collection=None, one=None, revision=None, brain_regions=None):
    _logger.debug('loading spike sorting from disk')
    channels = Bunch({})
    collections = one.list_collections(eid, filename='channels*', collection=collection, revision=revision)
    if len(collections) == 0:
        _logger.warning(f"eid {eid}: no collection found with collection filter: {collection}, revision: {revision}")
    probes = list(set([c.split('/')[1] for c in collections]))
    for probe in probes:
        probe_collection = _get_spike_sorting_collection(collections, probe)
        channels[probe] = one.load_object(eid, collection=probe_collection, obj='channels')
        # if the spike sorter has not aligned data, try and get the alignment available
        if 'brainLocationIds_ccf_2017' not in channels[probe].keys():
            aligned_channel_collections = one.list_collections(
                eid, filename='channels.brainLocationIds_ccf_2017*', collection=f'alf/{probe}', revision=revision)
            if len(aligned_channel_collections) == 0:
                _logger.warning(f"no resolved alignment dataset found for {eid}/{probe}")
                continue
            _logger.debug(f"looking for a resolved alignment dataset in {aligned_channel_collections}")
            ac_collection = _get_spike_sorting_collection(aligned_channel_collections, probe)
            channels_aligned = one.load_object(eid, 'channels', collection=ac_collection)
            channels[probe] = channel_locations_interpolation(channels_aligned, channels[probe])
            # only have to reformat channels if we were able to load coordinates from disk
        channels[probe] = _channels_alf2bunch(channels[probe], brain_regions=brain_regions)
    return channels


def channel_locations_interpolation(channels_aligned, channels=None, brain_regions=None):
    """
    oftentimes the channel map for different spike sorters may be different so interpolate the alignment onto
    if there is no spike sorting in the base folder, the alignment doesn't have the localCoordinates field
    so we reconstruct from the Neuropixel map. This only happens for early pykilosort sorts
    :param channels_aligned: Bunch or dictionary of aligned channels containing at least keys
     'localCoordinates', 'mlapdv' and 'brainLocationIds_ccf_2017'
     OR
      'x', 'y', 'z', 'acronym', 'axial_um'
      those are the guide for the interpolation
    :param channels: Bunch or dictionary of aligned channels containing at least keys 'localCoordinates'
    :param brain_regions: None (default) or ibllib.atlas.BrainRegions object
     if None will return a dict with keys 'localCoordinates', 'mlapdv', 'brainLocationIds_ccf_2017
     if a brain region object is provided, outputts a dict with keys
      'x', 'y', 'z', 'acronym', 'atlas_id', 'axial_um', 'lateral_um'
    :return: Bunch or dictionary of channels with brain coordinates keys
    """
    NEUROPIXEL_VERSION = 1
    h = trace_header(version=NEUROPIXEL_VERSION)
    if channels is None:
        channels = {'localCoordinates': np.c_[h['x'], h['y']]}
    nch = channels['localCoordinates'].shape[0]
    if set(['x', 'y', 'z']).issubset(set(channels_aligned.keys())):
        channels_aligned = _channels_bunch2alf(channels_aligned)
    if 'localCoordinates' in channels_aligned.keys():
        aligned_depths = channels_aligned['localCoordinates'][:, 1]
    else:  # this is a edge case for a few spike sorting sessions
        assert channels_aligned['mlapdv'].shape[0] == 384
        aligned_depths = h['y']
    depth_aligned, ind_aligned = np.unique(aligned_depths, return_index=True)
    depths, ind, iinv = np.unique(channels['localCoordinates'][:, 1], return_index=True, return_inverse=True)
    channels['mlapdv'] = np.zeros((nch, 3))
    for i in np.arange(3):
        channels['mlapdv'][:, i] = np.interp(
            depths, depth_aligned, channels_aligned['mlapdv'][ind_aligned, i])[iinv]
    # the brain locations have to be interpolated by nearest neighbour
    fcn_interp = interp1d(depth_aligned, channels_aligned['brainLocationIds_ccf_2017'][ind_aligned], kind='nearest')
    channels['brainLocationIds_ccf_2017'] = fcn_interp(depths)[iinv].astype(np.int32)
    if brain_regions is not None:
        return _channels_alf2bunch(channels, brain_regions=brain_regions)
    else:
        return channels


def _load_channel_locations_traj(eid, probe=None, one=None, revision=None, aligned=False,
                                 brain_atlas=None):
    print('from traj')
    channels = Bunch()
    brain_atlas = brain_atlas or AllenAtlas
    # need to find the collection bruh
    insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe)[0]
    collection = _collection_filter_from_args(probe=probe)
    collections = one.list_collections(eid, filename='channels*', collection=collection,
                                       revision=revision)
    probe_collection = _get_spike_sorting_collection(collections, probe)
    chn_coords = one.load_dataset(eid, 'channels.localCoordinates', collection=probe_collection)
    depths = chn_coords[:, 1]

    tracing = insertion.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}). \
        get('tracing_exists', False)
    resolved = insertion.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}). \
        get('alignment_resolved', False)
    counts = insertion.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}). \
        get('alignment_count', 0)

    if tracing:
        xyz = np.array(insertion['json']['xyz_picks']) / 1e6
        if resolved:

            _logger.info(f'Channel locations for {eid}/{probe} have been resolved. '
                         f'Channel and cluster locations obtained from ephys aligned histology '
                         f'track.')

            traj = one.alyx.rest('trajectories', 'list', session=eid, probe=probe,
                                 provenance='Ephys aligned histology track')[0]
            align_key = insertion['json']['extended_qc']['alignment_stored']
            feature = traj['json'][align_key][0]
            track = traj['json'][align_key][1]
            ephysalign = EphysAlignment(xyz, depths, track_prev=track,
                                        feature_prev=feature,
                                        brain_atlas=brain_atlas, speedy=True)
            chans = ephysalign.get_channel_locations(feature, track)
            channels[probe] = _channels_traj2bunch(chans, brain_atlas)

        elif counts > 0 and aligned:
            _logger.info(f'Channel locations for {eid}/{probe} have not been '
                         f'resolved. However, alignment flag set to True so channel and cluster'
                         f' locations will be obtained from latest available ephys aligned '
                         f'histology track.')
            # get the latest user aligned channels
            traj = one.alyx.rest('trajectories', 'list', session=eid, probe=probe,
                                 provenance='Ephys aligned histology track')[0]
            align_key = insertion['json']['extended_qc']['alignment_stored']
            feature = traj['json'][align_key][0]
            track = traj['json'][align_key][1]
            ephysalign = EphysAlignment(xyz, depths, track_prev=track,
                                        feature_prev=feature,
                                        brain_atlas=brain_atlas, speedy=True)
            chans = ephysalign.get_channel_locations(feature, track)

            channels[probe] = _channels_traj2bunch(chans, brain_atlas)

        else:
            _logger.info(f'Channel locations for {eid}/{probe} have not been resolved. '
                         f'Channel and cluster locations obtained from histology track.')
            # get the channels from histology tracing
            xyz = xyz[np.argsort(xyz[:, 2]), :]
            chans = histology.interpolate_along_track(xyz, (depths + TIP_SIZE_UM) / 1e6)

            channels[probe] = _channels_traj2bunch(chans, brain_atlas)

        channels[probe]['axial_um'] = chn_coords[:, 1]
        channels[probe]['lateral_um'] = chn_coords[:, 0]

    else:
        _logger.warning(f'Histology tracing for {probe} does not exist. '
                        f'No channels for {probe}')
        channels = None

    return channels


def load_channel_locations(eid, probe=None, one=None, aligned=False, brain_atlas=None):
    """
    Load the brain locations of each channel for a given session/probe

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    probe : [str, list of str]
        The probe label(s), e.g. 'probe01'
    one : one.api.OneAlyx
        An instance of ONE (shouldn't be in 'local' mode)
    aligned : bool
        Whether to get the latest user aligned channel when not resolved or use histology track
    brain_atlas : ibllib.atlas.BrainAtlas
        Brain atlas object (default: Allen atlas)

    Returns
    -------
    dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains channel locations with keys ('acronym',
        'atlas_id', 'x', 'y', 'z').  Atlas IDs non-lateralized.
    """
    one = one or ONE()
    brain_atlas = brain_atlas or AllenAtlas()
    if isinstance(eid, dict):
        ses = eid
        eid = ses['url'][-36:]
    else:
        eid = one.to_eid(eid)
    collection = _collection_filter_from_args(probe=probe)
    channels = _load_channels_locations_from_disk(eid, one=one, collection=collection,
                                                  brain_regions=brain_atlas.regions)
    incomplete_probes = [k for k in channels if 'x' not in channels[k]]
    for iprobe in incomplete_probes:
        channels_ = _load_channel_locations_traj(eid, probe=iprobe, one=one, aligned=aligned,
                                                 brain_atlas=brain_atlas)
        if channels_ is not None:
            channels[iprobe] = channels_[iprobe]
    return channels


def load_spike_sorting_fast(eid, one=None, probe=None, dataset_types=None, spike_sorter=None, revision=None,
                            brain_regions=None, nested=True, collection=None, return_collection=False):
    """
    From an eid, loads spikes and clusters for all probes
    The following set of dataset types are loaded:
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'spikes.clusters',
        'spikes.times',
        'probes.description'
    :param eid: experiment UUID or pathlib.Path of the local session
    :param one: an instance of OneAlyx
    :param probe: name of probe to load in, if not given all probes for session will be loaded
    :param dataset_types: additional spikes/clusters objects to add to the standard default list
    :param spike_sorter: name of the spike sorting you want to load (None for default)
    :param collection: name of the spike sorting collection to load - exclusive with spike sorter name ex: "alf/probe00"
    :param return_channels: (bool) defaults to False otherwise tries and load channels from disk
    :param brain_regions: ibllib.atlas.regions.BrainRegions object - will label acronyms if provided
    :param nested: if a single probe is required, do not output a dictionary with the probe name as key
    :param return_collection: (False) if True, will return the collection used to load
    :return: spikes, clusters, channels (dict of bunch, 1 bunch per probe)
    """
    if collection is None:
        collection = _collection_filter_from_args(probe, spike_sorter)
    _logger.debug(f"load spike sorting with collection filter {collection}")
    kwargs = dict(eid=eid, one=one, collection=collection, revision=revision, dataset_types=dataset_types,
                  brain_regions=brain_regions)
    spikes, clusters, channels = _load_spike_sorting(**kwargs, return_channels=True)
    clusters = merge_clusters_channels(clusters, channels, keys_to_add_extra=None)
    if nested is False:
        k = list(spikes.keys())[0]
        channels = channels[k]
        clusters = clusters[k]
        spikes = spikes[k]
    if return_collection:
        return spikes, clusters, channels, collection
    else:
        return spikes, clusters, channels


def load_spike_sorting(eid, one=None, probe=None, dataset_types=None, spike_sorter=None, revision=None,
                       brain_regions=None):
    """
    From an eid, loads spikes and clusters for all probes
    The following set of dataset types are loaded:
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'spikes.clusters',
        'spikes.times',
        'probes.description'
    :param eid: experiment UUID or pathlib.Path of the local session
    :param one: an instance of OneAlyx
    :param probe: name of probe to load in, if not given all probes for session will be loaded
    :param dataset_types: additional spikes/clusters objects to add to the standard default list
    :param spike_sorter: name of the spike sorting you want to load (None for default)
    :param return_channels: (bool) defaults to False otherwise tries and load channels from disk
    :param brain_regions: ibllib.atlas.regions.BrainRegions object - will label acronyms if provided
    :return: spikes, clusters (dict of bunch, 1 bunch per probe)
    """
    collection = _collection_filter_from_args(probe, spike_sorter)
    _logger.debug(f"load spike sorting with collection filter {collection}")
    spikes, clusters = _load_spike_sorting(eid=eid, one=one, collection=collection, revision=revision,
                                           return_channels=False, dataset_types=dataset_types,
                                           brain_regions=brain_regions)
    return spikes, clusters


def load_spike_sorting_with_channel(eid, one=None, probe=None, aligned=False, dataset_types=None,
                                    spike_sorter=None, brain_atlas=None):
    """
    For a given eid, get spikes, clusters and channels information, and merges clusters
    and channels information before returning all three variables.

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : one.api.OneAlyx
        An instance of ONE (shouldn't be in 'local' mode)
    probe : [str, list of str]
        The probe label(s), e.g. 'probe01'
    aligned : bool
        Whether to get the latest user aligned channel when not resolved or use histology track
    dataset_types : list of str
        Optional additional spikes/clusters objects to add to the standard default list
    spike_sorter : str
        Name of the spike sorting you want to load (None for default which is pykilosort if it's
        available otherwise the default MATLAB kilosort)
    brain_atlas : ibllib.atlas.BrainAtlas
        Brain atlas object (default: Allen atlas)

    Returns
    -------
    spikes : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains bunch(es) of spike data for the provided
        session and spike sorter, with keys ('clusters', 'times')
    clusters : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains bunch(es) of cluster data, with keys
        ('channels', 'depths', 'metrics')
    channels : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains channel locations with keys ('acronym',
        'atlas_id', 'x', 'y', 'z').  Atlas IDs non-lateralized.
    """
    # --- Get spikes and clusters data
    one = one or ONE()
    brain_atlas = brain_atlas or AllenAtlas()
    spikes, clusters = load_spike_sorting(eid, one=one, probe=probe, dataset_types=dataset_types,
                                          spike_sorter=spike_sorter)
    # -- Get brain regions and assign to clusters
    channels = load_channel_locations(eid, one=one, probe=probe, aligned=aligned,
                                      brain_atlas=brain_atlas)
    clusters = merge_clusters_channels(clusters, channels, keys_to_add_extra=None)
    return spikes, clusters, channels


def load_ephys_session(eid, one=None):
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

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : oneibl.one.OneAlyx, optional
        ONE object to use for loading. Will generate internal one if not used, by default None

    Returns
    -------
    spikes : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains bunch(es) of spike data for the provided
        session and spike sorter, with keys ('clusters', 'times')
    clusters : dict of one.alf.io.AlfBunch
        A dict with probe labels as keys, contains bunch(es) of cluster data, with keys
        ('channels', 'depths', 'metrics')
    trials : one.alf.io.AlfBunch of numpy.ndarray
        The session trials data
    """
    assert one
    spikes, clusters = load_spike_sorting(eid, one=one)
    trials = one.load_object(eid, 'trials')
    return spikes, clusters, trials


def _remove_old_clusters(session_path, probe):
    # gets clusters and spikes from a local session folder
    probe_path = session_path.joinpath('alf', probe)

    # look for clusters.metrics.csv file, if it exists delete as we now have .pqt file instead
    cluster_file = probe_path.joinpath('clusters.metrics.csv')

    if cluster_file.exists():
        os.remove(cluster_file)
        _logger.info('Deleting old clusters.metrics.csv file')


def merge_clusters_channels(dic_clus, channels, keys_to_add_extra=None):
    """
    Takes (default and any extra) values in given keys from channels and assign them to clusters.
    If channels does not contain any data, the new keys are added to clusters but left empty.

    Parameters
    ----------
    dic_clus : dict of one.alf.io.AlfBunch
        1 bunch per probe, containing cluster information
    channels : dict of one.alf.io.AlfBunch
        1 bunch per probe, containing channels bunch with keys ('acronym', 'atlas_id', 'x', 'y', z', 'localCoordinates')
    keys_to_add_extra : list of str
        Any extra keys to load into channels bunches

    Returns
    -------
    dict of one.alf.io.AlfBunch
        clusters (1 bunch per probe) with new keys values.
    """
    probe_labels = list(channels.keys())  # Convert dict_keys into list
    keys_to_add_default = ['acronym', 'atlas_id', 'x', 'y', 'z', 'axial_um', 'lateral_um']

    if keys_to_add_extra is None:
        keys_to_add = keys_to_add_default
    else:
        #  Append extra optional keys
        keys_to_add = list(set(keys_to_add_extra + keys_to_add_default))

    for label in probe_labels:
        clu_ch = dic_clus[label]['channels']
        for key in keys_to_add:
            try:
                assert key in channels[label].keys()  # Check key is in channels
                ch_key = channels[label][key]
                nch_key = len(ch_key) if ch_key is not None else 0
                if max(clu_ch) < nch_key:  # Check length as will use clu_ch as index
                    dic_clus[label][key] = ch_key[clu_ch]
                else:
                    _logger.warning(
                        f'Probe {label}: merging channels and clusters for key "{key}" has {nch_key} on channels'
                        f' but expected {max(clu_ch)}. Data in new cluster key "{key}" is returned empty.')
                    dic_clus[label][key] = []
            except AssertionError:
                _logger.warning(f'Either clusters or channels does not have key {key}, could not merge')
                continue

    return dic_clus


def load_passive_rfmap(eid, one=None):
    """
    For a given eid load in the passive receptive field mapping protocol data

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : oneibl.one.OneAlyx, optional
        An instance of ONE (may be in 'local' - offline - mode)

    Returns
    -------
    one.alf.io.AlfBunch
        Passive receptive field mapping data
    """
    one = one or ONE()

    # Load in the receptive field mapping data
    rf_map = one.load_object(eid, obj='passiveRFM', collection='alf')
    frames = np.fromfile(one.load_dataset(eid, '_iblrig_RFMapStim.raw.bin',
                                          collection='raw_passive_data'), dtype="uint8")
    y_pix, x_pix = 15, 15
    frames = np.transpose(np.reshape(frames, [y_pix, x_pix, -1], order="F"), [2, 1, 0])
    rf_map['frames'] = frames

    return rf_map


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
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : oneibl.one.OneAlyx, optional
        one object to use for loading. Will generate internal one if not used, by default None

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


def load_trials_df(eid, one=None, maxlen=None, t_before=0., t_after=0., ret_wheel=False,
                   ret_abswheel=False, wheel_binsize=0.02, addtl_types=[]):
    """
    Generate a pandas dataframe of per-trial timing information about a given session.
    Each row in the frame will correspond to a single trial, with timing values indicating timing
    session-wide (i.e. time in seconds since session start). Can optionally return a resampled
    wheel velocity trace of either the signed or absolute wheel velocity.

    The resulting dataframe will have a new set of columns, trial_start and trial_end, which define
    via t_before and t_after the span of time assigned to a given trial.
    (useful for bb.modeling.glm)

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : oneibl.one.OneAlyx, optional
        one object to use for loading. Will generate internal one if not used, by default None
    maxlen : float, optional
        Maximum trial length for inclusion in df. Trials where feedback - response is longer
        than this value will not be included in the dataframe, by default None
    t_before : float, optional
        Time before stimulus onset to include for a given trial, as defined by the trial_start
        column of the dataframe. If zero, trial_start will be identical to stimOn, by default 0.
    t_after : float, optional
        Time after feedback to include in the trail, as defined by the trial_end
        column of the dataframe. If zero, trial_end will be identical to feedback, by default 0.
    ret_wheel : bool, optional
        Whether to return the time-resampled wheel velocity trace, by default False
    ret_abswheel : bool, optional
        Whether to return the time-resampled absolute wheel velocity trace, by default False
    wheel_binsize : float, optional
        Time bins to resample wheel velocity to, by default 0.02
    addtl_types : list, optional
        List of additional types from an ONE trials object to include in the dataframe. Must be
        valid keys to the dict produced by one.load_object(eid, 'trials'), by default empty.

    Returns
    -------
    pandas.DataFrame
        Dataframe with trial-wise information. Indices are the actual trial order in the original
        data, preserved even if some trials do not meet the maxlen criterion. As a result will not
        have a monotonic index. Has special columns trial_start and trial_end which define start
        and end times via t_before and t_after
    """
    if not one:
        one = ONE()

    if ret_wheel and ret_abswheel:
        raise ValueError('ret_wheel and ret_abswheel cannot both be true.')

    # Define which datatypes we want to pull out
    trialstypes = ['choice',
                   'probabilityLeft',
                   'feedbackType',
                   'feedback_times',
                   'contrastLeft',
                   'contrastRight',
                   'goCue_times',
                   'stimOn_times']
    trialstypes.extend(addtl_types)

    # A quick function to remap probabilities in those sessions where it was not computed correctly
    def remap_trialp(probs):
        # Block probabilities in trial data aren't accurate and need to be remapped
        validvals = np.array([0.2, 0.5, 0.8])
        diffs = np.abs(np.array([x - validvals for x in probs]))
        maps = diffs.argmin(axis=1)
        return validvals[maps]

    trials = one.load_object(eid, 'trials', collection='alf')
    starttimes = trials.stimOn_times
    endtimes = trials.feedback_times
    tmp = {key: value for key, value in trials.items() if key in trialstypes}

    if maxlen is not None:
        with np.errstate(invalid='ignore'):
            keeptrials = (endtimes - starttimes) <= maxlen
    else:
        keeptrials = range(len(starttimes))
    trialdata = {x: tmp[x][keeptrials] for x in trialstypes}
    trialdata['probabilityLeft'] = remap_trialp(trialdata['probabilityLeft'])
    trialsdf = pd.DataFrame(trialdata)
    if maxlen is not None:
        trialsdf.set_index(np.nonzero(keeptrials)[0], inplace=True)
    trialsdf['trial_start'] = trialsdf['stimOn_times'] - t_before
    trialsdf['trial_end'] = trialsdf['feedback_times'] + t_after
    tdiffs = trialsdf['trial_end'] - np.roll(trialsdf['trial_start'], -1)
    if np.any(tdiffs[:-1] > 0):
        logging.warning(f'{sum(tdiffs[:-1] > 0)} trials overlapping due to t_before and t_after '
                        'values. Try reducing one or both!')
    if not ret_wheel and not ret_abswheel:
        return trialsdf

    wheel = one.load_object(eid, 'wheel', collection='alf')
    whlpos, whlt = wheel.position, wheel.timestamps
    starttimes = trialsdf['trial_start']
    endtimes = trialsdf['trial_end']
    wh_endlast = 0
    trials = []
    for (start, end) in np.vstack((starttimes, endtimes)).T:
        wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
        wh_endind = np.searchsorted(whlt[wh_endlast:], end, side='right') + wh_endlast + 4
        wh_endlast = wh_endind
        tr_whlpos = whlpos[wh_startind - 1:wh_endind + 1]
        tr_whlt = whlt[wh_startind - 1:wh_endind + 1] - start
        tr_whlt[0] = 0.  # Manual previous-value interpolation
        whlseries = TimeSeries(tr_whlt, tr_whlpos, columns=['whlpos'])
        whlsync = sync(wheel_binsize, timeseries=whlseries, interp='previous')
        trialstartind = np.searchsorted(whlsync.times, 0)
        trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
        trpos = whlsync.values[trialstartind:trialendind + trialstartind]
        whlvel = trpos[1:] - trpos[:-1]
        whlvel = np.insert(whlvel, 0, 0)
        if np.abs((trialendind - len(whlvel))) > 0:
            raise IndexError('Mismatch between expected length of wheel data and actual.')
        if ret_wheel:
            trials.append(whlvel)
        elif ret_abswheel:
            trials.append(np.abs(whlvel))
    trialsdf['wheel_velocity'] = trials
    return trialsdf


def load_channels_from_insertion(ins, depths=None, one=None, ba=None):

    PROV_2_VAL = {
        'Resolved': 90,
        'Ephys aligned histology track': 70,
        'Histology track': 50,
        'Micro-manipulator': 30,
        'Planned': 10}

    one = one or ONE()
    ba = ba or atlas.AllenAtlas()
    traj = one.alyx.rest('trajectories', 'list', probe_insertion=ins['id'])
    val = [PROV_2_VAL[tr['provenance']] for tr in traj]
    idx = np.argmax(val)
    traj = traj[idx]
    if depths is None:
        depths = SITES_COORDINATES[:, 1]
    if traj['provenance'] == 'Planned' or traj['provenance'] == 'Micro-manipulator':
        ins = atlas.Insertion.from_dict(traj)
        # Deepest coordinate first
        xyz = np.c_[ins.tip, ins.entry].T
        xyz_channels = histology.interpolate_along_track(xyz, (depths +
                                                               TIP_SIZE_UM) / 1e6)
    else:
        xyz = np.array(ins['json']['xyz_picks']) / 1e6
        if traj['provenance'] == 'Histology track':
            xyz = xyz[np.argsort(xyz[:, 2]), :]
            xyz_channels = histology.interpolate_along_track(xyz, (depths +
                                                                   TIP_SIZE_UM) / 1e6)
        else:
            align_key = ins['json']['extended_qc']['alignment_stored']
            feature = traj['json'][align_key][0]
            track = traj['json'][align_key][1]
            ephysalign = EphysAlignment(xyz, depths, track_prev=track,
                                        feature_prev=feature,
                                        brain_atlas=ba, speedy=True)
            xyz_channels = ephysalign.get_channel_locations(feature, track)
    return xyz_channels

"""Functions for loading IBL ephys and trial data using the Open Neurophysiology Environment."""
from dataclasses import dataclass, field
import gc
import logging
import os
from pathlib import Path


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from one.api import ONE, One
import one.alf.io as alfio
from one.alf.files import get_alf_path
from one.alf.exceptions import ALFObjectNotFound
from one.alf import cache
from neuropixel import TIP_SIZE_UM, trace_header
import spikeglx

from iblutil.util import Bunch
from ibllib.io.extractors.training_wheel import extract_wheel_moves, extract_first_movement_times
from ibllib.atlas import atlas, AllenAtlas, BrainRegions
from ibllib.pipes import histology
from ibllib.pipes.ephys_alignment import EphysAlignment

import brainbox.plot
from brainbox.ephys_plots import plot_brain_regions
from brainbox.metrics.single_units import quick_unit_metrics
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
from brainbox.behavior.dlc import likelihood_threshold, get_pupil_diameter, get_smooth_pupil_diameter

_logger = logging.getLogger('ibllib')


SPIKES_ATTRIBUTES = ['clusters', 'times', 'amps', 'depths']
CLUSTERS_ATTRIBUTES = ['channels', 'depths', 'metrics', 'uuids']


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
        Defaults to False otherwise loads channels from disk

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
    pnames = list(set(c.split('/')[1] for c in collections))
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
                eid, filename='channels.brainLocationIds_ccf_2017*', collection=probe_collection, revision=revision)
            if len(aligned_channel_collections) == 0:
                _logger.debug(f"no resolved alignment dataset found for {eid}/{probe}")
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
    if {'x', 'y', 'z'}.issubset(set(channels_aligned.keys())):
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
                                 brain_atlas=None, return_source=False):
    if not hasattr(one, 'alyx'):
        return {}, None
    _logger.debug(f"trying to load from traj {probe}")
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

            _logger.debug(f'Channel locations for {eid}/{probe} have been resolved. '
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
            source = 'resolved'
        elif counts > 0 and aligned:
            _logger.debug(f'Channel locations for {eid}/{probe} have not been '
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
            source = 'aligned'
        else:
            _logger.debug(f'Channel locations for {eid}/{probe} have not been resolved. '
                          f'Channel and cluster locations obtained from histology track.')
            # get the channels from histology tracing
            xyz = xyz[np.argsort(xyz[:, 2]), :]
            chans = histology.interpolate_along_track(xyz, (depths + TIP_SIZE_UM) / 1e6)
            channels[probe] = _channels_traj2bunch(chans, brain_atlas)
            source = 'traced'
        channels[probe]['axial_um'] = chn_coords[:, 1]
        channels[probe]['lateral_um'] = chn_coords[:, 0]

    else:
        _logger.warning(f'Histology tracing for {probe} does not exist. No channels for {probe}')
        source = ''
        channels = None

    if return_source:
        return channels, source
    else:
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
    optional: string 'resolved', 'aligned', 'traced' or ''
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
        channels_, source = _load_channel_locations_traj(eid, probe=iprobe, one=one, aligned=aligned,
                                                         brain_atlas=brain_atlas, return_source=True)
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
    :param brain_regions: ibllib.atlas.regions.BrainRegions object - will label acronyms if provided
    :param nested: if a single probe is required, do not output a dictionary with the probe name as key
    :param return_collection: (False) if True, will return the collection used to load
    :return: spikes, clusters, channels (dict of bunch, 1 bunch per probe)
    """
    _logger.warning('Deprecation warning: brainbox.io.one.load_spike_sorting_fast will be removed in future versions.'
                    'Use brainbox.io.one.SpikeSortingLoader instead')
    if collection is None:
        collection = _collection_filter_from_args(probe, spike_sorter)
    _logger.debug(f"load spike sorting with collection filter {collection}")
    kwargs = dict(eid=eid, one=one, collection=collection, revision=revision, dataset_types=dataset_types,
                  brain_regions=brain_regions)
    spikes, clusters, channels = _load_spike_sorting(**kwargs, return_channels=True)
    clusters = merge_clusters_channels(clusters, channels, keys_to_add_extra=None)
    if nested is False and len(spikes.keys()) == 1:
        k = list(spikes.keys())[0]
        channels = channels[k]
        clusters = clusters[k]
        spikes = spikes[k]
    if return_collection:
        return spikes, clusters, channels, collection
    else:
        return spikes, clusters, channels


def load_spike_sorting(eid, one=None, probe=None, dataset_types=None, spike_sorter=None, revision=None,
                       brain_regions=None, return_collection=False):
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
    :param brain_regions: ibllib.atlas.regions.BrainRegions object - will label acronyms if provided
    :param return_collection:(bool - False) if True, returns the collection for loading the data
    :return: spikes, clusters (dict of bunch, 1 bunch per probe)
    """
    _logger.warning('Deprecation warning: brainbox.io.one.load_spike_sorting will be removed in future versions.'
                    'Use brainbox.io.one.SpikeSortingLoader instead')
    collection = _collection_filter_from_args(probe, spike_sorter)
    _logger.debug(f"load spike sorting with collection filter {collection}")
    spikes, clusters = _load_spike_sorting(eid=eid, one=one, collection=collection, revision=revision,
                                           return_channels=False, dataset_types=dataset_types,
                                           brain_regions=brain_regions)
    if return_collection:
        return spikes, clusters, collection
    else:
        return spikes, clusters


def load_spike_sorting_with_channel(eid, one=None, probe=None, aligned=False, dataset_types=None,
                                    spike_sorter=None, brain_atlas=None, nested=True, return_collection=False):
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
    return_collection: bool
        Returns an extra argument with the collection chosen

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
    _logger.warning('Deprecation warning: brainbox.io.one.load_spike_sorting will be removed in future versions.'
                    'Use brainbox.io.one.SpikeSortingLoader instead')
    one = one or ONE()
    brain_atlas = brain_atlas or AllenAtlas()
    spikes, clusters, collection = load_spike_sorting(
        eid, one=one, probe=probe, dataset_types=dataset_types, spike_sorter=spike_sorter, return_collection=True)
    # -- Get brain regions and assign to clusters
    channels = load_channel_locations(eid, one=one, probe=probe, aligned=aligned,
                                      brain_atlas=brain_atlas)
    clusters = merge_clusters_channels(clusters, channels, keys_to_add_extra=None)
    if nested is False and len(spikes.keys()) == 1:
        k = list(spikes.keys())[0]
        channels = channels[k]
        clusters = clusters[k]
        spikes = spikes[k]
    if return_collection:
        return spikes, clusters, channels, collection
    else:
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
    one : one.api.OneAlyx, optional
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
        depths = trace_header(version=1)[:, 1]
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


@dataclass
class SpikeSortingLoader:
    """
    Object that will load spike sorting data for a given probe insertion.
    This class can be instantiated in several manners
    - With Alyx database probe id:
            SpikeSortingLoader(pid=pid, one=one)
    - With Alyx database eic and probe name:
            SpikeSortingLoader(eid=eid, pname='probe00', one=one)
    - From a local session and probe name:
            SpikeSortingLoader(session_path=session_path, pname='probe00')
    NB: When no ONE instance is passed, any datasets that are loaded will not be recorded.
    """
    one: One = None
    atlas: None = None
    pid: str = None
    eid: str = ''
    pname: str = ''
    session_path: Path = ''
    # the following properties are the outcome of the post init function
    collections: list = None
    datasets: list = None   # list of all datasets belonging to the session
    # the following properties are the outcome of a reading function
    files: dict = None
    collection: str = ''
    histology: str = ''  # 'alf', 'resolved', 'aligned' or 'traced'
    spike_sorter: str = 'pykilosort'
    spike_sorting_path: Path = None
    _sync: dict = None

    def __post_init__(self):
        # pid gets precedence
        if self.pid is not None:
            try:
                self.eid, self.pname = self.one.pid2eid(self.pid)
            except NotImplementedError:
                if self.eid == '' or self.pname == '':
                    raise IOError("Cannot infer session id and probe name from pid. "
                                  "You need to pass eid and pname explicitly when instantiating SpikeSortingLoader.")
            self.session_path = self.one.eid2path(self.eid)
        # then eid / pname combination
        elif self.session_path is None or self.session_path == '':
            self.session_path = self.one.eid2path(self.eid)
        # fully local providing a session path
        else:
            if self.one:
                self.eid = self.one.to_eid(self.session_path)
            else:
                self.one = One(cache_dir=self.session_path.parents[2], mode='local')
                df_sessions = cache._make_sessions_df(self.session_path)
                self.one._cache['sessions'] = df_sessions.set_index('id')
                self.one._cache['datasets'] = cache._make_datasets_df(self.session_path, hash_files=False)
                self.eid = str(self.session_path.relative_to(self.session_path.parents[2]))
        # populates default properties
        self.collections = self.one.list_collections(
            self.eid, filename='spikes*', collection=f"alf/{self.pname}*")
        self.datasets = self.one.list_datasets(self.eid)
        if self.atlas is None:
            self.atlas = AllenAtlas()
        self.files = {}

    @staticmethod
    def _get_attributes(dataset_types):
        """returns attributes to load for spikes and clusters objects"""
        if dataset_types is None:
            return SPIKES_ATTRIBUTES, CLUSTERS_ATTRIBUTES
        else:
            spike_attributes = [sp.split('.')[1] for sp in dataset_types if 'spikes.' in sp]
            cluster_attributes = [cl.split('.')[1] for cl in dataset_types if 'clusters.' in cl]
            spike_attributes = list(set(SPIKES_ATTRIBUTES + spike_attributes))
            cluster_attributes = list(set(CLUSTERS_ATTRIBUTES + cluster_attributes))
            return spike_attributes, cluster_attributes

    def _get_spike_sorting_collection(self, spike_sorter='pykilosort'):
        """
        Filters a list or array of collections to get the relevant spike sorting dataset
        if there is a pykilosort, load it
        """
        collection = next(filter(lambda c: c == f'alf/{self.pname}/{spike_sorter}', self.collections), None)
        # otherwise, prefers the shortest
        collection = collection or next(iter(sorted(filter(lambda c: f'alf/{self.pname}' in c, self.collections), key=len)), None)
        _logger.debug(f"selecting: {collection} to load amongst candidates: {self.collections}")
        return collection

    def download_spike_sorting_object(self, obj, spike_sorter='pykilosort', dataset_types=None, collection=None,
                                      missing='raise', **kwargs):
        """
        Downloads an ALF object
        :param obj: object name, str between 'spikes', 'clusters' or 'channels'
        :param spike_sorter: (defaults to 'pykilosort')
        :param dataset_types: list of extra dataset types, for example ['spikes.samples']
        :param collection: string specifiying the collection, for example 'alf/probe01/pykilosort'
        :param kwargs: additional arguments to be passed to one.api.One.load_object
        :param missing: 'raise' (default) or 'ignore'
        :return:
        """
        if len(self.collections) == 0:
            return {}, {}, {}
        self.collection = self._get_spike_sorting_collection(spike_sorter=spike_sorter)
        collection = collection or self.collection
        _logger.debug(f"loading spike sorting object {obj} from {collection}")
        spike_attributes, cluster_attributes = self._get_attributes(dataset_types)
        attributes = {'spikes': spike_attributes, 'clusters': cluster_attributes}
        try:
            self.files[obj] = self.one.load_object(
                self.eid, obj=obj, attribute=attributes.get(obj, None),
                collection=collection, download_only=True, **kwargs)
        except ALFObjectNotFound as e:
            if missing == 'raise':
                raise e

    def download_spike_sorting(self, **kwargs):
        """
        Downloads spikes, clusters and channels
        :param spike_sorter: (defaults to 'pykilosort')
        :param dataset_types: list of extra dataset types
        :return:
        """
        for obj in ['spikes', 'clusters', 'channels']:
            self.download_spike_sorting_object(obj=obj, **kwargs)
        self.spike_sorting_path = self.files['spikes'][0].parent

    def load_channels(self, **kwargs):
        """
        Loads channels
        The channel locations can come from several sources, it will load the most advanced version of the histology available,
        regardless of the spike sorting version loaded. The steps are (from most advanced to fresh out of the imaging):
        -   alf: the final version of channel locations, same as resolved with the difference that data is on file
        -   resolved: channel locations alignments have been agreed upon
        -   aligned: channel locations have been aligned, but review or other alignments are pending, potentially not accurate
        -   traced: the histology track has been recovered from microscopy, however the depths may not match, inaccurate data

        :param spike_sorter: (defaults to 'pykilosort')
        :param dataset_types: list of extra dataset types
        :return:
        """
        # we do not specify the spike sorter on purpose here: the electrode sites do not depend on the spike sorting
        self.download_spike_sorting_object(obj='electrodeSites', collection=f'alf/{self.pname}', missing='ignore')
        if 'electrodeSites' in self.files:
            channels = alfio.load_object(self.files['electrodeSites'], wildcards=self.one.wildcards)
        else:  # otherwise, we try to load the channel object from the spike sorting folder - this may not contain histology
            self.download_spike_sorting_object(obj='channels', **kwargs)
            channels = alfio.load_object(self.files['channels'], wildcards=self.one.wildcards)
        if 'brainLocationIds_ccf_2017' not in channels:
            _logger.debug(f"loading channels from alyx for {self.files['channels']}")
            _channels, self.histology = _load_channel_locations_traj(
                self.eid, probe=self.pname, one=self.one, brain_atlas=self.atlas, return_source=True, aligned=True)
            if _channels:
                channels = _channels[self.pname]
        else:
            channels = _channels_alf2bunch(channels, brain_regions=self.atlas.regions)
            self.histology = 'alf'
        return channels

    def load_spike_sorting(self, spike_sorter='pykilosort', **kwargs):
        """
        Loads spikes, clusters and channels

        There could be several spike sorting collections, by default the loader will get the pykilosort collection

        The channel locations can come from several sources, it will load the most advanced version of the histology available,
        regardless of the spike sorting version loaded. The steps are (from most advanced to fresh out of the imaging):
        -   alf: the final version of channel locations, same as resolved with the difference that data is on file
        -   resolved: channel locations alignments have been agreed upon
        -   aligned: channel locations have been aligned, but review or other alignments are pending, potentially not accurate
        -   traced: the histology track has been recovered from microscopy, however the depths may not match, inaccurate data

        :param spike_sorter: (defaults to 'pykilosort')
        :param dataset_types: list of extra dataset types
        :return:
        """
        if len(self.collections) == 0:
            return {}, {}, {}
        self.files = {}
        self.spike_sorter = spike_sorter
        self.download_spike_sorting(spike_sorter=spike_sorter, **kwargs)
        channels = self.load_channels(spike_sorter=spike_sorter, **kwargs)
        clusters = alfio.load_object(self.files['clusters'], wildcards=self.one.wildcards)
        spikes = alfio.load_object(self.files['spikes'], wildcards=self.one.wildcards)

        return spikes, clusters, channels

    @staticmethod
    def compute_metrics(spikes, clusters=None):
        nc = clusters['channels'].size if clusters else np.unique(spikes['clusters']).size
        metrics = pd.DataFrame(quick_unit_metrics(
            spikes['clusters'], spikes['times'], spikes['amps'], spikes['depths'], cluster_ids=np.arange(nc)))
        return metrics

    @staticmethod
    def merge_clusters(spikes, clusters, channels, cache_dir=None, compute_metrics=False):
        """
        Merge the metrics and the channel information into the clusters dictionary
        :param spikes:
        :param clusters:
        :param channels:
        :param cache_dir: if specified, will look for a cached parquet file to speed up. This is to be used
         for clusters or analysis applications (defaults to None).
        :param compute_metrics: if True, will explicitly recompute metrics (defaults to false)
        :return: cluster dictionary containing metrics and histology
        """
        if spikes == {}:
            return
        nc = clusters['channels'].size
        # recompute metrics if they are not available
        metrics = None
        if 'metrics' in clusters:
            metrics = clusters.pop('metrics')
            if metrics.shape[0] != nc:
                metrics = None
        if metrics is None or compute_metrics is True:
            _logger.debug("recompute clusters metrics")
            metrics = SpikeSortingLoader.compute_metrics(spikes, clusters)
            if isinstance(cache_dir, Path):
                metrics.to_parquet(Path(cache_dir).joinpath('clusters.metrics.pqt'))
        for k in metrics.keys():
            clusters[k] = metrics[k].to_numpy()
        for k in channels.keys():
            clusters[k] = channels[k][clusters['channels']]
        if cache_dir is not None:
            _logger.debug(f'caching clusters metrics in {cache_dir}')
            pd.DataFrame(clusters).to_parquet(Path(cache_dir).joinpath('clusters.pqt'))
        return clusters

    @property
    def url(self):
        """Gets flatiron URL for the session"""
        webclient = getattr(self.one, '_web_client', None)
        return webclient.rel_path2url(get_alf_path(self.session_path)) if webclient else None

    def samples2times(self, values, direction='forward'):
        """
        :param values: numpy array of times in seconds or samples to resync
        :param direction: 'forward' (samples probe time to seconds main time) or 'reverse'
         (seconds main time to samples probe time)
        :return:
        """
        if self._sync is None:
            timestamps = self.one.load_dataset(
                self.eid, dataset='_spikeglx_*.timestamps.npy', collection=f'raw_ephys_data/{self.pname}')
            self._sync = {
                'timestamps': timestamps,
                'forward': interp1d(timestamps[:, 0], timestamps[:, 1], fill_value='extrapolate'),
                'reverse': interp1d(timestamps[:, 1], timestamps[:, 0], fill_value='extrapolate'),
            }
        return self._sync[direction](values)

    @property
    def pid2ref(self):
        return f"{self.one.eid2ref(self.eid, as_dict=False)}_{self.pname}"

    def raster(self, spikes, channels, save_dir=None, br=None, label='raster'):
        """
        :param spikes: spikes dictionary
        :param save_dir: optional if specified
        :return:
        """
        br = br or BrainRegions()
        fig, axs = plt.subplots(2, 2, gridspec_kw={
            'width_ratios': [.95, .05], 'height_ratios': [.1, .9]}, figsize=(16, 9), sharex='col')
        axs[0, 1].set_axis_off()
        # axs[0, 0].set_xticks([])
        brainbox.plot.driftmap(spikes['times'], spikes['depths'], t_bin=0.007, d_bin=10, vmax=0.5, ax=axs[1, 0])
        title_str = f"{self.pid2ref}, {self.pid} \n" \
                    f"{spikes['clusters'].size:_} spikes, {np.unique(spikes['clusters']).size:_} clusters"
        axs[0, 0].title.set_text(title_str)
        if 'atlas_id' in channels:
            plot_brain_regions(channels['atlas_id'], channel_depths=channels['axial_um'],
                               brain_regions=br, display=True, ax=axs[1, 1], title=self.histology)
        axs[1, 0].set_ylim(0, 3800)
        axs[1, 0].set_xlim(spikes['times'][0], spikes['times'][-1])
        fig.tight_layout()

        self.download_spike_sorting_object('drift', self.spike_sorter, missing='ignore')
        if 'drift' in self.files:
            drift = alfio.load_object(self.files['drift'], wildcards=self.one.wildcards)
            axs[0, 0].plot(drift['times'], drift['um'], 'k', alpha=.5)

        if save_dir is not None:
            png_file = save_dir.joinpath(f"{self.pid}_{self.pid2ref}_{label}.png") if Path(save_dir).is_dir() else Path(save_dir)
            fig.savefig(png_file)
            plt.close(fig)
            gc.collect()
        else:
            return fig, axs


@dataclass
class SessionLoader:
    """
    Object to load session data for a give session in the recommended way.

    Parameters
    ----------
    one: one.api.ONE instance
        Can be in remote or local mode (required)
    session_path: string or pathlib.Path
        The absolute path to the session (one of session_path or eid is required)
    eid: string
        database UUID of the session (one of session_path or eid is required)

    If both are provided, session_path takes precedence over eid.

    Examples
    --------
    1) Load all available session data for one session:
        >>> from one.api import ONE
        >>> from brainbox.io.one import SessionLoader
        >>> one = ONE()
        >>> sess_loader = SessionLoader(one=one, session_path='/mnt/s0/Data/Subjects/cortexlab/KS022/2019-12-10/001/')
        # Object is initiated, but no data is loaded as you can see in the data_info attribute
        >>> sess_loader.data_info
                    name  is_loaded
        0         trials      False
        1          wheel      False
        2           pose      False
        3  motion_energy      False
        4          pupil      False

        # Loading all available session data, the data_info attribute now shows which data has been loaded
        >>> sess_loader.load_session_data()
        >>> sess_loader.data_info
                    name  is_loaded
        0         trials       True
        1          wheel       True
        2           pose       True
        3  motion_energy       True
        4          pupil      False

        # The data is loaded in pandas dataframes that you can access via the respective attributes, e.g.
        >>> type(sess_loader.trials)
        pandas.core.frame.DataFrame
        >>> sess_loader.trials.shape
        (626, 18)
        # Each data comes with its own timestamps in a column called 'times'
        >>> sess_loader.wheel['times']
        0             0.134286
        1             0.135286
        2             0.136286
        3             0.137286
        4             0.138286
              ...
        # For camera data (pose, motionEnergy) the respective functions load the data into one dataframe per camera.
        # The dataframes of all cameras are collected in a dictionary
        >>> type(sess_loader.pose)
        dict
        >>> sess_loader.pose.keys()
        dict_keys(['leftCamera', 'rightCamera', 'bodyCamera'])
        >>> sess_loader.pose['bodyCamera'].columns
        Index(['times', 'tail_start_x', 'tail_start_y', 'tail_start_likelihood'], dtype='object')
        # In order to control the loading of specific data by e.g. specifying parameters, use the individual loading
        functions:
        >>> sess_loader.load_wheel(sampling_rate=100)
    """
    one: One = None
    session_path: Path = ''
    eid: str = ''
    data_info: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    trials: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    wheel: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    pose: dict = field(default_factory=dict, repr=False)
    motion_energy: dict = field(default_factory=dict, repr=False)
    pupil: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def __post_init__(self):
        """
        Function that runs automatically after initiation of the dataclass attributes.
        Checks for required inputs, sets session_path and eid, creates data_info table.
        """
        if self.one is None:
            raise ValueError("An input to one is required. If not connection to a database is desired, it can be "
                             "a fully local instance of One.")
        # If session path is given, takes precedence over eid
        if self.session_path is not None and self.session_path != '':
            self.eid = self.one.to_eid(self.session_path)
            self.session_path = Path(self.session_path)
        # Providing no session path, try to infer from eid
        else:
            if self.eid is not None and self.eid != '':
                self.session_path = self.one.eid2path(self.eid)
            else:
                raise ValueError("If no session path is given, eid is required.")

        data_names = [
            'trials',
            'wheel',
            'pose',
            'motion_energy',
            'pupil'
        ]
        self.data_info = pd.DataFrame(columns=['name', 'is_loaded'], data=zip(data_names, [False] * len(data_names)))

    def load_session_data(self, trials=True, wheel=True, pose=True, motion_energy=True, pupil=True, reload=False):
        """
        Function to load available session data into the SessionLoader object. Input parameters allow to control which
        data is loaded. Data is loaded into an attribute of the SessionLoader object with the same name as the input
        parameter (e.g. SessionLoader.trials, SessionLoader.pose). Information about which data is loaded is stored
        in SessionLoader.data_info

        Parameters
        ----------
        trials: boolean
            Whether to load all trials data into SessionLoader.trials, default is True
        wheel: boolean
            Whether to load wheel data (position, velocity, acceleration) into SessionLoader.wheel, default is True
        pose: boolean
            Whether to load pose tracking results (DLC) for each available camera into SessionLoader.pose,
            default is True
        motion_energy: boolean
            Whether to load motion energy data (whisker pad for left/right camera, body for body camera)
            into SessionLoader.motion_energy, default is True
        pupil: boolean
            Whether to load pupil diameter (raw and smooth) for the left/right camera into SessionLoader.pupil,
            default is True
        reload: boolean
            Whether to reload data that has already been loaded into this SessionLoader object, default is False
        """
        load_df = self.data_info.copy()
        load_df['to_load'] = [
            trials,
            wheel,
            pose,
            motion_energy,
            pupil
        ]
        load_df['load_func'] = [
            self.load_trials,
            self.load_wheel,
            self.load_pose,
            self.load_motion_energy,
            self.load_pupil
        ]

        for idx, row in load_df.iterrows():
            if row['to_load'] is False:
                _logger.debug(f"Not loading {row['name']} data, set to False.")
            elif row['is_loaded'] is True and reload is False:
                _logger.debug(f"Not loading {row['name']} data, is already loaded and reload=False.")
            else:
                try:
                    _logger.info(f"Loading {row['name']} data")
                    row['load_func']()
                    self.data_info.loc[idx, 'is_loaded'] = True
                except BaseException as e:
                    _logger.warning(f"Could not load {row['name']} data.")
                    _logger.debug(e)

    def load_trials(self):
        """
        Function to load trials data into SessionLoader.trials
        """
        # itiDuration frequently has a mismatched dimension, and we don't need it, exclude using regex
        self.one.wildcards = False
        self.trials = self.one.load_object(self.eid, 'trials', collection='alf', attribute=r'(?!itiDuration).*').to_df()
        self.one.wildcards = True
        self.data_info.loc[self.data_info['name'] == 'trials', 'is_loaded'] = True

    def load_wheel(self, fs=1000, corner_frequency=20, order=8):
        """
        Function to load wheel data (position, velocity, acceleration) into SessionLoader.wheel. The wheel position
        is first interpolated to a uniform sampling rate. Then velocity and acceleration are computed, during which
        a Butterworth low-pass filter is applied.

        Parameters
        ----------
        fs: int, float
            Sampling frequency for the wheel position, default is 1000 Hz
        corner_frequency: int, float
            Corner frequency of Butterworth low-pass filter, default is 20
        order: int, float
            Order of Butterworth low_pass filter, default is 8
        """
        wheel_raw = self.one.load_object(self.eid, 'wheel')
        if wheel_raw['position'].shape[0] != wheel_raw['timestamps'].shape[0]:
            raise ValueError("Length mismatch between 'wheel.position' and 'wheel.timestamps")
        # resample the wheel position and compute velocity, acceleration
        self.wheel = pd.DataFrame(columns=['times', 'position', 'velocity', 'acceleration'])
        self.wheel['position'], self.wheel['times'] = interpolate_position(
            wheel_raw['timestamps'], wheel_raw['position'], freq=fs)
        self.wheel['velocity'], self.wheel['acceleration'] = velocity_filtered(
            self.wheel['position'], fs=fs, corner_frequency=corner_frequency, order=order)
        self.wheel = self.wheel.apply(np.float32)
        self.data_info.loc[self.data_info['name'] == 'wheel', 'is_loaded'] = True

    def load_pose(self, likelihood_thr=0.9, views=['left', 'right', 'body']):
        """
        Function to load the pose estimation results (DLC) into SessionLoader.pose. SessionLoader.pose is a
        dictionary where keys are the names of the cameras for which pose data is loaded, and values are pandas
        Dataframes with the timestamps and pose data, one row for each body part tracked for that camera.

        Parameters
        ----------
        likelihood_thr: float
            The position of each tracked body part come with a likelihood of that estimate for each time point.
            Estimates for time points with likelihood < likelihood_thr are set to NaN. To skip thresholding set
            likelihood_thr=1. Default is 0.9
        views: list
            List of camera views for which to try and load data. Possible options are {'left', 'right', 'body'}
        """
        # empty the dictionary so that if one loads only one view, after having loaded several, the others don't linger
        self.pose = {}
        for view in views:
            pose_raw = self.one.load_object(self.eid, f'{view}Camera', attribute=['dlc', 'times'])
            # Double check if video timestamps are correct length or can be fixed
            times_fixed, dlc = self._check_video_timestamps(view, pose_raw['times'], pose_raw['dlc'])
            self.pose[f'{view}Camera'] = likelihood_threshold(dlc, likelihood_thr)
            self.pose[f'{view}Camera'].insert(0, 'times', times_fixed)
            self.data_info.loc[self.data_info['name'] == 'pose', 'is_loaded'] = True

    def load_motion_energy(self, views=['left', 'right', 'body']):
        """
        Function to load the motion energy data into SessionLoader.motion_energy. SessionLoader.motion_energy is a
        dictionary where keys are the names of the cameras for which motion energy data is loaded, and values are
        pandas Dataframes with the timestamps and motion energy data.
        The motion energy for the left and right camera is calculated for a square roughly covering the whisker pad
        (whiskerMotionEnergy). The motion energy for the body camera is calculated for a square covering much of the
        body (bodyMotionEnergy).

        Parameters
        ----------
        views: list
            List of camera views for which to try and load data. Possible options are {'left', 'right', 'body'}
        """
        names = {'left': 'whiskerMotionEnergy',
                 'right': 'whiskerMotionEnergy',
                 'body': 'bodyMotionEnergy'}
        # empty the dictionary so that if one loads only one view, after having loaded several, the others don't linger
        self.motion_energy = {}
        for view in views:
            me_raw = self.one.load_object(self.eid, f'{view}Camera', attribute=['ROIMotionEnergy', 'times'])
            # Double check if video timestamps are correct length or can be fixed
            times_fixed, motion_energy = self._check_video_timestamps(
                view, me_raw['times'], me_raw['ROIMotionEnergy'])
            self.motion_energy[f'{view}Camera'] = pd.DataFrame(columns=[names[view]], data=motion_energy)
            self.motion_energy[f'{view}Camera'].insert(0, 'times', times_fixed)
            self.data_info.loc[self.data_info['name'] == 'motion_energy', 'is_loaded'] = True

    def load_licks(self):
        """
        Not yet implemented
        """
        pass

    def load_pupil(self, snr_thresh=5.):
        """
        Function to load raw and smoothed pupil diameter data from the left camera into SessionLoader.pupil.

        Parameters
        ----------
        snr_thresh: float
            An SNR is calculated from the raw and smoothed pupil diameter. If this snr < snr_thresh the data
            will be considered unusable and will be discarded.
        """
        # Try to load from features
        feat_raw = self.one.load_object(self.eid, 'leftCamera', attribute=['times', 'features'])
        if 'features' in feat_raw.keys():
            times_fixed, feats = self._check_video_timestamps('left', feat_raw['times'], feat_raw['features'])
            self.pupil = feats.copy()
            self.pupil.insert(0, 'times', times_fixed)

        # If unavailable compute on the fly
        else:
            _logger.info('Pupil diameter not available, trying to compute on the fly.')
            if (self.data_info[self.data_info['name'] == 'pose']['is_loaded'].values[0]
                    and 'leftCamera' in self.pose.keys()):
                # If pose data is already loaded, we don't know if it was threshold at 0.9, so we need a little stunt
                copy_pose = self.pose['leftCamera'].copy()  # Save the previously loaded pose data
                self.load_pose(views=['left'], likelihood_thr=0.9)  # Load new with threshold 0.9
                dlc_thr = self.pose['leftCamera'].copy()  # Save the threshold pose data in new variable
                self.pose['leftCamera'] = copy_pose.copy()  # Get previously loaded pose data back in place
            else:
                self.load_pose(views=['left'], likelihood_thr=0.9)
                dlc_thr = self.pose['leftCamera'].copy()

            self.pupil['pupilDiameter_raw'] = get_pupil_diameter(dlc_thr)
            try:
                self.pupil['pupilDiameter_smooth'] = get_smooth_pupil_diameter(self.pupil['pupilDiameter_raw'], 'left')
            except BaseException as e:
                _logger.error("Loaded raw pupil diameter but computing smooth pupil diameter failed. "
                              "Saving all NaNs for pupilDiameter_smooth.")
                _logger.debug(e)
                self.pupil['pupilDiameter_smooth'] = np.nan

        if not np.all(np.isnan(self.pupil['pupilDiameter_smooth'])):
            good_idxs = np.where(
                ~np.isnan(self.pupil['pupilDiameter_smooth']) & ~np.isnan(self.pupil['pupilDiameter_raw']))[0]
            snr = (np.var(self.pupil['pupilDiameter_smooth'][good_idxs]) /
                   (np.var(self.pupil['pupilDiameter_smooth'][good_idxs] - self.pupil['pupilDiameter_raw'][good_idxs])))
            if snr < snr_thresh:
                self.pupil = pd.DataFrame()
                raise ValueError(f'Pupil diameter SNR ({snr:.2f}) below threshold SNR ({snr_thresh}), removing data.')

    def _check_video_timestamps(self, view, video_timestamps, video_data):
        """
        Helper function to check for the length of the video frames vs video timestamps and fix in case
        timestamps are longer than video frames.
        """
        # If camera times are shorter than video data, or empty, no current fix
        if video_timestamps.shape[0] < video_data.shape[0]:
            if video_timestamps.shape[0] == 0:
                msg = f'Camera times empty for {view}Camera.'
            else:
                msg = f'Camera times are shorter than video data for {view}Camera.'
            _logger.warning(msg)
            raise ValueError(msg)
        # For pre-GPIO sessions, it is possible that the camera times are longer than the actual video.
        # This is because the first few frames are sometimes not recorded. We can remove the first few
        # timestamps in this case
        elif video_timestamps.shape[0] > video_data.shape[0]:
            video_timestamps_fixed = video_timestamps[-video_data.shape[0]:]
            return video_timestamps_fixed, video_data
        else:
            return video_timestamps, video_data

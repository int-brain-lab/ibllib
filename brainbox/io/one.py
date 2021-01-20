from pathlib import Path
import logging
import numpy as np
import pandas as pd
import os

import alf.io
from ibllib.io import spikeglx
from ibllib.atlas import regions_from_allen_csv
from ibllib.io.extractors.training_wheel import extract_wheel_moves, extract_first_movement_times
from oneibl.one import ONE

from brainbox.core import Bunch, TimeSeries
from brainbox.processing import sync

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
        insertions = one.alyx.rest('insertions', 'list', session=eid, name=probe)[0]
        labels = [probe]
        if not insertions['json']:
            tracing = [False]
            resolved = [False]
            counts = [0]
        else:
            tracing = [(insertions.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                       get('tracing_exists', False))]
            resolved = [(insertions.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                        get('alignment_resolved', False))]
            counts = [(insertions.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                      get('alignment_count', 0))]
        probe_id = [insertions['id']]
    # No specific probe specified, load any that is available
    # Need to catch for the case where we have two of the same probe insertions
    else:
        insertions = one.alyx.rest('insertions', 'list', session=eid)
        labels = [ins['name'] for ins in insertions]
        try:
            tracing = [ins.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                       get('tracing_exists', False) for ins in insertions]
            resolved = [ins.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                        get('alignment_resolved', False) for ins in insertions]
            counts = [ins.get('json', {'temp': 0}).get('extended_qc', {'temp': 0}).
                      get('alignment_count', 0) for ins in insertions]
        except Exception:
            tracing = [False for ins in insertions]
            resolved = [False for ins in insertions]
            counts = [0 for ins in insertions]

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

                # If we have successfully downloaded the data
                if 'brainLocationIds_ccf_2017' in chans.keys():

                    channels[label] = Bunch({
                        'atlas_id': chans['brainLocationIds_ccf_2017'],
                        'acronym': r.get(chans['brainLocationIds_ccf_2017'])['acronym'],
                        'x': chans['mlapdv'][:, 0] / 1e6,
                        'y': chans['mlapdv'][:, 1] / 1e6,
                        'z': chans['mlapdv'][:, 2] / 1e6,
                        'axial_um': chans['localCoordinates'][:, 1],
                        'lateral_um': chans['localCoordinates'][:, 0]
                    })
                # Otherwise we just get the channels from alyx. Shouldn't happen often, only if
                # data is still inbetween ftp and flatiron after being resolved
                else:
                    traj_id = one.alyx.rest('trajectories', 'list', session=eid, probe_name=label,
                                            provenance='Ephys aligned histology track')[0]['id']
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
                    'atlas_id': np.array([ch['brain_region'] for ch in chans]),
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
                                        provenance='Histology track')[0]['id']
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
            logger.warning(f'Histology tracing for {label} does not exist. '
                           f'No channels for {label}')

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


def load_spike_sorting(eid, one=None, probe=None, dataset_types=None, force=False):
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
    :param probe: name of probe to load in, if not given all probes for session will be loaded
    :param dataset_types: additional spikes/clusters objects to add to the standard default list
    :param force: by default function looks for data on local computer and loads this in. If you
    want to connect to database and make sure files are still the same set force=True
    :return: spikes, clusters (dict of bunch, 1 bunch per probe)
    """
    if isinstance(eid, Path):
        # Do everything locally without ONE
        session_path = eid
        if isinstance(probe, str):
            labels = [probe]
        else:
            probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')
            labels = [pr['label'] for pr in probes['description']]
        spikes = Bunch({})
        clusters = Bunch({})
        for label in labels:
            _spikes, _clusters = _load_spike_sorting_local(session_path, label)
            spikes[label] = _spikes
            clusters[label] = _clusters

        return spikes, clusters

    else:
        session_path = one.path_from_eid(eid)
        if not session_path:
            logger.warning('Session not found')
            return (None, None), 'no session path'

        one = one or ONE()
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
            # Append extra optional DS
            dtypes = list(set(dataset_types + dtypes_default))

        if isinstance(probe, str):
            labels = [probe]
        else:
            insertions = one.alyx.rest('insertions', 'list', session=eid)
            labels = [ins['name'] for ins in insertions]

        spikes = Bunch({})
        clusters = Bunch({})
        for label in labels:
            _spikes, _clusters = _load_spike_sorting_local(session_path, label)
            spike_dtypes = [sp for sp in dtypes if 'spikes.' in sp]
            spike_local = ['spikes.' + sp for sp in list(_spikes.keys())]
            spike_exists = all([sp in spike_local for sp in spike_dtypes])
            cluster_dtypes = [cl for cl in dtypes if 'clusters.' in cl]
            cluster_local = ['clusters.' + cl for cl in list(_clusters.keys())]
            cluster_exists = all([cl in cluster_local for cl in cluster_dtypes])

            if not spike_exists or not cluster_exists or force:
                logger.info(f'Did not find local files for spikes and clusters for {session_path} '
                            f'and {label}. Downloading....')
                one.load(eid, dataset_types=dtypes, download_only=True)
                _spikes, _clusters = _load_spike_sorting_local(session_path, label)
                if not _spikes:
                    logger.warning(
                        f'Could not load spikes datasets for session {session_path} and {label}. '
                        f'Spikes for {probe} will return an empty dict')
                else:
                    spikes[label] = _spikes
                if not _clusters:
                    logger.warning(
                        f'Could not load clusters datasets for session {session_path} and {label}.'
                        f' Clusters for {label} will return an empty dict')
                else:
                    clusters[label] = _clusters
            else:
                logger.info(f'Local files for spikes and clusters for {session_path} '
                            f'and {label} found. To re-download set force=True')

                spikes[label] = _spikes
                clusters[label] = _clusters

        return spikes, clusters


def _load_spike_sorting_local(session_path, probe):
    # gets clusters and spikes from a local session folder
    probe_path = session_path.joinpath('alf', probe)

    # look for clusters.metrics.csv file, if it exists delete as we now have .pqt file instead
    cluster_file = probe_path.joinpath('clusters.metrics.csv')

    if cluster_file.exists():
        os.remove(cluster_file)
        logger.info('Deleting old clusters.metrics.csv file')

    try:
        spikes = alf.io.load_object(probe_path, object='spikes')
    except Exception:
        spikes = {}
    try:
        clusters = alf.io.load_object(probe_path, object='clusters')
    except Exception:
        clusters = {}

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
    keys_to_add_default = ['acronym', 'atlas_id', 'x', 'y', 'z']

    if keys_to_add_extra is None:
        keys_to_add = keys_to_add_default
    else:
        #  Append extra optional keys
        keys_to_add = list(set(keys_to_add_extra + keys_to_add_default))

    for label in probe_labels:
        try:
            clu_ch = dic_clus[label]['channels']

            for key in keys_to_add:
                assert key in channels[label].keys()  # Check key is in channels
                ch_key = channels[label][key]

                if max(clu_ch) < len(ch_key):  # Check length as will use clu_ch as index
                    dic_clus[label][key] = ch_key[clu_ch]
                else:
                    print(f'Channels in probe {label} does not have'
                          f' the right element number compared to cluster.'
                          f' Data in new cluster key {key} is thus returned empty.')
                    dic_clus[label][key] = []
        except KeyError:
            logger.warning(
                f'Either clusters or channels does not have key {label}, could not'
                f' merge')
            continue

    return dic_clus


def load_spike_sorting_with_channel(eid, one=None, probe=None, dataset_types=None, aligned=False,
                                    force=False):
    """
    For a given eid, get spikes, clusters and channels information, and merges clusters
    and channels information before returning all three variables.
    :param eid:
    :param one:
    :param dataset_types: additional dataset_types to load
    :param aligned: whether to get the latest user aligned channel when not resolved or use
    histology track
    :return: spikes, clusters, channels (dict of bunch, 1 bunch per probe)
    """
    # --- Get spikes and clusters data
    one = one or ONE()
    dic_spk_bunch, dic_clus = load_spike_sorting(eid, one=one, probe=probe,
                                                 dataset_types=dataset_types, force=force)
    # -- Get brain regions and assign to clusters
    channels = load_channel_locations(eid, one=one, probe=probe, aligned=aligned)

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


def load_trials_df(eid, one=None, maxlen=None, t_before=0., t_after=0., ret_wheel=False,
                   ret_abswheel=False, wheel_binsize=0.02):
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
    eid : str
        Session UUID string to pass to ONE
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
    trialstypes = ['trials.choice',
                   'trials.probabilityLeft',
                   'trials.feedbackType',
                   'trials.feedback_times',
                   'trials.contrastLeft',
                   'trials.contrastRight',
                   'trials.goCue_times',
                   'trials.stimOn_times', ]

    # A quick function to remap probabilities in those sessions where it was not computed correctly
    def remap_trialp(probs):
        # Block probabilities in trial data aren't accurate and need to be remapped
        validvals = np.array([0.2, 0.5, 0.8])
        diffs = np.abs(np.array([x - validvals for x in probs]))
        maps = diffs.argmin(axis=1)
        return validvals[maps]

    starttimes = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    endtimes = one.load(eid, dataset_types=['trials.feedback_times'])[0]
    tmp = one.load(eid, dataset_types=trialstypes)

    if maxlen is not None:
        with np.errstate(invalid='ignore'):
            keeptrials = (endtimes - starttimes) <= maxlen
    else:
        keeptrials = range(len(starttimes))
    trialdata = {x.split('.')[1]: tmp[i][keeptrials] for i, x in enumerate(trialstypes)}
    trialdata['probabilityLeft'] = remap_trialp(trialdata['probabilityLeft'])
    trialsdf = pd.DataFrame(trialdata)
    if maxlen is not None:
        trialsdf.set_index(np.nonzero(keeptrials)[0], inplace=True)
    trialsdf['trial_start'] = trialsdf['stimOn_times'] - t_before
    trialsdf['trial_end'] = trialsdf['feedback_times'] + t_after
    if not ret_wheel and not ret_abswheel:
        return trialsdf

    wheel = one.load_object(eid, 'wheel')
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

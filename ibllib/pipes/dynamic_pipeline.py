import logging
import re
from collections import OrderedDict
from pathlib import Path
from itertools import chain
import yaml

import spikeglx

import ibllib.io.session_params as sess_params
import ibllib.io.extractors.base
import ibllib.pipes.ephys_preprocessing as epp
import ibllib.pipes.tasks as mtasks
import ibllib.pipes.base_tasks as bstasks
import ibllib.pipes.widefield_tasks as wtasks
import ibllib.pipes.sync_tasks as stasks
import ibllib.pipes.behavior_tasks as btasks
import ibllib.pipes.video_tasks as vtasks
import ibllib.pipes.ephys_tasks as etasks
import ibllib.pipes.audio_tasks as atasks
from ibllib.pipes.photometry_tasks import TaskFibrePhotometryPreprocess, TaskFibrePhotometryRegisterRaw

_logger = logging.getLogger(__name__)


def acquisition_description_legacy_session(session_path, save=False):
    """
    From a legacy session create a dictionary corresponding to the acquisition description
    :return: dict
    """
    extractor_type = ibllib.io.extractors.base.get_session_extractor_type(session_path=session_path)
    etype2protocol = dict(biased='choice_world_biased', habituation='choice_world_habituation',
                          training='choice_world_training', ephys='choice_world_recording')
    dict_ad = get_acquisition_description(etype2protocol[extractor_type])
    if save:
        sess_params.write_params(session_path=session_path, data=dict_ad)
    return dict_ad


def get_acquisition_description(protocol):
    """"
    This is a set of example acquisition descriptions for experiments
    -   choice_world_recording
    -   choice_world_biased
    -   choice_world_training
    -   choice_world_habituation
    -   choice_world_passive
    That are part of the IBL pipeline
    """
    if protocol == 'choice_world_recording':   # canonical ephys
        devices = {
            'cameras': {
                'right': {'collection': 'raw_video_data', 'sync_label': 'audio'},
                'body': {'collection': 'raw_video_data', 'sync_label': 'audio'},
                'left': {'collection': 'raw_video_data', 'sync_label': 'audio'},
            },
            'neuropixel': {
                'probe00': {'collection': 'raw_ephys_data/probe00', 'sync_label': 'imec_sync'},
                'probe01': {'collection': 'raw_ephys_data/probe01', 'sync_label': 'imec_sync'}
            },
            'microphone': {
                'microphone': {'collection': 'raw_behavior_data', 'sync_label': None}
            },
        }
        acquisition_description = {  # this is the current ephys pipeline description
            'devices': devices,
            'tasks': [
                {'ephysChoiceWorld': {'collection': 'raw_behavior_data', 'sync_label': 'bpod'}},
                {'passiveChoiceWorld': {'collection': 'raw_passive_data', 'sync_label': 'bpod'}}
            ],
            'sync': {
                'nidq': {'collection': 'raw_ephys_data', 'extension': 'bin', 'acquisition_software': 'spikeglx'}
            },
            'procedures': ['Ephys recording with acute probe(s)'],
            'projects': ['ibl_neuropixel_brainwide_01']
        }
    else:
        devices = {
            'cameras': {
                'left': {'collection': 'raw_video_data', 'sync_label': 'frame2ttl'},
            },
            'microphone': {
                'microphone': {'collection': 'raw_behavior_data', 'sync_label': None}
            },
        }
        acquisition_description = {  # this is the current ephys pipeline description
            'devices': devices,
            'sync': {
                'bpod': {'collection': 'raw_behavior_data', 'extension': 'bin'}
            },
            'procedures': ['Behavior training/tasks'],
            'projects': ['ibl_neuropixel_brainwide_01']
        }
        if protocol == 'choice_world_biased':
            key = 'biasedChoiceWorld'
        elif protocol == 'choice_world_training':
            key = 'trainingChoiceWorld'
        elif protocol == 'choice_world_habituation':
            key = 'habituationChoiceWorld'
        else:
            raise ValueError(f'Unknown protocol "{protocol}"')
        acquisition_description['tasks'] = [{key: {
            'collection': 'raw_behavior_data',
            'sync_label': 'bpod', 'main': True
        }}]
    acquisition_description['version'] = '1.0.0'
    return acquisition_description


def make_pipeline(session_path, **pkwargs):
    """
    Creates a pipeline of extractor tasks from a session's experiment description file.

    Parameters
    ----------
    session_path : str, Path
        The absolute session path, i.e. '/path/to/subject/yyyy-mm-dd/nnn'.
    **pkwargs
        Optional arguments passed to the ibllib.pipes.tasks.Pipeline constructor.

    Returns
    -------
    ibllib.pipes.tasks.Pipeline
        A task pipeline object.
    """
    # NB: this pattern is a pattern for dynamic class creation
    # tasks['SyncPulses'] = type('SyncPulses', (epp.EphysPulses,), {})(session_path=session_path)
    if not session_path or not (session_path := Path(session_path)).exists():
        raise ValueError('Session path does not exist')
    tasks = OrderedDict()
    acquisition_description = sess_params.read_params(session_path)
    if not acquisition_description:
        raise ValueError('Experiment description file not found or is empty')
    devices = acquisition_description.get('devices', {})
    kwargs = {'session_path': session_path}

    # Registers the experiment description file
    tasks['ExperimentDescriptionRegisterRaw'] = type('ExperimentDescriptionRegisterRaw',
                                                     (bstasks.ExperimentDescriptionRegisterRaw,), {})(**kwargs)

    # Syncing tasks
    (sync, sync_args), = acquisition_description['sync'].items()
    sync_args['sync_collection'] = sync_args.pop('collection')  # rename the key so it matches task run arguments
    sync_args['sync_ext'] = sync_args.pop('extension')
    sync_args['sync_namespace'] = sync_args.pop('acquisition_software', None)
    sync_kwargs = {'sync': sync, **sync_args}
    sync_tasks = []
    if sync == 'nidq' and sync_args['sync_collection'] == 'raw_ephys_data':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (etasks.EphysSyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
        tasks[f'SyncPulses_{sync}'] = type(f'SyncPulses_{sync}', (etasks.EphysSyncPulses,), {})(
            **kwargs, **sync_kwargs, parents=[tasks['SyncRegisterRaw']])
        sync_tasks = [tasks[f'SyncPulses_{sync}']]
    elif sync == 'nidq':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (stasks.SyncMtscomp,), {})(**kwargs, **sync_kwargs)
        tasks[f'SyncPulses_{sync}'] = type(f'SyncPulses_{sync}', (stasks.SyncPulses,), {})(
            **kwargs, **sync_kwargs, parents=[tasks['SyncRegisterRaw']])
        sync_tasks = [tasks[f'SyncPulses_{sync}']]
    elif sync == 'tdms':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (stasks.SyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
    elif sync == 'bpod':
        pass
        # ATM we don't have anything for this not sure it will be needed in the future

    # Behavior tasks
    task_protocols = acquisition_description.get('tasks', [])
    for i, (protocol, task_info) in enumerate(chain(*map(dict.items, task_protocols))):
        collection = task_info.get('collection', f'raw_task_data_{i:02}')
        task_kwargs = {'protocol': protocol, 'collection': collection}
        # For now the order of protocols in the list will take precedence. If collections are numbered,
        # check that the numbers match the order.  This may change in the future.
        if re.match(r'^raw_task_data_\d{2}$', collection):
            task_kwargs['protocol_number'] = i
            if int(collection.split('_')[-1]) != i:
                _logger.warning('Number in collection name does not match task order')
        if extractors := task_info.get('extractors', False):
            extractors = (extractors,) if isinstance(extractors, str) else extractors
            task_name = None  # to avoid unbound variable issue in the first round
            for j, task in enumerate(extractors):
                # Assume previous task in the list is parent
                parents = [] if j == 0 else [tasks[task_name]]
                # Make sure extractor and sync task don't collide
                for sync_option in ('nidq', 'bpod'):
                    if sync_option in task.lower() and not sync == sync_option:
                        raise ValueError(f'Extractor "{task}" and sync "{sync}" do not match')
                try:
                    task = getattr(btasks, task)
                except AttributeError:
                    raise NotImplementedError  # TODO Attempt to import from personal project repo
                # Rename the class to something more informative
                task_name = f'{task.__name__}_{i:02}'
                # For now we assume that the second task in the list is always the trials extractor, which is dependent
                # on the sync task and sync arguments
                if j == 1:
                    tasks[task_name] = type(task_name, (task,), {})(
                        **kwargs, **sync_kwargs, **task_kwargs, parents=parents + sync_tasks
                    )
                else:
                    tasks[task_name] = type(task_name, (task,), {})(**kwargs, **task_kwargs, parents=parents)
                # For the next task, we assume that the previous task is the parent
        else:  # Legacy block to handle sessions without defined extractors
            # -   choice_world_recording
            # -   choice_world_biased
            # -   choice_world_training
            # -   choice_world_habituation
            if 'habituation' in protocol:
                registration_class = btasks.HabituationRegisterRaw
                behaviour_class = btasks.HabituationTrialsBpod
                compute_status = False
            elif 'passiveChoiceWorld' in protocol:
                registration_class = btasks.PassiveRegisterRaw
                behaviour_class = btasks.PassiveTask
                compute_status = False
            elif sync_kwargs['sync'] == 'bpod':
                registration_class = btasks.TrialRegisterRaw
                behaviour_class = btasks.ChoiceWorldTrialsBpod
                compute_status = True
            elif sync_kwargs['sync'] == 'nidq':
                registration_class = btasks.TrialRegisterRaw
                behaviour_class = btasks.ChoiceWorldTrialsNidq
                compute_status = True
            else:
                raise NotImplementedError
            tasks[f'RegisterRaw_{protocol}_{i:02}'] = type(f'RegisterRaw_{protocol}_{i:02}', (registration_class,), {})(
                **kwargs, **task_kwargs)
            parents = [tasks[f'RegisterRaw_{protocol}_{i:02}']] + sync_tasks
            tasks[f'Trials_{protocol}_{i:02}'] = type(f'Trials_{protocol}_{i:02}', (behaviour_class,), {})(
                **kwargs, **sync_kwargs, **task_kwargs, parents=parents)
            if compute_status:
                tasks[f"TrainingStatus_{protocol}_{i:02}"] = type(f'TrainingStatus_{protocol}_{i:02}', (
                    btasks.TrainingStatus,), {})(**kwargs, **task_kwargs, parents=[tasks[f'Trials_{protocol}_{i:02}']])

    # Ephys tasks
    if 'neuropixel' in devices:
        ephys_kwargs = {'device_collection': 'raw_ephys_data'}
        tasks['EphysRegisterRaw'] = type('EphysRegisterRaw', (etasks.EphysRegisterRaw,), {})(**kwargs, **ephys_kwargs)

        all_probes = []
        register_tasks = []
        for pname, probe_info in devices['neuropixel'].items():
            meta_file = spikeglx.glob_ephys_files(Path(session_path).joinpath(probe_info['collection']), ext='meta')
            meta_file = meta_file[0].get('ap')
            nptype = spikeglx._get_neuropixel_version_from_meta(spikeglx.read_meta_data(meta_file))
            nshanks = spikeglx._get_nshanks_from_meta(spikeglx.read_meta_data(meta_file))

            if (nptype == 'NP2.1') or (nptype == 'NP2.4' and nshanks == 1):
                tasks[f'EphyCompressNP21_{pname}'] = type(f'EphyCompressNP21_{pname}', (etasks.EphysCompressNP21,), {})(
                    **kwargs, **ephys_kwargs, pname=pname)
                all_probes.append(pname)
                register_tasks.append(tasks[f'EphyCompressNP21_{pname}'])
            elif nptype == 'NP2.4' and nshanks > 1:
                tasks[f'EphyCompressNP24_{pname}'] = type(f'EphyCompressNP24_{pname}', (etasks.EphysCompressNP24,), {})(
                    **kwargs, **ephys_kwargs, pname=pname, nshanks=nshanks)
                register_tasks.append(tasks[f'EphyCompressNP24_{pname}'])
                all_probes += [f'{pname}{chr(97 + int(shank))}' for shank in range(nshanks)]
            else:
                tasks[f'EphysCompressNP1_{pname}'] = type(f'EphyCompressNP1_{pname}', (etasks.EphysCompressNP1,), {})(
                    **kwargs, **ephys_kwargs, pname=pname)
                register_tasks.append(tasks[f'EphysCompressNP1_{pname}'])
                all_probes.append(pname)

        if nptype == '3A':
            tasks['EphysPulses'] = type('EphysPulses', (etasks.EphysPulses,), {})(
                **kwargs, **ephys_kwargs, **sync_kwargs, pname=all_probes, parents=register_tasks + sync_tasks)

        for pname in all_probes:
            register_task = [reg_task for reg_task in register_tasks if pname[:7] in reg_task.name]

            if nptype != '3A':
                tasks[f'EphysPulses_{pname}'] = type(f'EphysPulses_{pname}', (etasks.EphysPulses,), {})(
                    **kwargs, **ephys_kwargs, **sync_kwargs, pname=[pname], parents=register_task + sync_tasks)
                tasks[f'Spikesorting_{pname}'] = type(f'Spikesorting_{pname}', (etasks.SpikeSorting,), {})(
                    **kwargs, **ephys_kwargs, pname=pname, parents=[tasks[f'EphysPulses_{pname}']])
            else:
                tasks[f'Spikesorting_{pname}'] = type(f'Spikesorting_{pname}', (etasks.SpikeSorting,), {})(
                    **kwargs, **ephys_kwargs, pname=pname, parents=[tasks['EphysPulses']])

            tasks[f'RawEphysQC_{pname}'] = type(f'RawEphysQC_{pname}', (etasks.RawEphysQC,), {})(
                **kwargs, **ephys_kwargs, pname=pname, parents=register_task)
            tasks[f'EphysCellQC_{pname}'] = type(f'EphysCellQC_{pname}', (etasks.EphysCellsQc,), {})(
                **kwargs, **ephys_kwargs, pname=pname, parents=[tasks[f'Spikesorting_{pname}']])

    # Video tasks
    if 'cameras' in devices:
        video_kwargs = {'device_collection': 'raw_video_data',
                        'cameras': list(devices['cameras'].keys())}
        video_compressed = sess_params.get_video_compressed(acquisition_description)

        if video_compressed:
            # This is for widefield case where the video is already compressed
            tasks[tn] = type((tn := 'VideoConvert'), (vtasks.VideoConvert,), {})(
                **kwargs, **video_kwargs)
            dlc_parent_task = tasks['VideoConvert']
            tasks[tn] = type((tn := f'VideoSyncQC_{sync}'), (vtasks.VideoSyncQcCamlog,), {})(
                **kwargs, **video_kwargs, **sync_kwargs)
        else:
            tasks[tn] = type((tn := 'VideoRegisterRaw'), (vtasks.VideoRegisterRaw,), {})(
                **kwargs, **video_kwargs)
            tasks[tn] = type((tn := 'VideoCompress'), (vtasks.VideoCompress,), {})(
                **kwargs, **video_kwargs, **sync_kwargs)
            dlc_parent_task = tasks['VideoCompress']
            if sync == 'bpod':
                tasks[tn] = type((tn := f'VideoSyncQC_{sync}'), (vtasks.VideoSyncQcBpod,), {})(
                    **kwargs, **video_kwargs, **sync_kwargs, parents=[tasks['VideoCompress']])
            elif sync == 'nidq':
                tasks[tn] = type((tn := f'VideoSyncQC_{sync}'), (vtasks.VideoSyncQcNidq,), {})(
                    **kwargs, **video_kwargs, **sync_kwargs, parents=[tasks['VideoCompress']] + sync_tasks)

        if len(video_kwargs['cameras']) == 3:
            tasks[tn] = type((tn := 'DLC'), (epp.EphysDLC,), {})(
                **kwargs, parents=[dlc_parent_task])
            tasks['PostDLC'] = type('PostDLC', (epp.EphysPostDLC,), {})(
                **kwargs, parents=[tasks['DLC'], tasks[f'VideoSyncQC_{sync}']])

    # Audio tasks
    if 'microphone' in devices:
        (microphone, micro_kwargs), = devices['microphone'].items()
        micro_kwargs['device_collection'] = micro_kwargs.pop('collection')
        if sync_kwargs['sync'] == 'bpod':
            tasks['AudioRegisterRaw'] = type('AudioRegisterRaw', (atasks.AudioSync,), {})(
                **kwargs, **sync_kwargs, **micro_kwargs, collection=micro_kwargs['device_collection'])
        elif sync_kwargs['sync'] == 'nidq':
            tasks['AudioRegisterRaw'] = type('AudioRegisterRaw', (atasks.AudioCompress,), {})(**kwargs, **micro_kwargs)

    # Widefield tasks
    if 'widefield' in devices:
        (_, wfield_kwargs), = devices['widefield'].items()
        wfield_kwargs['device_collection'] = wfield_kwargs.pop('collection')
        tasks['WideFieldRegisterRaw'] = type('WidefieldRegisterRaw', (wtasks.WidefieldRegisterRaw,), {})(
            **kwargs, **wfield_kwargs)
        tasks['WidefieldCompress'] = type('WidefieldCompress', (wtasks.WidefieldCompress,), {})(
            **kwargs, **wfield_kwargs, parents=[tasks['WideFieldRegisterRaw']])
        tasks['WidefieldPreprocess'] = type('WidefieldPreprocess', (wtasks.WidefieldPreprocess,), {})(
            **kwargs, **wfield_kwargs, parents=[tasks['WidefieldCompress']])
        tasks['WidefieldSync'] = type('WidefieldSync', (wtasks.WidefieldSync,), {})(
            **kwargs, **wfield_kwargs, **sync_kwargs,
            parents=[tasks['WideFieldRegisterRaw'], tasks['WidefieldCompress']] + sync_tasks)
        tasks['WidefieldFOV'] = type('WidefieldFOV', (wtasks.WidefieldFOV,), {})(
            **kwargs, **wfield_kwargs, parents=[tasks['WidefieldPreprocess']])

    if 'photometry' in devices:
        # {'collection': 'raw_photometry_data', 'sync_label': 'frame_trigger', 'regions': ['Region1G', 'Region3G']}
        photometry_kwargs = devices['photometry']
        tasks['TaskFibrePhotometryRegisterRaw'] = type('TaskFibrePhotometryRegisterRaw', (
            TaskFibrePhotometryRegisterRaw,), {})(**kwargs, **photometry_kwargs)
        tasks['TaskFibrePhotometryPreprocess'] = type('TaskFibrePhotometryPreprocess', (
            TaskFibrePhotometryPreprocess,), {})(**kwargs, **photometry_kwargs, **sync_kwargs,
                                                 parents=[tasks['TaskFibrePhotometryRegisterRaw']] + sync_tasks)

    p = mtasks.Pipeline(session_path=session_path, **pkwargs)
    p.tasks = tasks
    return p


def make_pipeline_dict(pipeline, save=True):
    task_dicts = pipeline.create_tasks_list_from_pipeline()
    # TODO better name
    if save:
        with open(Path(pipeline.session_path).joinpath('pipeline_tasks.yaml'), 'w') as file:
            _ = yaml.dump(task_dicts, file)
    return task_dicts


def load_pipeline_dict(path):
    with open(Path(path).joinpath('pipeline_tasks.yaml'), 'r') as file:
        task_list = yaml.full_load(file)

    return task_list

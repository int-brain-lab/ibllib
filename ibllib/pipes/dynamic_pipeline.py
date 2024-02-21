"""Task pipeline creation from an acquisition description.

The principal function here is `make_pipeline` which reads an `_ibl_experiment.description.yaml`
file and determines the set of tasks required to preprocess the session.

In the experiment description file there is a 'tasks' key that defines each task protocol and the
location of the raw data (i.e. task collection). The protocol subkey may contain an 'extractors'
field that should contain a list of dynamic pipeline task class names for extracting the task data.
These must be subclasses of the :class:`ibllib.pipes.base_tasks.DynamicTask` class. If the
extractors key is absent or empty, the tasks are chosen based on the sync label and protocol name.

NB: The standard behvaiour extraction task classes (e.g.
:class:`ibllib.pipes.behaviour_tasks.ChoiceWorldTrialsBpod` and :class:`ibllib.pipes.behaviour_tasks.ChoiceWorldTrialsNidq`)
handle the clock synchronization, behaviour plots and QC. This is typically independent of the Bpod
trials extraction (i.e. extraction of trials data from the Bpod raw data, in Bpod time). The Bpod
trials extractor class is determined by the :func:`ibllib.io.extractors.base.protocol2extractor`
map. IBL protocols may be added to the ibllib.io.extractors.task_extractor_map.json file, while
non-IBL ones should be in projects.base.task_extractor_map.json file located in the personal
projects repo. The Bpod trials extractor class must be a subclass of the
:class:`ibllib.io.extractors.base.BaseBpodTrialsExtractor` class, and located in either the
personal projects repo or in :py:mod:`ibllib.io.extractors.bpod_trials` module.
"""
import logging
import re
from collections import OrderedDict
from pathlib import Path
from itertools import chain
import yaml

import spikeglx

import ibllib.io.session_params as sess_params
from ibllib.io.extractors.base import get_pipeline, get_session_extractor_type
import ibllib.pipes.tasks as mtasks
import ibllib.pipes.base_tasks as bstasks
import ibllib.pipes.widefield_tasks as wtasks
import ibllib.pipes.mesoscope_tasks as mscope_tasks
import ibllib.pipes.sync_tasks as stasks
import ibllib.pipes.behavior_tasks as btasks
import ibllib.pipes.video_tasks as vtasks
import ibllib.pipes.ephys_tasks as etasks
import ibllib.pipes.audio_tasks as atasks
import ibllib.pipes.photometry_tasks as ptasks
# from ibllib.pipes.photometry_tasks import FibrePhotometryPreprocess, FibrePhotometryRegisterRaw

_logger = logging.getLogger(__name__)


def acquisition_description_legacy_session(session_path, save=False):
    """
    From a legacy session create a dictionary corresponding to the acquisition description.

    Parameters
    ----------
    session_path : str, pathlib.Path
        A path to a session to describe.
    save : bool
        If true, saves the acquisition description file to _ibl_experiment.description.yaml.

    Returns
    -------
    dict
        The legacy acquisition description.
    """
    extractor_type = get_session_extractor_type(session_path)
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
                'left': {'collection': 'raw_video_data', 'sync_label': 'audio'},
            },
            'microphone': {
                'microphone': {'collection': 'raw_behavior_data', 'sync_label': None}
            },
        }
        acquisition_description = {  # this is the current ephys pipeline description
            'devices': devices,
            'sync': {'bpod': {'collection': 'raw_behavior_data'}},
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
            'sync_label': 'bpod', 'main': True  # FIXME: What is purpose of main key?
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
    pkwargs
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
    kwargs = {'session_path': session_path, 'one': pkwargs.get('one')}

    # Registers the experiment description file
    tasks['ExperimentDescriptionRegisterRaw'] = type('ExperimentDescriptionRegisterRaw',
                                                     (bstasks.ExperimentDescriptionRegisterRaw,), {})(**kwargs)

    # Syncing tasks
    (sync, sync_args), = acquisition_description['sync'].items()
    sync_args['sync_collection'] = sync_args.pop('collection')  # rename the key so it matches task run arguments
    sync_args['sync_ext'] = sync_args.pop('extension', None)
    sync_args['sync_namespace'] = sync_args.pop('acquisition_software', None)
    sync_kwargs = {'sync': sync, **sync_args}
    sync_tasks = []
    if sync == 'nidq' and sync_args['sync_collection'] == 'raw_ephys_data':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (etasks.EphysSyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
        tasks[f'SyncPulses_{sync}'] = type(f'SyncPulses_{sync}', (etasks.EphysSyncPulses,), {})(
            **kwargs, **sync_kwargs, parents=[tasks['SyncRegisterRaw']])
        sync_tasks = [tasks[f'SyncPulses_{sync}']]
    elif sync_args['sync_namespace'] == 'timeline':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (stasks.SyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
    elif sync == 'nidq':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (stasks.SyncMtscomp,), {})(**kwargs, **sync_kwargs)
        tasks[f'SyncPulses_{sync}'] = type(f'SyncPulses_{sync}', (stasks.SyncPulses,), {})(
            **kwargs, **sync_kwargs, parents=[tasks['SyncRegisterRaw']])
        sync_tasks = [tasks[f'SyncPulses_{sync}']]
    elif sync == 'tdms':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (stasks.SyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
    elif sync == 'bpod':
        pass  # ATM we don't have anything for this not sure it will be needed in the future

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
            for j, extractor in enumerate(extractors):
                # Assume previous task in the list is parent
                parents = [] if j == 0 else [tasks[task_name]]
                # Make sure extractor and sync task don't collide
                for sync_option in ('nidq', 'bpod'):
                    if sync_option in extractor.lower() and not sync == sync_option:
                        raise ValueError(f'Extractor "{extractor}" and sync "{sync}" do not match')
                # Look for the extractor in the behavior extractors module
                if hasattr(btasks, extractor):
                    task = getattr(btasks, extractor)
                # This may happen that the extractor is tied to a specific sync task: look for TrialsChoiceWorldBpod for # example
                elif hasattr(btasks, extractor + sync.capitalize()):
                    task = getattr(btasks, extractor + sync.capitalize())
                else:
                    # lookup in the project extraction repo if we find an extractor class
                    import projects.extraction_tasks
                    if hasattr(projects.extraction_tasks, extractor):
                        task = getattr(projects.extraction_tasks, extractor)
                    else:
                        raise NotImplementedError(
                            f'Extractor "{extractor}" not found in main IBL pipeline nor in personal projects')
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
            if 'passiveChoiceWorld' in protocol:
                registration_class = btasks.PassiveRegisterRaw
                behaviour_class = btasks.PassiveTask
                compute_status = False
            elif sync_kwargs['sync'] == 'bpod':
                if 'habituation' in protocol:
                    registration_class = btasks.HabituationRegisterRaw
                    behaviour_class = btasks.HabituationTrialsBpod
                    compute_status = False
                else:
                    registration_class = btasks.TrialRegisterRaw
                    behaviour_class = btasks.ChoiceWorldTrialsBpod
                    compute_status = True
            elif sync_kwargs['sync'] == 'nidq':
                if 'habituation' in protocol:
                    registration_class = btasks.HabituationRegisterRaw
                    behaviour_class = btasks.HabituationTrialsNidq
                    compute_status = False
                else:
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
        cams = list(devices['cameras'].keys())
        subset_cams = [c for c in cams if c in ('left', 'right', 'body', 'belly')]
        video_kwargs = {'device_collection': 'raw_video_data', 'cameras': cams}
        video_compressed = sess_params.get_video_compressed(acquisition_description)

        if video_compressed:
            # This is for widefield case where the video is already compressed
            tasks[tn] = type((tn := 'VideoConvert'), (vtasks.VideoConvert,), {})(**kwargs, **video_kwargs)
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
                # Here we restrict to videos that we support (left, right or body)
                video_kwargs['cameras'] = subset_cams
                tasks[tn] = type((tn := f'VideoSyncQC_{sync}'), (vtasks.VideoSyncQcNidq,), {})(
                    **kwargs, **video_kwargs, **sync_kwargs, parents=[tasks['VideoCompress']] + sync_tasks)

        if sync_kwargs['sync'] != 'bpod':
            # Here we restrict to videos that we support (left, right or body)
            # Currently there is no plan to run DLC on the belly cam
            subset_cams = [c for c in cams if c in ('left', 'right', 'body')]
            video_kwargs['cameras'] = subset_cams
            tasks[tn] = type((tn := 'DLC'), (vtasks.DLC,), {})(
                **kwargs, **video_kwargs, parents=[dlc_parent_task])

            # The PostDLC plots require a trials object for QC
            # Find the first task that outputs a trials.table dataset
            trials_task = (
                t for t in tasks.values() if any('trials.table' in f for f in t.signature.get('output_files', []))
            )
            if trials_task := next(trials_task, None):
                parents = [tasks['DLC'], tasks[f'VideoSyncQC_{sync}'], trials_task]
                trials_collection = getattr(trials_task, 'output_collection', 'alf')
            else:
                parents = [tasks['DLC'], tasks[f'VideoSyncQC_{sync}']]
                trials_collection = 'alf'
            tasks[tn] = type((tn := 'PostDLC'), (vtasks.EphysPostDLC,), {})(
                **kwargs, cameras=subset_cams, trials_collection=trials_collection, parents=parents)

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

    # Mesoscope tasks
    if 'mesoscope' in devices:
        (_, mscope_kwargs), = devices['mesoscope'].items()
        mscope_kwargs['device_collection'] = mscope_kwargs.pop('collection')
        tasks['MesoscopeRegisterSnapshots'] = type('MesoscopeRegisterSnapshots', (mscope_tasks.MesoscopeRegisterSnapshots,), {})(
            **kwargs, **mscope_kwargs)
        tasks['MesoscopePreprocess'] = type('MesoscopePreprocess', (mscope_tasks.MesoscopePreprocess,), {})(
            **kwargs, **mscope_kwargs)
        tasks['MesoscopeFOV'] = type('MesoscopeFOV', (mscope_tasks.MesoscopeFOV,), {})(
            **kwargs, **mscope_kwargs, parents=[tasks['MesoscopePreprocess']])
        tasks['MesoscopeSync'] = type('MesoscopeSync', (mscope_tasks.MesoscopeSync,), {})(
            **kwargs, **mscope_kwargs, **sync_kwargs)
        tasks['MesoscopeCompress'] = type('MesoscopeCompress', (mscope_tasks.MesoscopeCompress,), {})(
            **kwargs, **mscope_kwargs, parents=[tasks['MesoscopePreprocess']])

    if 'photometry' in devices:
        # {'collection': 'raw_photometry_data', 'sync_label': 'frame_trigger', 'regions': ['Region1G', 'Region3G']}
        photometry_kwargs = devices['photometry']
        tasks['FibrePhotometryRegisterRaw'] = type('FibrePhotometryRegisterRaw', (
            ptasks.FibrePhotometryRegisterRaw,), {})(**kwargs, **photometry_kwargs)
        tasks['FibrePhotometryPreprocess'] = type('FibrePhotometryPreprocess', (
            ptasks.FibrePhotometryPreprocess,), {})(**kwargs, **photometry_kwargs, **sync_kwargs,
                                                    parents=[tasks['FibrePhotometryRegisterRaw']] + sync_tasks)

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


def get_trials_tasks(session_path, one=None):
    """
    Return a list of pipeline trials extractor task objects for a given session.

    This function supports both legacy and dynamic pipeline sessions.

    Parameters
    ----------
    session_path : str, pathlib.Path
        An absolute path to a session.
    one : one.api.One
        An ONE instance.

    Returns
    -------
    list of pipes.tasks.Task
        A list of task objects for the provided session.

    """
    # Check for an experiment.description file; ensure downloaded if possible
    if one and one.to_eid(session_path):  # to_eid returns None if session not registered
        one.load_datasets(session_path, ['_ibl_experiment.description'], download_only=True, assert_present=False)
    experiment_description = sess_params.read_params(session_path)

    # If experiment description file then use this to make the pipeline
    if experiment_description is not None:
        tasks = []
        pipeline = make_pipeline(session_path, one=one)
        trials_tasks = [t for t in pipeline.tasks if 'Trials' in t]
        for task in trials_tasks:
            t = pipeline.tasks.get(task)
            t.__init__(session_path, **t.kwargs)
            tasks.append(t)
    else:
        # Otherwise default to old way of doing things
        if one and one.to_eid(session_path):
            one.load_dataset(session_path, '_iblrig_taskSettings.raw', collection='raw_behavior_data', download_only=True)
        pipeline = get_pipeline(session_path)
        if pipeline == 'training':
            from ibllib.pipes.training_preprocessing import TrainingTrials
            tasks = [TrainingTrials(session_path, one=one)]
        elif pipeline == 'ephys':
            from ibllib.pipes.ephys_preprocessing import EphysTrials
            tasks = [EphysTrials(session_path, one=one)]
        else:
            try:
                # try to find a custom extractor in the personal projects extraction class
                import projects.base
                task_type = get_session_extractor_type(session_path)
                assert (PipelineClass := projects.base.get_pipeline(task_type))
                pipeline = PipelineClass(session_path, one=one)
                trials_task_name = next((task for task in pipeline.tasks if 'Trials' in task), None)
                assert trials_task_name, (f'No "Trials" tasks for custom pipeline '
                                          f'"{pipeline.name}" with extractor type "{task_type}"')
                task = pipeline.tasks.get(trials_task_name)
                task(session_path)
                tasks = [task]
            except (ModuleNotFoundError, AssertionError) as ex:
                _logger.warning('Failed to get trials tasks: %s', ex)
                tasks = []

    return tasks

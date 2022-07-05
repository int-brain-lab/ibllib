from collections import OrderedDict
from pathlib import Path
import yaml
import ibllib.io.session_params as sess_params

import ibllib.pipes.ephys_preprocessing as epp
import ibllib.pipes.tasks as mtasks
import ibllib.pipes.widefield_tasks as wtasks
import ibllib.pipes.sync_tasks as stasks
import ibllib.pipes.behavior_tasks as btasks
import ibllib.pipes.video_tasks as vtasks
import ibllib.pipes.ephys_tasks as etasks
import ibllib.pipes.audio_tasks as atasks

import spikeglx


def acquisition_description_legacy_session():
    """
    From a legacy session create a dictionary corresponding to the acquisition description
    :return: dict
    """


def get_acquisition_description(protocol):
    """"
    This is a set of example acqusition descriptions for experiments
    -   choice_world_recording
    -   choice_world_biased
    -   choice_world_training
    -   choice_world_habituation
    -   choice_world_passive
    That are part of the IBL pipeline
    """
    if protocol == 'choice_world_recording':   # canonical ephys
        acquisition_description = {  # this is the current ephys pipeline description
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
            'tasks': {
                'ephysChoiceWorld': {'collection': 'raw_behavior_data', 'sync_label': 'bpod'},
                'passiveChoiceWorld': {'collection': 'raw_passive_data', 'sync_label': 'bpod'},
            },
            'sync': {
                'nidq': {'collection': 'raw_ephys_data', 'extension': 'bin', 'acquisition_software': 'spikeglx'}
            },
            'procedures': ['Ephys recording with acute probe(s)'],
            'projects': ['ibl_neuropixel_brainwide_01']
        }
    else:
        # TODO make ordered dict
        acquisition_description = {  # this is the current ephys pipeline description
            'cameras': {
                'left': {'collection': 'raw_video_data', 'sync_label': 'frame2ttl'},
            },
            'microphone': {
                'microphone': {'collection': 'raw_behavior_data', 'sync_label': None}
            },
            'tasks': {
                'trainingChoiceWorld': {'collection': 'raw_behavior_data', 'sync_label': 'bpod', 'main': True}
            },
            'sync': {
                'bpod': {'collection': 'raw_behavior_data', 'extension': 'bin'}
            },
            'procedures': ['Behavior training/tasks'],
            'projects': ['ibl_neuropixel_brainwide_01']
        }
    return acquisition_description


class DummyTask(mtasks.Task):
    def _run(self):
        pass


def make_pipeline(session_path=None, **pkwargs):
    """
    :param session_path:
    :param one: passed to the Pipeline init: one instance to register tasks to
    :param eid: passed to the Pipeline init
    :return:
    """
    # NB: this pattern is a pattern for dynamic class creation
    # tasks['SyncPulses'] = type('SyncPulses', (epp.EphysPulses,), {})(session_path=session_path)
    assert session_path
    tasks = OrderedDict()
    acquisition_description = sess_params.read_params(session_path)

    kwargs = {'session_path': session_path}

    # Syncing tasks
    (sync, sync_args), = acquisition_description['sync'].items()
    sync_args['sync_collection'] = sync_args.pop('collection')  # rename the key so it matches task run arguments
    sync_args['sync_ext'] = sync_args.pop('extension')
    sync_args['sync_namespace'] = sync_args.pop('acquisition_software', None)
    sync_kwargs = {'sync': sync, **sync_args}
    sync_tasks = []
    if sync == 'nidq' and sync_args['sync_collection'] == 'raw_ephys_data':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (etasks.EphysSyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
        tasks[f'SyncPulses_{sync}'] = type(f'SyncPulses_{sync}', (etasks.EphysSyncPulses,), {})\
            (**kwargs, **sync_kwargs, parents=[tasks['SyncRegisterRaw']])
        sync_tasks = [tasks[f'SyncPulses_{sync}']]
    elif sync == 'nidq':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (stasks.SyncMtscomp,), {})(**kwargs, **sync_kwargs)
        tasks[f'SyncPulses_{sync}'] = type(f'SyncPulses_{sync}', (stasks.SyncPulses,), {})\
            (**kwargs, **sync_kwargs, parents=[tasks['SyncRegisterRaw']])
        sync_tasks = [tasks[f'SyncPulses_{sync}']]
    elif sync == 'tdms':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (stasks.SyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
    elif sync == 'bpod':
        pass
        # ATM we don't have anything for this not sure it will be needed in the future

    # Behavior tasks
    # TODO this is not doing at all what we were envisaging and going back to the old way of protocol linked to hardware
    # TODO change at next iteration of dynamic pipeline, once we have the basic workflow working
    for protocol, task_info in acquisition_description.get('tasks', []).items():
        task_kwargs = {'protocol': protocol, 'collection': task_info['collection']}
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
        tasks[f'RegisterRaw_{protocol}'] = type(f'RegisterRaw_{protocol}', (registration_class,), {})(**kwargs, **task_kwargs)
        parents = [tasks[f'RegisterRaw_{protocol}']] + sync_tasks
        tasks[f'Trials_{protocol}'] = type(f'Trials_{protocol}', (behaviour_class,), {})\
            (**kwargs, **sync_kwargs, **task_kwargs, parents=parents)
        if compute_status:
            tasks[f"TrainingStatus_{protocol}"] = type(f"TrainingStatus_{protocol}", (btasks.TrainingStatus,), {})\
                (**kwargs, **task_kwargs, parents=[tasks[f'Trials_{protocol}']])

    # Ephys tasks
    if 'neuropixel' in acquisition_description:
        ephys_kwargs = {'device_collection': 'raw_ephys_data'}
        tasks['EphysRegisterRaw'] = type('EphysRegisterRaw', (etasks.EphysRegisterRaw,), {})(**kwargs, **ephys_kwargs)

        all_probes = []
        register_tasks = []
        for pname, probe_info in acquisition_description['neuropixel'].items():
            meta_file = spikeglx.glob_ephys_files(Path(session_path).joinpath(probe_info['collection']), ext='meta')
            meta_file = meta_file[0].get('ap')
            nptype = spikeglx._get_neuropixel_version_from_meta(spikeglx.read_meta_data(meta_file))
            nshanks = spikeglx._get_nshanks_from_meta(spikeglx.read_meta_data(meta_file))

            if nptype == 'NP2.1':
                tasks[f'EphyCompressNP21_{pname}'] = type(f'EphyCompressNP21_{pname}', (etasks.EphysCompressNP21,), {})\
                    (**kwargs, **ephys_kwargs, pname=pname)
                all_probes.append(pname)
                register_tasks.append(tasks[f'EphyCompressNP21_{pname}'])
            elif nptype == 'NP2.4':
                tasks[f'EphyCompressNP24_{pname}'] = type(f'EphyCompressNP24_{pname}', (etasks.EphysCompressNP24,), {})(
                    **kwargs, **ephys_kwargs, pname=pname, nshanks=nshanks)
                register_tasks.append(tasks[f'EphyCompressNP24_{pname}'])
                all_probes += [f'{pname}{chr(97 + int(shank))}' for shank in range(nshanks)]
            else:
                tasks[f'EphysCompressNP1_{pname}'] = type(f'EphyCompressNP1_{pname}', (etasks.EphysCompressNP1,), {})\
                    (**kwargs, **ephys_kwargs, pname=pname)
                register_tasks.append(tasks[f'EphysCompressNP1_{pname}'])
                all_probes.append(pname)

        if nptype == '3A':
            tasks['EphysPulses'] = type('EphysPulses', (etasks.EphysPulses,), {})\
                (**kwargs, **ephys_kwargs, **sync_kwargs, pname=all_probes, parents=register_tasks + sync_tasks)

        for pname in all_probes:
            register_task = [reg_task for reg_task in register_tasks if pname[:7] in reg_task.name]

            if nptype != '3A':
                tasks[f'EphysPulses_{pname}'] = type(f'EphysPulses_{pname}', (etasks.EphysPulses,), {})\
                    (**kwargs, **ephys_kwargs, **sync_kwargs, pname=[pname], parents=register_task + sync_tasks)
                tasks[f'Spikesorting_{pname}'] = type(f'Spikesorting_{pname}', (etasks.SpikeSorting,), {}) \
                    (**kwargs, **ephys_kwargs, pname=pname, parents=[tasks[f'EphysPulses_{pname}']])
            else:
                tasks[f'Spikesorting_{pname}'] = type(f'Spikesorting_{pname}', (etasks.SpikeSorting,), {}) \
                    (**kwargs, **ephys_kwargs, pname=pname, parents=[tasks['EphysPulses']])

            tasks[f'RawEphysQC_{pname}'] = type(f'RawEphysQC_{pname}', (etasks.RawEphysQC,), {})\
                (**kwargs, **ephys_kwargs, pname=pname, parents=register_task)
            tasks[f'EphysCellQC_{pname}'] = type(f'EphysCellQC_{pname}', (etasks.EphysCellsQc,), {})\
                (**kwargs, **ephys_kwargs, pname=pname, parents=[tasks[f'Spikesorting_{pname}']])

    # Video tasks
    if 'cameras' in acquisition_description:
        video_kwargs = {'device_collection': 'raw_video_data',
                        'cameras': list(acquisition_description['cameras'].keys())}
        tasks[tn] = type((tn := 'VideoRegisterRaw'), (vtasks.VideoRegisterRaw,), {})(**kwargs, **video_kwargs)
        tasks[tn] = type((tn := 'VideoCompress'), (vtasks.VideoCompress,), {})(**kwargs, **video_kwargs)

        if sync == 'bpod':
            collection = sess_params.get_task_collection(acquisition_description)
            tasks[tn] = type((tn := 'VideoSyncQCBpod'), (vtasks.VideoSyncQcBpod,), {})(
                **kwargs, **video_kwargs,**sync_kwargs, collection=collection, parents=[tasks['VideoCompress']])
        elif sync == 'nidq':
            tasks[tn] = type((tn := 'VideoSyncQCNidq'), (vtasks.VideoSyncQcNidq,), {})(
                **kwargs, **video_kwargs, **sync_kwargs, parents=[tasks['VideoCompress']] + sync_tasks)

        if len(video_kwargs['cameras']) == 3:
            tasks[tn] = type((tn := 'DLC'), (epp.EphysDLC,), {})(
                **kwargs, parents=[tasks['VideoCompress']])
            tasks['PostDLC'] = type('PostDLC', (epp.EphysPostDLC,), {})(
                **kwargs, parents=[tasks['DLC'], tasks['VideoSyncQC']])

    # Audio tasks
    if 'microphone' in acquisition_description:
        (microphone, micro_kwargs), = acquisition_description['microphone'].items()
        micro_kwargs['device_collection'] = micro_kwargs.pop('collection')
        if sync_kwargs['sync'] == 'bpod':
            tasks['AudioRegisterRaw'] = type('AudioRegisterRaw', (atasks.AudioSync,), {})\
                (**kwargs, **sync_kwargs, **micro_kwargs, collection=collection)
        elif sync_kwargs['sync'] == 'nidq':
            tasks['AudioRegisterRaw'] = type('AudioRegisterRaw', (atasks.AudioCompress,), {})(**kwargs, **micro_kwargs)

    # Widefield tasks
    if 'widefield' in acquisition_description:
        (_, wfield_kwargs), = acquisition_description['widefield'].items()
        wfield_kwargs['device_collection'] = wfield_kwargs.pop('collection')

        tasks['WideFieldRegisterRaw'] = type('WidefieldRegisterRaw', (wtasks.WidefieldRegisterRaw,), {})\
            (**kwargs, **wfield_kwargs)
        tasks['WidefieldCompress'] = type('WidefieldCompress', (wtasks.WidefieldCompress,), {})\
            (**kwargs, **wfield_kwargs)
        tasks['WidefieldPreprocess'] = type('WidefieldPreprocess', (wtasks.WidefieldPreprocess,), {})\
            (**kwargs, **wfield_kwargs, parents=[tasks['WideFieldRegisterRaw'], tasks['WidefieldCompress']])
        tasks['WidefieldSync'] = type('WidefieldSync', (wtasks.WidefieldSync,), {})\
            (**kwargs, **wfield_kwargs, **sync_kwargs, parents=[tasks['WideFieldRegisterRaw'],
                                                                tasks['WidefieldCompress']] + sync_tasks)
        tasks['WidefieldFOV'] = type('WidefieldFOV', (wtasks.WidefieldFOV,), {})\
            (**kwargs, **wfield_kwargs, parents=[tasks['WidefieldPreprocess']])

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

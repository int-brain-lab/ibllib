from collections import OrderedDict
from pathlib import Path
import ibllib.io.session_params as sess_params

import ibllib.pipes.ephys_preprocessing as epp
import ibllib.pipes.training_preprocessing as tpp
import ibllib.pipes.tasks as mtasks
import ibllib.pipes.widefield_tasks as wtasks
import ibllib.pipes.sync_tasks as stasks
import ibllib.pipes.behavior_tasks as btasks
import ibllib.pipes.video_tasks as vtasks
import ibllib.pipes.ephys_tasks as etasks

import spikeglx
## collection - collection with task data
## protocol - protocol of task
## sync_collection - collection with main sync data
## device collection - collection with raw device files
## sync - type of sync
## sync_ext - extension of sync
## pname - probe name
## nshanks - number of shanks on probe



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
                'right': {'collection': 'raw_video_data', 'sync_label': 'frame2ttl'},
                'body': {'collection': 'raw_video_data', 'sync_label': 'frame2ttl'},
                'left': {'collection': 'raw_video_data', 'sync_label': 'frame2ttl'},
            },
            'neuropixel': {
                'probe00': {'collection': 'raw_ephys_data/probe00', 'sync_label': 'imec_sync'},
                'probe01': {'collection': 'raw_ephys_data/probe01', 'sync_label': 'imec_sync'}
            },
            'microphone': {
                'harp': {'collection': 'raw_behavior_data', 'sync_label': None}
            },
            'tasks': {
                'choice_world_training': {'collection': 'raw_behavior_data', 'sync_label': 'bpod', 'main': True},
                'choice_world_passive': {'collection': 'raw_passive_data', 'sync_label': 'bpod', 'main': False},
            },
            'sync': {
                'bpod': {'collection': 'raw_behavior_data', 'extension': '.bin'}
            },
            'procedures': ['Ephys recording with acute probe(s)'],
            'projects': ['ibl_neuropixel_brainwide_01']
        }
    else:
        acquisition_description = {  # this is the current ephys pipeline description
            'cameras': {
                'left': {'collection': 'raw_video_data', 'sync_label': 'frame2ttl'},
            },
            'microphone': {
                'xonar': {'collection': 'raw_behavior_data', 'sync_label': None}
            },
            'tasks': {
                protocol: {'collection': 'raw_behavior_data', 'sync_label': 'bpod', 'main': True},
            },
            'sync': {
                'bpod': {'collection': 'raw_behavior_data', 'extension': '.bin'}
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
    sync_kwargs = {'sync': sync, **sync_args}
    sync_tasks = []
    if sync == 'nidq' and sync_args['sync_collection'] == 'raw_ephys_data':
        tasks['SyncRegisterRaw'] = type('SyncRegisterRaw', (etasks.EphysSyncRegisterRaw,), {})(**kwargs, **sync_kwargs)
        tasks[f'SyncPulses_{sync}'] = type(f'SyncPulses_{sync}', (etasks.EphysSyncPulses,), {})(**kwargs, **sync_kwargs)
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
    for protocol, task_info in acquisition_description.get('tasks', []).items():
        task_kwargs = {'protocol': protocol, 'collection': task_info['collection']}
        # -   choice_world_recording
        # -   choice_world_biased
        # -   choice_world_training
        # -   choice_world_habituation
        if protocol == 'choice_world_habituation':
            registration_class = btasks.HabituationRegisterRaw
            behaviour_class = btasks.HabituationTrialsBpod
            compute_status = False
        elif protocol == 'choice_world_passive':
            registration_class = btasks.PassiveRegisterRaw
            behaviour_class = btasks.PassiveRegisterRaw
            compute_status = False
        elif protocol in ['choice_world_training', 'choice_world_biased']:
            registration_class = btasks.TrialRegisterRaw
            behaviour_class = btasks.ChoiceWorldTrialsBpod
            compute_status = False
        else:
            raise NotImplementedError
        tasks[f'RegisterRaw_{protocol}'] = type(f'RegisterRaw_{protocol}', (registration_class,), {})(**kwargs, **task_kwargs)
        parents = [tasks[f'RegisterRaw_{protocol}']] + sync_tasks
        tasks[f'Trials_{protocol}'] = type(f'Trials_{protocol}', (behaviour_class,), {})\
            (**kwargs, **sync_kwargs, **task_kwargs, parents=parents)
        if compute_status:
            # TODO move this
            tasks["Training Status"] = type("Training Status", (tpp.TrainingStatus,), {})\
                (**kwargs, parents=[tasks[f'Trials_{protocol}']])

    # Ephys tasks
    if 'neuropixel' in acquisition_description:
        ephys_kwargs = {'device_collection': 'raw_ephys_data'}
        tasks['EphysRegisterRaw'] = type(f'EphysRegisterRaw', (etasks.EphysRegisterRaw,), {})(**kwargs, **ephys_kwargs)

        all_probes = []
        register_tasks = []
        for pname, probe_info in acquisition_description['neuropixel'].items():
            meta_file = spikeglx.glob_ephys_files(Path(session_path).joinpath(probe_info['collection']), ext='.meta')
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
                all_probes.append([f'{pname}{chr(97 + int(shank))}' for shank in range(nshanks)])
            else:
                tasks[f'EphysCompressNP1_{pname}'] = type(f'EphyCompressNP1_{pname}', (etasks.EphysCompressNP1,), {})\
                    (**kwargs, **ephys_kwargs, pname=pname)
                register_tasks.append(tasks[f'EphyCompressNP21_{pname}'])
                all_probes.append(pname)

        tasks['EphysPulses'] = type(f'EphysPulses', (etasks.EphysPulses,), {})\
            (**kwargs, **ephys_kwargs, **sync_kwargs, pname=all_probes, parents=register_tasks + sync_tasks)

        for pname in all_probes:
            tasks[f'RawEphysQC_{pname}'] = type(f'RawEphysQC_{pname}', (etasks.RawEphysQC,), {})\
                (**kwargs, **ephys_kwargs, pname=pname, parents=register_tasks)
            tasks[f'Spikesorting_{pname}'] = type(f'RawEphysQC_{pname}', (etasks.SpikeSorting,), {})\
                (**kwargs, **ephys_kwargs, pname=pname, parents=register_tasks + [tasks['EphysPulses']])
            tasks[f'EphysCellQC_{pname}'] = type(f'EphysCellQC_{pname}', (etasks.EphysCellsQc,), {})\
                (**kwargs, **ephys_kwargs, pname=pname, parents=[tasks[f'Spikesorting_{pname}']])


    # Video tasks
    if 'cameras' in acquisition_description:
        video_kwargs = {'device_collection': 'raw_video_data',
                        'cameras': list(acquisition_description['cameras'].keys()),
                        'main_task_collection': sess_params.get_main_task_collection(acquisition_description)}

        tasks[tn] = type((tn := 'VideoRegisterRaw'), (vtasks.VideoRegisterRaw,), {})(**kwargs, **video_kwargs)
        tasks[tn] = type((tn := f'VideoCompress'), (vtasks.VideoCompress,), {})(
            **kwargs, **video_kwargs, parents=[tasks['VideoRegisterRaw']])
        tasks[tn] = type((tn := f'VideoSyncQC'), (vtasks.VideoSyncQc,), {})(
            **kwargs, **video_kwargs, parents=[tasks[f'VideoCompress']] + sync_tasks)
        tasks[tn] = type((tn := 'DLC'), (epp.EphysDLC,), {})(
            **kwargs, parents=[tasks['VideoCompress']])
        tasks['PostDLC'] = type('PostDLC', (epp.EphysPostDLC,), {})(
            **kwargs, parents=[tasks['DLC'], tasks['VideoSyncQC']])

    # Audio tasks
    if 'microphone' in acquisition_description:
        (microphone, micro_kwargs), = acquisition_description['microphone'].items()
        micro_kwargs['device_collection'] = micro_kwargs.pop('collection')
        micro_kwargs['main_task_collection'] = sess_params.get_main_task_collection(acquisition_description)
        if microphone == 'xonar':
            tasks['Audio'] = type('Audio', (tpp.TrainingAudio,), {})(**kwargs, **micro_kwargs)
        elif microphone == 'harp':
            tasks['Audio'] = type('Audio', (epp.EphysAudio,), {})(**kwargs, **micro_kwargs)

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





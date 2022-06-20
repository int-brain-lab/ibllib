from collections import OrderedDict
import ibllib.pipes.tasks as mtasks
import ibllib.pipes.ephys_preprocessing as epp
import ibllib.pipes.training_preprocessing as tpp
import ibllib.pipes.widefield_tasks as wtasks
import ibllib.pipes.sync_tasks as stasks
import ibllib.pipes.behavior_tasks as btasks
import ibllib.io.session_params


def acquisition_description_legacy_session():
    """
    From a legacy session create a dictionary corresponding to the acquisition description
    :return: dict
    """


def get_acquisition_description(experiment):
    """"
    This is a set of example acqusition descriptions for experiments
    -   choice_world_recording
    -   choice_world_biased
    -   choice_world_training
    -   choice_world_habituation
    That are part of the IBL pipeline
    """
    if experiment == 'choice_world_recording':   # canonical ephys
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
                'passive_world_training': {'collection': 'raw_passive_data', 'sync_label': 'bpod', 'main': False},
            },
            'sync': {
                'bpod': {'collection': 'raw_behavior_data', 'extension': '.bin'}
            }
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
                experiment: {'collection': 'raw_behavior_data', 'sync_label': 'bpod', 'main': True},
            },
            'sync': {
                'bpod': {'collection': 'raw_behavior_data', 'extension': '.bin'}
            }
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
    acquisition_description = ibllib.io.session_params.read_params(session_path)
    # Syncing tasks
    (sync, sync_collection), = acquisition_description['sync'].items()
    kwargs = {'session_path': session_path, 'runtime_args': dict(sync_collection=sync_collection)}
    if sync == 'nidq' and sync_collection == 'raw_ephys_data':
        tasks[f'SyncPulses{sync}'] = type(f'SyncPulses{sync}', (epp.EphysPulses,), {})(**kwargs)
    elif sync == 'nidq':
        tasks['SyncCompress'] = type(f'SyncCompress{sync}', (stasks.SyncPulses,), {})(**kwargs)  # this renames the files so needs to be level0
        tasks[f'SyncPulses{sync}'] = type(f'SyncPulses{sync}', (stasks.SyncPulses,), {})\
            (**kwargs, parents=[tasks['SyncCompress']])
    elif sync == 'bpod':
        # at the moment we don't have anything for this
        tasks[f'SyncPulses{sync}'] = type(f'SyncPulses{sync}', (epp.EphysPulses,), {})(**kwargs)

    # Behavior tasks
    for protocol, protocol_collection in acquisition_description.get('tasks', []).items():
        kwargs = {'session_path': session_path, 'runtime_args': dict(protocol=protocol, protocol_collection=protocol_collection)}
        if 'passive' in protocol:
            tasks[f'RegisterRaw_{protocol}'] = type(f'RegisterRaw_{protocol}', (btasks.PassiveRegisterRaw,), {})(**kwargs)
            tasks[f'Trials_{protocol}'] = type(f'Trials_{protocol}', (btasks.PassiveRegisterRaw,), {})\
                (**kwargs, parents=[tasks[f'SyncPulses{sync}']])
        else:
            tasks[f'RegisterRaw_{protocol}'] = type(f'RegisterRaw_{protocol}', (btasks.TrialRegisterRaw,), {})(**kwargs)
            if sync == 'bpod':
                tasks[f'Trials_{protocol}'] = type(f'Trials_{protocol}', (btasks.TrainingTrialsBpod,), {})(**kwargs)
            elif sync == 'nidq':
                tasks[f'Trials_{protocol}'] = type(f'Trials_{protocol}', (btasks.TrainingTrialsFPGA,), {})\
                    (**kwargs, parents=[tasks[f'SyncPulses{sync}']])

            tasks["Training Status"] = type("Training Status", (tpp.TrainingStatus,), {})\
                (session_path=session_path, parents=[tasks[f'Trials_{protocol}']])

    # Ephys tasks
    if 'neuropixel' in acquisition_description:
        for pname, rpath in acquisition_description['neuropixel'].items():
            kwargs = {'session_path': session_path, 'runtime_args': dict(pname=pname)}
            tasks[f'Compression_{pname}'] = type(
                f'Compression_{pname}', (epp.EphysMtscomp,), {})(**kwargs)

            tasks[f'SpikeSorting_{pname}'] = type(
                f'SpikeSorting_{pname}', (epp.SpikeSorting,), {})(
                **kwargs, parents=[tasks[f'Compression_{pname}'], tasks[f'SyncPulses{sync}']])

            tasks[f'CellsQC_{pname}'] = type(
                f'CellsQC_{pname}', (epp.EphysCellsQc,), {})(
                **kwargs, parents=[tasks[f'SpikeSorting_{pname}']])

    # Video tasks
    if 'cameras' in acquisition_description:
        tasks['VideoCompress'] = type('VideoCompress', (epp.EphysVideoCompress,), {})(session_path=session_path)
        tasks['VideoSync'] = type('VideoSync', (epp.EphysVideoSyncQc,), {})(
            session_path=session_path, parents=[tasks['VideoCompress'], tasks[f'SyncPulses{sync}']])
        tasks['DLC'] = type('DLC', (epp.EphysDLC,), {})(session_path=session_path, parents=[tasks['VideoCompress']])
        tasks['PostDLC'] = type('PostDLC', (epp.EphysPostDLC,), {})(
            session_path=session_path, parents=[tasks['DLC'], tasks['VideoSync']])

    # Audio tasks
    if 'microphone' in acquisition_description:
        microphone_device = list(acquisition_description['microphone'].keys())[0]
        if microphone_device == 'xonar':
            tasks['Audio'] = type('Audio', (tpp.TrainingAudio,), {})(session_path=session_path)
        elif microphone_device == 'harp':
            tasks['Audio'] = type('Audio', (epp.EphysAudio,), {})(session_path=session_path)

    # Widefield tasks
    if 'widefield' in acquisition_description:
        # TODO all dependencies
        tasks['WideFieldRegisterRaw'] = type('WidefieldRegisterRaw', (wtasks.WidefieldRegisterRaw,), {})(session_path=session_path)
        tasks['WidefieldCompress'] = type('WidefieldCompress', (wtasks.WidefieldCompress,), {})(session_path=session_path)
        tasks['WidefieldPreprocess'] = type('WidefieldPreprocess', (wtasks.WidefieldPreprocess,), {})(session_path=session_path)
        tasks['WidefieldSync'] = type('WidefieldSync', (wtasks.WidefieldSync,), {})(session_path=session_path,
                                                                                    parents=[tasks[f'SyncPulses{sync}']])
        tasks['WidefieldFOV'] = type('WidefieldFOV', (wtasks.WidefieldFOV,), {})(session_path=session_path)

    p = mtasks.Pipeline(session_path=session_path, **pkwargs)
    p.tasks = tasks

    return p

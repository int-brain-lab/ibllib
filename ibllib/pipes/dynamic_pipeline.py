from collections import OrderedDict
import ibllib.pipes.tasks as mtasks
import ibllib.pipes.ephys_preprocessing as epp
import ibllib.pipes.training_preprocessing as tpp
import ibllib.pipes.widefield_tasks as wtasks
import ibllib.pipes.sync_tasks as stasks
import ibllib.pipes.behavior_tasks as btasks
import ibllib.pipes.video_tasks as vtasks
import ibllib.io.session_params


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
    acquisition_description = ibllib.io.session_params.read_params(session_path)
    # Syncing tasks
    (sync, sync_args), = acquisition_description['sync'].items()
    sync_args['task_collection'] = sync_args.pop('collection')  # rename the key so it matches task run arguments
    kwargs = {'session_path': session_path, **sync_args}
    sync_tasks = []
    if sync == 'nidq' and sync_args['task_collection'] == 'raw_ephys_data':
        tasks[f'SyncPulses{sync}'] = type(f'SyncPulses{sync}', (epp.EphysPulses,), {})(**kwargs)
    elif sync == 'nidq':
        # this renames the files so needs to be level 0
        tasks['SyncCompress'] = type(f'SyncCompress{sync}', (stasks.SyncPulses,), {})(**kwargs)
        tasks[f'SyncPulses{sync}'] = type(f'SyncPulses{sync}', (stasks.SyncPulses,), {})\
            (**kwargs, parents=[tasks['SyncCompress']])
        sync_tasks = [tasks[f'SyncPulses{sync}']]
    elif sync == 'bpod':
        pass
        # ATM we don't have anything for this not sure it will be needed in the future

    # Behavior tasks
    for protocol, task_info in acquisition_description.get('tasks', []).items():
        kwargs = {'session_path': session_path, 'protocol': protocol, 'collection': task_info['collection']}
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
        else:
            raise NotImplementedError
        tasks[f'RegisterRaw_{protocol}'] = type(f'RegisterRaw_{protocol}', (registration_class,), {})(**kwargs)
        parents = [tasks[f'RegisterRaw_{protocol}']] + sync_tasks
        tasks[f'Trials_{protocol}'] = type(
            f'Trials_{protocol}', (behaviour_class,), {})(**kwargs, parents=parents)
        if compute_status:
            tasks["Training Status"] = type("Training Status", (tpp.TrainingStatus,), {})\
                (session_path=session_path, parents=[tasks[f'Trials_{protocol}']])

    # Ephys tasks
    if 'neuropixel' in acquisition_description:
        for pname, rpath in acquisition_description['neuropixel'].items():
            kwargs = {'session_path': session_path, 'pname': pname}
            tasks[f'Compression_{pname}'] = type(
                f'Compression_{pname}', (epp.EphysMtscomp,), {})(**kwargs)

            tasks[f'SpikeSorting_{pname}'] = type(
                f'SpikeSorting_{pname}', (epp.SpikeSorting,), {})(
                **kwargs, parents=[tasks[f'Compression_{pname}']] + sync_tasks)

            tasks[f'CellsQC_{pname}'] = type(
                f'CellsQC_{pname}', (epp.EphysCellsQc,), {})(
                **kwargs, parents=[tasks[f'SpikeSorting_{pname}']])

    # Video tasks
    if 'cameras' in acquisition_description:
        kwargs = dict(session_path=session_path, cameras=list(acquisition_description['cameras'].keys()))
        tasks[tn] = type((tn := 'VideoRegisterRaw'), (vtasks.VideoRegisterRaw,), {})(**kwargs)
        for cam_name, cam_args in acquisition_description['cameras'].items():
            kwargs = {'session_path': session_path, 'cameras': [cam_name], **cam_args}
            tasks[tn] = type((tn := f'VideoCompress_{cam_name}'), (vtasks.VideoCompress,), {})(
                **kwargs, parents=[tasks['VideoRegisterRaw']] + sync_tasks)
            tasks[tn] = type((tn := f'VideoSync_{cam_name}'), (vtasks.VideoSyncQc,), {})(
                **kwargs, parents=[tasks[f'VideoCompress_{cam_name}']] + sync_tasks)
        tasks[tn] = type((tn := 'DLC'), (epp.EphysDLC,), {})(
            session_path=session_path, parents=[tasks[k] for k in tasks if k.startswith('VideoCompress')])
        tasks['PostDLC'] = type('PostDLC', (epp.EphysPostDLC,), {})(
            session_path=session_path, parents=[tasks['DLC']] + [tasks[k] for k in tasks if k.startswith('VideoSync')])

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
                                                                                    parents=sync_tasks)
        tasks['WidefieldFOV'] = type('WidefieldFOV', (wtasks.WidefieldFOV,), {})(session_path=session_path)

    p = mtasks.Pipeline(session_path=session_path, **pkwargs)
    p.tasks = tasks

    return p

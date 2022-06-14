from collections import OrderedDict
import ibllib.pipes.tasks as mtasks
import ibllib.pipes.ephys_preprocessing as epp
import ibllib.pipes.training_preprocessing as tpp


def get_acquisition_description(experiment):
    """"
    This is a set of example acqusition descriptions for experiments:
    -   choice_world_recording
    -   choice_world_biased
    -   choice_world_training
    -   choice_world_habituation
    That are part of the IBL pipeline
    """
    if experiment == 'choice_world_recording':   # canonical ephys
        acquisition_description = {  # this is the current ephys pipeline description
            'cameras': [
                'raw_video_data/_iblrig_rightCamera.raw',
                'raw_video_data/_iblrig_bodyCamera.raw.mp4',
                'raw_video_data/_iblrig_leftCamera.raw.mp4'
            ],
            'neuropixel': {
                'probe00': 'raw_ephys_data/probe00',
                'probe01': 'raw_ephys_data/probe01',
            },
            'tasks': {
                'choice_world_recording': 'raw_behavior_data',
                'choice_world_passive': 'raw_passive_data'
            },
            'microphone': {},
            'sync': 'nidq'
        }
    else:
        acquisition_description = {  # this is the current ephys pipeline description
            'cameras': [
                'raw_video_data/_iblrig_leftCamera.raw.mp4'
            ],
            'tasks': {
                experiment: 'raw_passive_data'
            },
            'microphone': {},
            'sync': 'bpod'
        }
    return acquisition_description


class DummyTask(mtasks.Task):
    def _run(self):
        pass


def make_pipeline(pipeline_description, session_path=None):
    # NB: this pattern is a pattern for dynamic class creation
    # tasks['SyncPulses'] = type('SyncPulses', (epp.EphysPulses,), {})(session_path=session_path)
    assert session_path
    tasks = OrderedDict()
    sync = pipeline_description['sync']
    if sync == 'nidq':
        tasks[f'SyncPulses{sync}'] = type(f'SyncPulses{sync}', (epp.EphysPulses,), {})(session_path=session_path)
    elif sync == 'bpod':
        tasks[f'SyncPulses{sync}'] = type(f'SyncPulses{sync}', (epp.EphysPulses,), {})(session_path=session_path)



    if 'neuropixel' in pipeline_description:
        for pname, rpath in pipeline_description['neuropixel'].items():
            kwargs = {'session_path': session_path, 'args': dict(pname=pname)}
            tasks[f'Compression_{pname}'] = type(
                f'Compression_{pname}', (epp.EphysMtscomp,), {})(**kwargs)

            tasks[f'SpikeSorting_{pname}'] = type(
                f'SpikeSorting_{pname}', (epp.SpikeSorting,), {})(
                **kwargs, parents=[tasks[f'Compression_{pname}'], tasks[f'SyncPulses{sync}']])

            tasks[f'CellsQC_{pname}'] = type(
                f'CellsQC_{pname}', (epp.EphysCellsQc,), {})(
                **kwargs, parents=[tasks[f'SpikeSorting_{pname}']])

    for protocol in pipeline_description.get('tasks', []):
        tasks[protocol] = type(protocol, (DummyTask,), {})(session_path=session_path, parents=[tasks[f'SyncPulses{sync}']])
        if protocol in ['']:
            tasks[protocol] = type(protocol, (DummyTask,), {})(session_path=session_path, parents=[tasks[f'SyncPulses{sync}']])
        if protocol.startswith('choice_world') and protocol != 'choice_world_passive':
            tasks["Training Status"] = type("Training Status", (DummyTask,), {})(session_path=session_path, parents=[tasks[protocol]])

    if 'cameras' in pipeline_description:
        tasks['VideoCompress'] = type('VideoCompress', (epp.EphysVideoCompress,), {})(session_path=session_path)
        tasks['DLC'] = type('DLC', (epp.EphysDLC,), {})(session_path=session_path, parents=[tasks['VideoCompress']])
        tasks['VideoSync'] = type('VideoSync', (epp.EphysVideoSyncQc,), {})(
            session_path=session_path, parents=[tasks['VideoCompress'], tasks[f'SyncPulses{sync}']])
        tasks['PostDLC'] = type('PostDLC', (epp.EphysPostDLC,), {})(
            session_path=session_path, parents=[tasks['DLC'], tasks['VideoSync']])

    if 'microphone' in pipeline_description:
        tasks['Audio'] = type('Audio', (epp.EphysAudio,), {})(session_path=session_path)

    p = mtasks.Pipeline(session_path=session_path)
    p.tasks = tasks

    return p


session_meta = {
    'protocols': {'biasedChoiceWorld': 'raw_behaviour_data',
                  'passiveChoiceWorld': 'raw_passive_data'},  #TODO how do we get the collection from the rig brahhh
    'devices': ['ephys', 'video', 'widefield'],
    'sync': 'fpga'
}


def read_session_meta(session_path):

    protocols = session_meta['protocols']
    devices = session_meta['devices']
    sync = session_meta['sync']

    return protocols, devices, sync


def build_pipeline(session_path):
    protocols, devices, sync = read_session_meta(session_path)



def get_device_pipeline(device, session_path, sync):
    if device == 'ephys':
        tasks = get_ephys_pipeline(session_path, sync)

    elif device == 'audio':
        tasks = get_audio_pipeline(session_path, sync) # this requires a bit of thought as it can also be per protocol

    elif device == 'video':
        tasks = get_video_pipeline(session_path, sync)

    elif device == 'widefield':
        tasks = get_widefield_pipeline(session_path, sync)

    elif device == 'fibrephotometry':
        tasks = get_fibrephotometry_pipeline(session_path, sync)

    return tasks


class DynamicPipeline(mtasks.Pipeline):
    """
    This is trying to replicate the current IBL
    """
    label = __name__

    def __init__(self, session_path, **kwargs):
        super(DynamicPipeline, self).__init__(session_path, **kwargs)

        tasks = OrderedDict()
        protocols, devices, sync = read_session_meta(session_path)

        # Create job for syncing
        sync_task = get_sync_pipeline(sync, session_path)
        tasks.update(sync_task)

        # Create task related jobs
        for prot, coll in protocols.items():
            tasks.update(get_protocol_pipeline(session_path, prot, coll, sync, sync_task))

        # Create device related jobs
        for device in devices:
            tasks.update(get_device_pipeline(device))



        # Loop through the protocols that were run and fill pipeline with tasks
        # TODO how do we do this
        for prot in protocols:
            tasks.update(get_protocol_subpipeline(prot))

        # Loop through the devices that we have and fill in the pipeline
        for device in devices:
            tasks.update(get_device_subpipeline(device))

        # How do we do audio??


if __name__ == '__main__':
    from one.api import ONE
    from ibllib.pipes.dynamic_pipeline import make_pipeline, get_acquisition_description

    one = ONE(base_url="https://alyx.internationalbrainlab.org", cache_dir="/media/olivier/F1C9-7D73/one")

    examples = {
        "451bc9c0-113c-408e-a924-122ffe44306e": 'choice_world_habituation',
        "cad382e5-7b1a-4e73-b892-318b5e9decc4": 'choice_world_training',
        "47e53f41-a928-4b6d-b8cc-78fd10d03456": 'choice_world_biased',
        "aed404ce-b3fb-454b-ac43-2f12198c9eaf": 'choice_world_recording',
    }

    for eid in examples:
        ad = get_acquisition_description(examples[eid])
        p = make_pipeline(ad, session_path='wpd')
        p.make_graph()

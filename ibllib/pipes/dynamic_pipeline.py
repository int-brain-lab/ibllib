

session_meta = {
    'protocols': {'biasedChoiceWorld': 'raw_behaviour_data',
                  'passiveChoiceWorld': 'raw_passive_data'} #TODO how do we get the collection from the rig brahhh
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


class DynamicPipeline(tasks.Pipeline):
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




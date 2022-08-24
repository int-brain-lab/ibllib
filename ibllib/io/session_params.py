"""
Functions to load in information from experiment.description.yaml file
"""
from pathlib import Path
import yaml


def read_params(session_path):
    """
    Read the yaml parameter file.
    :param session_path: session path (will joinpath session_params.yml) or full file path yaml
    :return: dictionary
    """
    session_path = Path(session_path)
    yaml_file = session_path.joinpath('experiment.description.yaml') if session_path.is_dir() else session_path
    if not yaml_file.exists():
        return

    with open(yaml_file, 'r') as fid:
        params = yaml.full_load(fid)

    if 'version' not in params:
        params['version'] = '1.0.0'
    return params


def get_cameras(sess_params):
    devices = sess_params.get('devices', {})
    cameras = devices.get('cameras', None)
    return None if not cameras else list(cameras.keys())


def get_sync(sess_params):
    sync = sess_params.get('sync', None)
    if not sync:
        return None
    else:
        (sync, _), = sync.items()
    return sync


def get_sync_collection(sess_params):
    sync = sess_params.get('sync', None)
    if not sync:
        return None
    else:
        (_, sync_details), = sync.items()
    return sync_details.get('collection', None)


def get_sync_extension(sess_params):
    sync = sess_params.get('sync', None)
    if not sync:
        return None
    else:
        (_, sync_details), = sync.items()
    return sync_details.get('extension', None)


def get_sync_namespace(sess_params):
    sync = sess_params.get('sync', None)
    if not sync:
        return None
    else:
        (_, sync_details), = sync.items()
    return sync_details.get('acquisition_software', None)


def get_task_protocol(sess_params, task_collection):
    protocols = sess_params.get('tasks', None)
    if not protocols:
        return None
    else:
        protocol = None
        for prot, details in sess_params.get('tasks').items():
            if details.get('collection') == task_collection:
                protocol = prot

        return protocol


def get_task_collection(sess_params):
    protocols = sess_params.get('tasks', None)
    if not protocols:
        return None
    elif len(protocols) > 1:
        return 'raw_behavior_data'
    else:
        for prot, details in protocols.items():
            return details['collection']


def get_device_collection(sess_params, device):
    # TODO
    return None

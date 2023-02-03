from one.webclient import no_cache

from ibllib.pipes.tasks import Task
import ibllib.io.session_params as sess_params
from ibllib.qc.base import sign_off_dict, SIGN_OFF_CATEGORIES
import logging

_logger = logging.getLogger(__name__)


class DynamicTask(Task):

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.session_params = self.read_params_file()

        # TODO Which should be default?
        # Sync collection
        self.sync_collection = self.get_sync_collection(kwargs.get('sync_collection', None))
        # Sync type
        self.sync = self.get_sync(kwargs.get('sync', None))
        # Sync extension
        self.sync_ext = self.get_sync_extension(kwargs.get('sync_ext', None))
        # Sync namespace
        self.sync_namespace = self.get_sync_namespace(kwargs.get('sync_namespace', None))

    def get_sync_collection(self, sync_collection=None):
        return sync_collection if sync_collection else sess_params.get_sync_collection(self.session_params)

    def get_sync(self, sync=None):
        return sync if sync else sess_params.get_sync_label(self.session_params)

    def get_sync_extension(self, sync_ext=None):
        return sync_ext if sync_ext else sess_params.get_sync_extension(self.session_params)

    def get_sync_namespace(self, sync_namespace=None):
        return sync_namespace if sync_namespace else sess_params.get_sync_namespace(self.session_params)

    def get_protocol(self, protocol=None, task_collection=None):
        return protocol if protocol else sess_params.get_task_protocol(self.session_params, task_collection)

    def get_task_collection(self, collection=None):
        if not collection:
            collection = sess_params.get_task_collection(self.session_params)
        # If inferring the collection from the experiment description, assert only one returned
        assert collection is None or isinstance(collection, str) or len(collection) == 1
        return collection

    def get_device_collection(self, device, device_collection=None):
        if device_collection:
            return device_collection
        collection_map = sess_params.get_collections(self.session_params['devices'])
        return collection_map.get(device)

    def read_params_file(self):
        params = sess_params.read_params(self.session_path)

        if params is None:
            return {}

        # TODO figure out the best way
        # if params is None and self.one:
        #     # Try to read params from alyx or try to download params file
        #     params = self.one.load_dataset(self.one.path2eid(self.session_path), 'params.yml')
        #     params = self.one.alyx.rest()

        return params


class BehaviourTask(DynamicTask):

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)

        self.collection = self.get_task_collection(kwargs.get('collection', None))
        # Task type (protocol)
        self.protocol = self.get_protocol(kwargs.get('protocol', None), task_collection=self.collection)

        self.protocol_number = self.get_protocol_number(kwargs.get('protocol_number'), task_protocol=self.protocol)

        self.output_collection = 'alf'
        # Do not use kwargs.get('number', None) -- this will return None if number is 0
        if self.protocol_number is not None:
            self.output_collection += f'/task_{self.protocol_number:02}'

    def get_protocol(self, protocol=None, task_collection=None):
        return protocol if protocol else sess_params.get_task_protocol(self.session_params, task_collection)

    def get_task_collection(self, collection=None):
        if not collection:
            collection = sess_params.get_task_collection(self.session_params)
        # If inferring the collection from the experiment description, assert only one returned
        assert collection is None or isinstance(collection, str) or len(collection) == 1
        return collection

    def get_protocol_number(self, number=None, task_protocol=None):
        if number is None:  # Do not use "if not number" as that will return True if number is 0
            number = sess_params.get_task_protocol_number(self.session_params, task_protocol)
        # If inferring the number from the experiment description, assert only one returned (or something went wrong)
        assert number is None or isinstance(number, int)
        return number


class VideoTask(DynamicTask):

    def __init__(self, session_path, cameras, **kwargs):
        super().__init__(session_path, cameras=cameras, **kwargs)
        self.cameras = cameras
        self.device_collection = self.get_device_collection('cameras', kwargs.get('device_collection', 'raw_video_data'))
        # self.collection = self.get_task_collection(kwargs.get('collection', None))


class AudioTask(DynamicTask):

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('microphone', kwargs.get('device_collection', 'raw_behavior_data'))


class EphysTask(DynamicTask):

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)

        self.pname = self.get_pname(kwargs.get('pname', None))
        self.nshanks, self.pextra = self.get_nshanks(kwargs.get('nshanks', None))
        self.device_collection = self.get_device_collection('neuropixel', kwargs.get('device_collection', 'raw_ephys_data'))

    def get_pname(self, pname):
        # pname can be a list or a string
        pname = self.kwargs.get('pname', pname)

        return pname

    def get_nshanks(self, nshanks=None):
        nshanks = self.kwargs.get('nshanks', nshanks)
        if nshanks is not None:
            pextra = [chr(97 + int(shank)) for shank in range(nshanks)]
        else:
            pextra = []

        return nshanks, pextra


class WidefieldTask(DynamicTask):
    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)

        self.device_collection = self.get_device_collection('widefield', kwargs.get('device_collection', 'raw_widefield_data'))


class RegisterRawDataTask(DynamicTask):  # TODO write test
    """
    Base register raw task.
    To rename files
     1. input and output must have the same length
     2. output files must have full filename
    """

    priority = 100
    job_size = 'small'

    def rename_files(self, symlink_old=False, **kwargs):

        # If no inputs are given, we don't do any renaming
        if len(self.input_files) == 0:
            return

        # Otherwise we need to make sure there is one to one correspondence for renaming files
        assert len(self.input_files) == len(self.output_files)

        for before, after in zip(self.input_files, self.output_files):
            old_file, old_collection, required = before
            old_path = self.session_path.joinpath(old_collection).glob(old_file)
            old_path = next(old_path, None)
            # if the file doesn't exist and it is not required we are okay to continue
            if not old_path and not required:
                continue

            new_file, new_collection, _ = after
            new_path = self.session_path.joinpath(new_collection, new_file)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.replace(new_path)
            if symlink_old:
                old_path.symlink_to(new_path)

    def _run(self, **kwargs):
        self.rename_files(**kwargs)
        out_files = []
        n_required = 0
        for file_sig in self.output_files:
            file_name, collection, required = file_sig
            n_required += required
            file_path = self.session_path.joinpath(collection).glob(file_name)
            file_path = next(file_path, None)
            if not file_path and not required:
                continue
            elif not file_path and required:
                _logger.error(f'expected {file_sig} missing')
            else:
                out_files.append(file_path)

        if len(out_files) < n_required:
            self.status = -1

        return out_files


class ExperimentDescriptionRegisterRaw(RegisterRawDataTask):
    """dict of list: custom sign off keys corresponding to specific devices"""
    sign_off_categories = SIGN_OFF_CATEGORIES

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [('*experiment.description.yaml', '', True)]
        }
        return signature

    def _run(self, **kwargs):
        # Register experiment description file
        out_files = super(ExperimentDescriptionRegisterRaw, self)._run(**kwargs)
        if not self.one.offline and self.status == 0:
            with no_cache(self.one.alyx):  # Ensure we don't load the cached JSON response
                eid = self.one.path2eid(self.session_path, query_type='remote')
            exp_dec = sess_params.read_params(out_files[0])
            data = sign_off_dict(exp_dec, sign_off_categories=self.sign_off_categories)
            self.one.alyx.json_field_update('sessions', eid, data=data)
        return out_files

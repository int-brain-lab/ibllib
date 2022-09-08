from ibllib.pipes.tasks import Task
import ibllib.io.session_params as sess_params
import logging
_logger = logging.getLogger('ibllib')


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
        # Task collection (this needs to be specified in the task kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))
        # Task type (protocol)
        self.protocol = self.get_protocol(kwargs.get('protocol', None), task_collection=self.collection)

    def get_sync_collection(self, sync_collection=None):

        return sync_collection if sync_collection else sess_params.get_sync_collection(self.session_params)

    def get_sync(self, sync=None):

        return sync if sync else sess_params.get_sync(self.session_params)
        # params_sync = sess_params.get_sync_collection(self.session_params)
        # return sync if not params_sync else params_sync

    def get_sync_extension(self, sync_ext=None):
        return sync_ext if sync_ext else sess_params.get_sync_extension(self.session_params)
        # params_sync_ext = sess_params.get_sync_extension(self.session_params)
        # return sync_ext if not params_sync_ext else params_sync_ext

    def get_sync_namespace(self, sync_namespace=None):
        return sync_namespace if sync_namespace else sess_params.get_sync_namespace(self.session_params)
        # params_sync_namespace = sess_params.get_sync_namespace(self.session_params)
        # return sync_namespace if not params_sync_namespace else params_sync_namespace

    def get_protocol(self, protocol=None, task_collection=None):
        return protocol if protocol else sess_params.get_task_protocol(self.session_params, task_collection)
        # params_protocol = sess_params.get_task_protocol(self.session_params, task_collection)
        # return protocol if not params_protocol else params_protocol

    def get_task_collection(self, collection=None):
        return collection if collection else sess_params.get_task_collection(self.session_params)
        # params_task_collection = sess_params.get_task_collection(self.session_params)
        # return collection if not params_task_collection else params_task_collection

    def get_device_collection(self, device, device_collection=None):
        return device_collection if device_collection else sess_params.get_device_collection(self.session_params, device)
        # params_device_collection = sess_params.get_device_collection(self.session_params, device)
        # return device_collection if not params_device_collection else params_device_collection

    def read_params_file(self):
        params = sess_params.read_params(self.session_path)

        if params is None:
            return {}

        # TODO figure out the best way
        # if params is None and self.one:
        #     # Try to read params from alyx or try to download params file
        #     params = self.one.load_datasets(self.one.path2eid(self.session_path), 'params.yml')
        #     params = self.one.alyx.rest()

        return params


class VideoTask(DynamicTask):

    def __init__(self, session_path, cameras, **kwargs):
        super().__init__(session_path, cameras=cameras, **kwargs)
        self.cameras = cameras
        self.device_collection = self.get_device_collection('cameras', kwargs.get('device_collection', 'raw_video_data'))


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
    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [('*experiment.description.yaml', '', True)]
        }
        return signature

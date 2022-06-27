from abc import ABC

from ibllib.pipes.tasks import Task
import ibllib.io.session_params as sess_params


class DynamicTask(Task):

    def __init__(self, session_path, **kwargs):

        super().__init__(session_path, **kwargs)

        self.session_params = self.read_params_file()

        # Sync collection
        self.sync_collection = self.get_sync_collection(kwargs.get('sync_collection', None))
        # Sync type
        self.sync = self.get_sync(kwargs.get('sync', None))
        # Sync extension
        self.sync_ext = self.get_sync_extension(kwargs.get('sync_ext', None))
        # Task collection (this needs to be specified in the task kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))
        # Task type (protocol)
        self.protocol = self.get_protocol(self.collection, kwargs.get('protocol', None))
        # Main task collection (for when task protocols are chained together)
        self.main_task_collection = self.get_main_task_collection(kwargs.get('main_collection', None))  # TODO improve name

    def get_sync_collection(self, sync_collection=None):

        params_sync_collection = sess_params.get_sync_collection(self.session_params)
        return sync_collection if not params_sync_collection else params_sync_collection

    def get_sync(self, sync=None):

        params_sync = sess_params.get_sync_collection(self.session_params)
        return sync if not params_sync else params_sync

    def get_sync_extension(self, sync_ext=None):

        params_sync_ext = sess_params.get_sync_extension(self.session_params)
        return sync_ext if not params_sync_ext else params_sync_ext

    def get_task_collection(self, task_collection=None):
        """
        Finds the collection of
        task_collection is a parameter that cannot be automatically inferred from the session_params file, e.g if two different
        protocols are run in one session, we would need two BehaviorTrials task, one for each task protocol.

        :param task_collection:
        :return:
        """
        # Attempt to get from runtime_args embedded in task architecture
        task_collection = self.kwargs.get('task_collection', task_collection)

        if not task_collection:
            task_collection = sess_params.get_main_task_collection(self.session_params)

        return task_collection

    def get_protocol(self, task_collection, protocol=None):
        params_protocol = sess_params.get_task_protocol(self.session_params, task_collection)
        return protocol if not params_protocol else params_protocol

    def get_main_task_collection(self, main_task_collection=None):

        params_main_task_collection = sess_params.get_main_task_collection(self.session_params)
        return main_task_collection if not params_main_task_collection else params_main_task_collection

    def get_device_collection(self, device, device_collection=None):
        params_device_collection = sess_params.get_device_collection(self.session_params, device)
        return device_collection if not params_device_collection else params_device_collection

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
        self.device_collection = self.get_device_collection('neuropixel', kwargs.get('device_collection', 'raw_ephys_data'))

    def get_pname(self, pname):
        pname = self.kwargs.get('pname', pname)

        return pname


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
    cpu = 1
    io_charge = 90
    level = 0
    force = False

    def rename_files(self, symlink_old=False):

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
        for file_sig in self.output_files:
            file_name, collection, required = file_sig
            file_path = self.session_path.joinpath(collection).glob(file_name)
            file_path = next(file_path, None)
            if not file_path and not required:
                continue
            out_files.append(file_path)

        return out_files

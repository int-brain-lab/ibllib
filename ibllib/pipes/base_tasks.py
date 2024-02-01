"""Abstract base classes for dynamic pipeline tasks."""
import logging
from pathlib import Path

from packaging import version
from one.webclient import no_cache
from iblutil.util import flatten

from ibllib.pipes.tasks import Task
import ibllib.io.session_params as sess_params
from ibllib.qc.base import sign_off_dict, SIGN_OFF_CATEGORIES
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap

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

    extractor = None
    """ibllib.io.extractors.base.BaseBpodExtractor: A trials extractor object."""

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
        """
        Return the task protocol name.

        This returns the task protocol based on the task collection. If `protocol` is not None, this
        acts as an identity function. If both `task_collection` and `protocol` are None, returns
        the protocol defined in the experiment description file only if a single protocol was run.
        If the `task_collection` is not None, the associated protocol name is returned.


        Parameters
        ----------
        protocol : str
            A task protocol name. If not None, the same value is returned.
        task_collection : str
            The task collection whose protocol name to return. May be None if only one protocol run.

        Returns
        -------
        str, None
            The task protocol name, or None, if no protocol found.

        Raises
        ------
        ValueError
            For session with multiple task protocols, a task collection must be passed.
        """
        if protocol:
            return protocol
        protocol = sess_params.get_task_protocol(self.session_params, task_collection) or None
        if isinstance(protocol, set):
            if len(protocol) == 1:
                protocol = next(iter(protocol))
            else:
                raise ValueError('Multiple task protocols for session. Task collection must be explicitly defined.')
        return protocol

    def get_task_collection(self, collection=None):
        """
        Return the task collection.

        If `collection` is not None, this acts as an identity function. Otherwise loads it from
        the experiment description if only one protocol was run.

        Parameters
        ----------
        collection : str
            A task collection. If not None, the same value is returned.

        Returns
        -------
        str, None
            The task collection, or None if no task protocols were run.

        Raises
        ------
        AssertionError
            Raised if multiple protocols were run and collection is None, or if experiment
            description file is improperly formatted.

        """
        if not collection:
            collection = sess_params.get_task_collection(self.session_params)
        # If inferring the collection from the experiment description, assert only one returned
        assert collection is None or isinstance(collection, str) or len(collection) == 1
        return collection

    def get_protocol_number(self, number=None, task_protocol=None):
        """
        Return the task protocol number.

        Numbering starts from 0. If the 'protocol_number' field is missing from the experiment
        description, None is returned. If `task_protocol` is None, the first protocol number if n
        protocols == 1, otherwise returns None.

        NB: :func:`ibllib.pipes.dynamic_pipeline.make_pipeline` will determine the protocol number
        from the order of the tasks in the experiment description if the task collection follows
        the pattern 'raw_task_data_XX'. If the task protocol does not follow this pattern, the
        experiment description file should explicitly define the number with the 'protocol_number'
        field.

        Parameters
        ----------
        number : int
            The protocol number. If not None, the same value is returned.
        task_protocol : str
            The task protocol name.

        Returns
        -------
        int, None
            The task protocol number, if defined.
        """
        if number is None:  # Do not use "if not number" as that will return True if number is 0
            number = sess_params.get_task_protocol_number(self.session_params, task_protocol)
        # If inferring the number from the experiment description, assert only one returned (or something went wrong)
        assert number is None or isinstance(number, int)
        return number

    @staticmethod
    def _spacer_support(settings):
        """
        Spacer support was introduced in v7.1 for iblrig v7 and v8.0.1 in v8.

        Parameters
        ----------
        settings : dict
            The task settings dict.

        Returns
        -------
        bool
            True if task spacers are to be expected.
        """
        v = version.parse
        ver = v(settings.get('IBLRIG_VERSION') or '100.0.0')
        return ver not in (v('100.0.0'), v('8.0.0')) and ver >= v('7.1.0')

    def extract_behaviour(self, save=True):
        """Extract trials data.

        This is an abstract method called by `_run` and `run_qc` methods.  Subclasses should return
        the extracted trials data and a list of output files. This method should also save the
        trials extractor object to the :prop:`extractor` property for use by `run_qc`.

        Parameters
        ----------
        save : bool
            Whether to save the extracted data as ALF datasets.

        Returns
        -------
        dict
            A dictionary of trials data.
        list of pathlib.Path
            A list of output file paths if save == true.
        """
        return None, None

    def run_qc(self, trials_data=None, update=True):
        """Run task QC.

        Subclass method should return the QC object. This just validates the trials_data is not
        None.

        Parameters
        ----------
        trials_data : dict
            A dictionary of extracted trials data. The output of :meth:`extract_behaviour`.
        update : bool
            If true, update Alyx with the QC outcome.

        Returns
        -------
        ibllib.qc.task_metrics.TaskQC
            A TaskQC object replete with task data and computed metrics.
        """
        self._assert_trials_data(trials_data)
        return None

    def _assert_trials_data(self, trials_data=None):
        """Check trials data available.

        Called by :meth:`run_qc`, this extracts the trial data if `trials_data` is None, and raises
        if :meth:`extract_behaviour` returns None.

        Parameters
        ----------
        trials_data : dict, None
            A dictionary of extracted trials data or None.

        Returns
        -------
        trials_data : dict
            A dictionary of extracted trials data. The output of :meth:`extract_behaviour`.
        """
        if not self.extractor or trials_data is None:
            trials_data, _ = self.extract_behaviour(save=False)
        if not (trials_data and self.extractor):
            raise ValueError('No trials data and/or extractor found')
        return trials_data


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


class MesoscopeTask(DynamicTask):
    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)

        self.device_collection = self.get_device_collection(
            'mesoscope', kwargs.get('device_collection', 'raw_imaging_data_[0-9]*'))

    def get_signatures(self, **kwargs):
        """
        From the template signature of the task, create the exact list of inputs and outputs to expect based on the
        available device collection folders

        Necessary because we don't know in advance how many device collection folders ("imaging bouts") to expect
        """
        self.session_path = Path(self.session_path)
        # Glob for all device collection (raw imaging data) folders
        raw_imaging_folders = [p.name for p in self.session_path.glob(self.device_collection)]
        # For all inputs and outputs that are part of the device collection, expand to one file per folder
        # All others keep unchanged
        self.input_files = [(sig[0], sig[1].replace(self.device_collection, folder), sig[2])
                            for folder in raw_imaging_folders for sig in self.signature['input_files']]
        self.output_files = [(sig[0], sig[1].replace(self.device_collection, folder), sig[2])
                             for folder in raw_imaging_folders for sig in self.signature['output_files']]

    def load_sync(self):
        """
        Load the sync and channel map.

        This method may be expanded to support other raw DAQ data formats.

        Returns
        -------
        one.alf.io.AlfBunch
            A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses
            and the corresponding channel numbers.
        dict
            A map of channel names and their corresponding indices.
        """
        alf_path = self.session_path / self.sync_collection
        if self.get_sync_namespace() == 'timeline':
            # Load the sync and channel map from the raw DAQ data
            sync, chmap = load_timeline_sync_and_chmap(alf_path)
        else:
            raise NotImplementedError
        return sync, chmap


class RegisterRawDataTask(DynamicTask):
    """
    Base register raw task.
    To rename files
     1. input and output must have the same length
     2. output files must have full filename
    """

    priority = 100
    job_size = 'small'

    def rename_files(self, symlink_old=False):

        # If either no inputs or no outputs are given, we don't do any renaming
        if not all(map(len, (self.input_files, self.output_files))):
            return

        # Otherwise we need to make sure there is one to one correspondence for renaming files
        assert len(self.input_files) == len(self.output_files)

        for before, after in zip(self.input_files, self.output_files):
            old_file, old_collection, required = before
            old_path = self.session_path.joinpath(old_collection).glob(old_file)
            old_path = next(old_path, None)
            # if the file doesn't exist and it is not required we are okay to continue
            if not old_path:
                if required:
                    raise FileNotFoundError(str(old_file))
                else:
                    continue

            new_file, new_collection, _ = after
            new_path = self.session_path.joinpath(new_collection, new_file)
            if old_path == new_path:
                continue
            new_path.parent.mkdir(parents=True, exist_ok=True)
            _logger.debug('%s -> %s', old_path.relative_to(self.session_path), new_path.relative_to(self.session_path))
            old_path.replace(new_path)
            if symlink_old:
                old_path.symlink_to(new_path)

    def register_snapshots(self, unlink=False, collection=None):
        """
        Register any photos in the snapshots folder to the session. Typically imaging users will
        take numerous photos for reference.  Supported extensions: .jpg, .jpeg, .png, .tif, .tiff

        If a .txt file with the same name exists in the same location, the contents will be added
        to the note text.

        Parameters
        ----------
        unlink : bool
            If true, files are deleted after upload.
        collection : str, list, optional
            Location of 'snapshots' folder relative to the session path. If None, uses
            'device_collection' attribute (if exists) or root session path.

        Returns
        -------
        list of dict
            The newly registered Alyx notes.
        """
        collection = getattr(self, 'device_collection', None) if collection is None else collection
        collection = collection or ''  # If not defined, use no collection
        if collection and '*' in collection:
            collection = [p.name for p in self.session_path.glob(collection)]
            # Check whether folders on disk contain '*'; this is to stop an infinite recursion
            assert not any('*' in c for c in collection), 'folders containing asterisks not supported'
        # If more that one collection exists, register snapshots in each collection
        if collection and not isinstance(collection, str):
            return flatten(filter(None, [self.register_snapshots(unlink, c) for c in collection]))
        snapshots_path = self.session_path.joinpath(*filter(None, (collection, 'snapshots')))
        if not snapshots_path.exists():
            return

        eid = self.one.path2eid(self.session_path, query_type='remote')
        if not eid:
            _logger.warning('Failed to upload snapshots: session not found on Alyx')
            return
        note = dict(user=self.one.alyx.user, content_type='session', object_id=eid, text='')

        notes = []
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        for snapshot in filter(lambda x: x.suffix.lower() in exts, snapshots_path.glob('*.*')):
            _logger.debug('Uploading "%s"...', snapshot.relative_to(self.session_path))
            if snapshot.with_suffix('.txt').exists():
                with open(snapshot.with_suffix('.txt'), 'r') as txt_file:
                    note['text'] = txt_file.read().strip()
            else:
                note['text'] = ''
            with open(snapshot, 'rb') as img_file:
                files = {'image': img_file}
                notes.append(self.one.alyx.rest('notes', 'create', data=note, files=files))
            if unlink:
                snapshot.unlink()
        # If nothing else in the snapshots folder, delete the folder
        if unlink and next(snapshots_path.rglob('*'), None) is None:
            snapshots_path.rmdir()
        _logger.info('%i snapshots uploaded to Alyx', len(notes))
        return notes

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

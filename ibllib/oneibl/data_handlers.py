"""Downloading of task dependent datasets and registration of task output datasets.

The DataHandler class is used by the pipes.tasks.Task class to ensure dependent datasets are
present and to register and upload the output datasets.  For examples on how to run a task using
specific data handlers, see :func:`ibllib.pipes.tasks`.
"""

import logging
import pandas as pd
from pathlib import Path, PurePosixPath
import shutil
import os
import abc
from collections import defaultdict
from time import time
from copy import copy

from one.api import ONE
from one.webclient import AlyxClient
from one.util import filter_datasets
from one.alf.path import add_uuid_string, get_alf_path, ensure_alf_path
from one.alf.cache import _make_datasets_df
from iblutil.util import flatten, ensure_list

from ibllib.oneibl.registration import register_dataset, get_lab, get_local_data_repository
from ibllib.oneibl.patcher import FTPPatcher, SDSCPatcher, SDSC_ROOT_PATH, SDSC_PATCH_PATH, S3Patcher


_logger = logging.getLogger(__name__)


def repository_store(repository_name):
    """Return the store part of an Alyx data repository name.

    Data repositories are named '<store>_<scope>', where the store is the physical location of the
    data (e.g. 'flatiron', 'aws') and the scope is either a lab name or 'aggregates', e.g.
    'flatiron_cortexlab', 'aws_aggregates'. The store determines which protocol is required to
    reach the repository (see :attr:`DataHandler.protocols`), while the scope is a property of the
    data itself.

    Parameters
    ----------
    repository_name : str
        The name of a data repository, e.g. 'flatiron_cortexlab'.

    Returns
    -------
    str
        The store part of the repository name, e.g. 'flatiron'.

    Notes
    -----
    - Once the per-lab repositories are consolidated into a single repository per store (e.g. one
      'flatiron' repository instead of 'flatiron_<lab>'), this will simply return the name as is.
    """
    return repository_name.split('_')[0]


class ExpectedDataset:
    """An expected input or output dataset."""

    inverted = False

    def __init__(self, name, collection, register=None, revision=None, unique=True):
        """
        An expected input or output dataset.

        NB: This should not be instantiated directly, but rather via the `input` or `output`
        static method.

        Parameters
        ----------
        name : str, None
            A dataset name or glob pattern.
        collection : str, None
            An ALF collection or pattern.
        register : bool
            Whether to register the file. Default is False for input files, True for output
            files.
        revision : str
            An optional revision.
        unique : bool
            Whether identifier pattern is expected to match a single dataset or several.  NB: This currently does not
            affect the output of `find_files`.
        """
        if not (collection is None or isinstance(collection, str)):
            collection = '/'.join(collection)
        self._identifiers = (collection, revision, name)
        self.operator = None
        self._register = register or False
        self.inverted = False
        self.name = None
        self.unique = unique

    @property
    def register(self):
        """bool: whether to register the output file."""
        return self._register

    @register.setter
    def register(self, value):
        """bool: whether to register the output file."""
        if self.operator is not None:
            raise AttributeError('cannot set register attribute for operator datasets')
        self._register = value

    @property
    def identifiers(self):
        """tuple: the identifying parts of the dataset.

        If no operator is applied, the identifiers are (collection, revision, name).
        If an operator is applied, a tuple of 3-element tuples is returned.
        """
        if self.operator is None:
            return self._identifiers
        # Flatten nested identifiers into tuple of 3-element tuples
        identifiers = []
        for x in self._identifiers:
            add = identifiers.extend if x.operator else identifiers.append
            add(x.identifiers)
        return tuple(identifiers)

    @property
    def glob_pattern(self):
        """str, tuple of str: one or more glob patterns."""
        if self.operator is None:
            return str(PurePosixPath(*filter(None, self._identifiers)))
        else:
            return tuple(flatten(x.glob_pattern for x in self._identifiers))

    def __repr__(self):
        """Represent the dataset object as a string.

        If the `name` property is not None, it is returned, otherwise the identifies are used to
        format the name.
        """
        name = self.__class__.__name__
        if self.name:
            return f'<{name}({self.name})>'
        if self.operator:
            sym = {'or': '|', 'and': '&', 'xor': '^'}
            patterns = [d.__repr__() for d in self._identifiers]
            pattern = f'{sym[self.operator]:^3}'.join(patterns)
            if self.inverted:
                pattern = f'~({pattern})'
        else:
            pattern = ('~' if self.inverted else '') + self.glob_pattern
        return f'<{name}({pattern})>'

    def find_files(self, session_path, register=False):
        """Find files on disk.

        Uses glob patterns to find dataset(s) on disk.

        Parameters
        ----------
        session_path : pathlib.Path, str
            A session path within which to glob for the dataset(s).
        register : bool
            Only return files intended to be registered.

        Returns
        -------
        bool
            True if the dataset is found on disk or is optional.
        list of pathlib.Path
            A sorted list of matching dataset files.
        missing, None, str, set of str
            One or more glob patterns that either didn't yield files (or did in the case of inverted datasets).

        Notes
        -----
        - Currently if `unique` is true and multiple files are found, all files are returned without an exception raised
          although this may change in the future.
        - If `register` is false, all files are returned regardless of whether they are intended to be registered.
        - If `register` is true, an input with register=True may not be returned if part of an OR operation.
        - If `inverted` is true, and files are found, the glob pattern is returned as missing.
        - If XOR, returns all patterns if all are present when only one should be, otherwise returns all missing
          patterns.
        - Missing (or unexpectedly found) patterns are returned despite the dataset being optional.
        """
        session_path = Path(session_path)
        ok, actual_files, missing = False, [], None
        if self.operator is None:
            if register and not self.register:
                return True, actual_files, missing
            actual_files = sorted(session_path.rglob(self.glob_pattern))
            # If no revision pattern provided and no files found, search for any revision
            if self._identifiers[1] is None and not any(actual_files):
                glob_pattern = str(PurePosixPath(self._identifiers[0], '#*#', self._identifiers[2]))
                actual_files = sorted(session_path.rglob(glob_pattern))
            ok = any(actual_files) != self.inverted
            if not ok:
                missing = self.glob_pattern
        elif self.operator == 'and':
            assert len(self._identifiers) == 2
            _ok, _actual_files, _missing = zip(*map(lambda x: x.find_files(session_path, register=register), self._identifiers))
            ok = all(_ok)
            actual_files = flatten(_actual_files)
            missing = set(filter(None, flatten(_missing)))
        elif self.operator == 'or':
            assert len(self._identifiers) == 2
            missing = set()
            for d in self._identifiers:
                ok, actual_files, _missing = d.find_files(session_path, register=register)
                if ok:
                    break
                if missing is not None:
                    missing.update(_missing) if isinstance(_missing, set) else missing.add(_missing)
        elif self.operator == 'xor':
            assert len(self._identifiers) == 2
            _ok, _actual_files, _missing = zip(*map(lambda x: x.find_files(session_path, register=register), self._identifiers))
            ok = sum(_ok) == 1  # and sum(map(bool, map(len, _actual_files))) == 1
            # Return only those datasets that are complete if OK
            actual_files = _actual_files[_ok.index(True)] if ok else flatten(_actual_files)
            if ok:
                missing = set()
            elif all(_ok):  # return all patterns if all present when only one should be, otherwise return all missing
                missing = set(flatten(self.glob_pattern))
            elif not any(_ok):  # return all missing glob patterns if none present
                missing = set(filter(None, flatten(_missing)))
        elif not isinstance(self.operator, str):
            raise TypeError(f'Unrecognized operator type "{type(self.operator)}"')
        else:
            raise NotImplementedError(f'logical {self.operator.upper()} not implemented')

        return ok, sorted(actual_files), missing

    def filter(self, session_datasets, **kwargs):
        """Filter dataset frame by expected datasets.

        Parameters
        ----------
        session_datasets : pandas.DataFrame
            A data frame of session datasets.
        kwargs
            Extra arguments for `one.util.filter_datasets`, namely revision_last_before, qc, and
            ignore_qc_not_set.

        Returns
        -------
        bool
            True if the required dataset(s) are present in the data frame.
        pandas.DataFrame
            A filtered data frame of containing the expected dataset(s).
        """
        # ok, datasets = False, session_datasets.iloc[0:0]
        if self.operator is None:
            collection, revision, file = self._identifiers
            if self._identifiers[1] is not None:
                raise NotImplementedError('revisions not yet supported')
            datasets = filter_datasets(session_datasets, file, collection, wildcards=True, assert_unique=self.unique, **kwargs)
            ok = datasets.empty == self.inverted
        elif self.operator == 'or':
            assert len(self._identifiers) == 2
            for d in self._identifiers:
                ok, datasets = d.filter(session_datasets, **kwargs)
                if ok:
                    break
        elif self.operator == 'xor':
            assert len(self._identifiers) == 2
            _ok, _datasets = zip(*map(lambda x: x.filter(session_datasets, **kwargs), self._identifiers))
            ok = sum(_ok) == 1
            if ok:
                # Return only those datasets that are complete.
                datasets = _datasets[_ok.index(True)]
            else:
                datasets = pd.concat(_datasets)
        elif self.operator == 'and':
            assert len(self._identifiers) == 2
            _ok, _datasets = zip(*map(lambda x: x.filter(session_datasets, **kwargs), self._identifiers))
            ok = all(_ok)
            datasets = pd.concat(_datasets)
        elif not isinstance(self.operator, str):
            raise TypeError(f'Unrecognized operator type "{type(self.operator)}"')
        else:
            raise NotImplementedError(f'logical {self.operator.upper()} not implemented')
        return ok, datasets

    def _apply_op(self, op, other):
        """Apply an operation between two datasets."""
        # Assert both instances of Input or both instances of Output
        if not isinstance(other, (self.__class__, tuple)):
            raise TypeError(
                f'logical operations not supported between objects of type '
                f'{self.__class__.__name__} and {other.__class__.__name__}'
            )
        # Assert operation supported
        if op not in {'or', 'xor', 'and'}:
            raise ValueError(op)
        # Convert tuple to ExpectDataset instance
        if isinstance(other, tuple):
            D = self.input if isinstance(self, Input) else self.output
            other = D(*other)
        # Returned instance should only be optional if both datasets are optional
        is_input = isinstance(self, Input)
        if all(isinstance(x, OptionalDataset) for x in (self, other)):
            D = OptionalInput if is_input else OptionalOutput
        else:
            D = Input if is_input else Output
        # Instantiate 'empty' object
        d = D(None, None)
        d._identifiers = (self, other)
        d.operator = op
        return d

    def __invert__(self):
        """Assert dataset doesn't exist on disk."""
        obj = copy(self)
        obj.inverted = not self.inverted
        return obj

    def __or__(self, b):
        """Assert either dataset exists or another does, or both exist."""
        return self._apply_op('or', b)

    def __xor__(self, b):
        """Assert either dataset exists or another does, not both."""
        return self._apply_op('xor', b)

    def __and__(self, b):
        """Assert that a second dataset exists together with the first."""
        return self._apply_op('and', b)

    @staticmethod
    def input(name, collection, required=True, register=False, **kwargs):
        """
        Create an expected input dataset.

        By default, expected input datasets are not automatically registered.

        Parameters
        ----------
        name : str
            A dataset name or glob pattern.
        collection : str, None
            An ALF collection or pattern.
        required : bool
            Whether file must always be present, or is an optional dataset. Default is True.
        register : bool
            Whether to register the input file. Default is False for input files, True for output
            files.
        revision : str
            An optional revision.
        unique : bool
            Whether identifier pattern is expected to match a single dataset or several.

        Returns
        -------
        Input, OptionalInput
            An instance of an Input dataset if required is true, otherwise an OptionalInput.
        """
        Class = Input if required else OptionalInput
        obj = Class(name, collection, register=register, **kwargs)
        return obj

    @staticmethod
    def output(name, collection, required=True, register=True, **kwargs):
        """
        Create an expected output dataset.

        By default, expected output datasets are automatically registered.

        Parameters
        ----------
        name : str
            A dataset name or glob pattern.
        collection : str, None
            An ALF collection or pattern.
        required : bool
            Whether file must always be present, or is an optional dataset. Default is True.
        register : bool
            Whether to register the output file. Default is False for input files, True for output
            files.
        revision : str
            An optional revision.
        unique : bool
            Whether identifier pattern is expected to match a single dataset or several.

        Returns
        -------
        Output, OptionalOutput
            An instance of an Output dataset if required is true, otherwise an OptionalOutput.
        """
        Class = Output if required else OptionalOutput
        obj = Class(name, collection, register=register, **kwargs)
        return obj


class OptionalDataset(ExpectedDataset):
    """An expected dataset that is not strictly required."""

    def find_files(self, session_path, register=False):
        """Find files on disk.

        Uses glob patterns to find dataset(s) on disk.

        Parameters
        ----------
        session_path : pathlib.Path, str
            A session path within which to glob for the dataset(s).
        register : bool
            Only return files intended to be registered.

        Returns
        -------
        True
            Always True as dataset is optional.
        list of pathlib.Path
            A list of matching dataset files.
        missing, None, str, set of str
            One or more glob patterns that either didn't yield files (or did in the case of inverted datasets).

        Notes
        -----
        - Currently if `unique` is true and multiple files are found, all files are returned without an exception raised
          although this may change in the future.
        - If `register` is false, all files are returned regardless of whether they are intended to be registered.
        - If `inverted` is true, and files are found, the glob pattern is returned as missing.
        - If XOR, returns all patterns if all are present when only one should be, otherwise returns all missing
          patterns.
        - Missing (or unexpectedly found) patterns are returned despite the dataset being optional.
        """
        ok, actual_files, missing = super().find_files(session_path, register=register)
        return True, actual_files, missing

    def filter(self, session_datasets, **kwargs):
        """Filter dataset frame by expected datasets.

        Parameters
        ----------
        session_datasets : pandas.DataFrame
            An data frame of session datasets.
        kwargs
            Extra arguments for `one.util.filter_datasets`, namely revision_last_before, qc,
            ignore_qc_not_set, and assert_unique.

        Returns
        -------
        True
            Always True as dataset is optional.
        pandas.DataFrame
            A filtered data frame of containing the expected dataset(s).
        """
        ok, datasets = super().filter(session_datasets, **kwargs)
        return True, datasets


class Input(ExpectedDataset):
    """An expected input dataset."""

    pass


class OptionalInput(Input, OptionalDataset):
    """An optional expected input dataset."""

    pass


class Output(ExpectedDataset):
    """An expected output dataset."""

    pass


class OptionalOutput(Output, OptionalDataset):
    """An optional expected output dataset."""

    pass


def _parse_signature(signature):
    """
    Ensure all a signature's expected datasets are instances of ExpectedDataset.

    Parameters
    ----------
    signature : Dict[str, list]
        A dict with keys {'input_files', 'output_files'} containing lists of tuples and/or
        ExpectedDataset instances.

    Returns
    -------
    Dict[str, list of ExpectedDataset]
        A dict containing all tuples converted to ExpectedDataset instances.
    """
    I, O = ExpectedDataset.input, ExpectedDataset.output  # noqa
    inputs = [i if isinstance(i, ExpectedDataset) else I(*i) for i in signature['input_files']]
    outputs = [o if isinstance(o, ExpectedDataset) else O(*o) for o in signature['output_files']]
    return {'input_files': inputs, 'output_files': outputs}


def dataset_from_name(name, datasets):
    """
    From a list of ExpectedDataset instances, return those that match a given name.

    Parameters
    ----------
    name : str, function
        The name of the dataset or a function to match the dataset name.
    datasets : list of ExpectedDataset
        A list of ExpectedDataset instances.

    Returns
    -------
    list of ExpectedDataset
        The ExpectedDataset instances that match the given name.

    """
    matches = []
    for dataset in datasets:
        if dataset.operator is None:
            if isinstance(name, str):
                if dataset._identifiers[2] == name:
                    matches.append(dataset)
            else:
                if name(dataset._identifiers[2]):
                    matches.append(dataset)
        else:
            matches.extend(dataset_from_name(name, dataset._identifiers))
    return matches


def update_collections(dataset, new_collection, substring=None, unique=None, exact_match=False):
    """
    Update the collection of a dataset.

    This updates all nested ExpectedDataset instances with the new collection and returns copies.

    Parameters
    ----------
    dataset : ExpectedDataset
        The dataset to update.
    new_collection : str, list of str
        The new collection or collections.
    substring : str, optional
        An optional substring in the collection to replace with new collection(s). If None, the
        entire collection will be replaced.
    unique : bool, optional
        When provided, this will be used to set the `unique` attribute of the new dataset(s). If
        None, the `unique` attribute will be set to True if the collection does not contain
        wildcards.
    exact_match : bool
        If True, the collection will be replaced only if it contains `substring`.

    Returns
    -------
    ExpectedDataset
        A copy of the dataset with the updated collection(s).

    """
    after = ensure_list(new_collection)
    D = ExpectedDataset.input if isinstance(dataset, Input) else ExpectedDataset.output
    if dataset.operator is None:
        collection, revision, name = dataset.identifiers
        if revision is not None:
            raise NotImplementedError
        if substring:
            if exact_match and substring not in collection:
                after = [collection]
            else:
                after = [(collection or '').replace(substring, x) or None for x in after]
        if unique is None:
            unique = [not set(name + (x or '')).intersection('*[?') for x in after]
        else:
            unique = [unique] * len(after)
        register = dataset.register
        updated = D(name, after[0], not isinstance(dataset, OptionalDataset), register, unique=unique[0])
        if len(after) > 1:
            for folder, unq in zip(after[1:], unique[1:]):
                updated &= D(name, folder, not isinstance(dataset, OptionalDataset), register, unique=unq)
    else:
        updated = copy(dataset)
        updated._identifiers = [
            update_collections(dd, new_collection, substring, unique, exact_match) for dd in updated._identifiers
        ]
    return updated


class Transfer(abc.ABC):
    """A protocol for moving files to/from a data repository, independent of *where* a task runs.

    A :class:`DataHandler` composes one `Transfer` for downloading missing inputs and one for
    uploading/registering outputs. The same `Transfer` instance may serve both roles (e.g.
    :class:`LocalTransfer`), or different mechanisms may be mixed freely, e.g. download via S3
    but upload via Globus (see :class:`RemoteAwsDataHandler`).

    `Transfer` instances are stateless and may be shared/reused across handlers: any state
    specific to a given task run (staged file paths, scratch directories, etc.) is read from and
    written to the `handler` passed into each method, never stored on `self`.
    """

    def setUp(self, handler, **kwargs):
        """Download or otherwise stage any missing input datasets.

        Default: assume inputs are already present and do nothing.
        """
        pass

    def uploadData(self, handler, outputs, version, **kwargs):
        """Register (and, depending on the mechanism, physically move) output datasets.

        Default: compute the per-output version list expected by subclasses, without
        registering anything (used e.g. by :class:`LocalDataHandler` for local-only runs).

        Parameters
        ----------
        handler : DataHandler
            The handler this transfer is acting on behalf of.
        outputs : list of pathlib.Path
            A set of ALF paths to register to Alyx.
        version : str, list of str
            The version of ibllib used to generate these output files.

        Returns
        -------
        list of dicts, dict
            The newly created/updated Alyx dataset records, or the version list if not overridden.
        """
        if isinstance(outputs, list):
            return [version for _ in outputs]
        return [version]

    def transfer(self, handler, items):
        """Move already-registered files to their destination data repository.

        Unlike `uploadData`, this does not register anything: the Alyx records already exist and
        only the data need moving. This is used by :meth:`DataHandler.transferData` to satisfy
        each of the file records created by registration, and is therefore only implemented by
        those protocols that can push to a repository independently of registration.

        Parameters
        ----------
        handler : DataHandler
            The handler this transfer is acting on behalf of.
        items : list of (pathlib.Path, dict, dict)
            One tuple per file to move, of (local file path, Alyx file record, Alyx dataset
            record). All file records are guaranteed to be on repositories of the same store,
            although not necessarily the same repository.

        Raises
        ------
        NotImplementedError
            This protocol cannot move data independently of registration.
        """
        raise NotImplementedError(f'{self.__class__.__name__} cannot transfer already-registered files')

    def cleanUp(self, handler, **kwargs):
        """Remove any local staging artefacts created by `setUp`. Default: no-op."""
        pass


class LocalTransfer(Transfer):
    """Register outputs directly against the local repository; no bytes are moved.

    Used when the task runs on the machine that already holds the data (typically a lab's local
    acquisition server). New datasets are picked up by Alyx's nightly Globus sync.
    """

    def uploadData(self, handler, outputs, version, clobber=False, **kwargs):
        """
        Upload and/or register output data.

        Parameters
        ----------
        handler : DataHandler
            The handler this transfer is acting on behalf of.
        outputs : list of pathlib.Path
            A set of ALF paths to register to Alyx.
        version : str, list of str
            The version of ibllib used to generate these output files.
        clobber : bool
            If True, re-upload outputs that have already been passed to this method.
        kwargs
            Optional keyword arguments for one.registration.RegistrationClient.register_files.

        Returns
        -------
        list of dicts, dict
            A list of newly created Alyx dataset records or the registration data if dry.
        """
        versions = super().uploadData(handler, outputs, version)
        data_repo = get_local_data_repository(handler.one.alyx)
        # If clobber = False, do not re-upload the outputs that have already been processed
        outputs = ensure_list(outputs)
        to_upload = list(filter(None if clobber else lambda x: x not in handler.processed, outputs))
        records = register_dataset(to_upload, one=handler.one, versions=versions, repository=data_repo, **kwargs) or []
        if kwargs.get('dry', False):
            return records
        # Store processed outputs
        handler.processed.update({k: v for k, v in zip(to_upload, records) if v})
        return [handler.processed[x] for x in outputs if x in handler.processed]


class GlobusTransfer(Transfer):
    """Move files to/from a Globus data repository, e.g. one of the flatiron endpoints."""

    @staticmethod
    def _globus_client(handler, repository):
        """Build a Globus client with the local and `repository` endpoints registered.

        Parameters
        ----------
        handler : DataHandler
            The handler this transfer is acting on behalf of.
        repository : str
            The name of the Globus data repository to register, e.g. 'flatiron_cortexlab'.

        Returns
        -------
        one.remote.globus.Globus
            A Globus client instance.
        """
        from one.remote.globus import Globus  # noqa

        globus = Globus(client_name='server', headless=True)
        # on local servers set up the local root path manually as some have different globus config paths
        globus.endpoints['local']['root_path'] = '/mnt/s0/Data/Subjects'
        # For cortex lab we need to get the endpoint from the ibl alyx
        if 'cortexlab' in repository and 'cortexlab' in handler.one.alyx.base_url:
            alyx = AlyxClient(base_url='https://alyx.internationalbrainlab.org', cache_rest=None)
        else:
            alyx = handler.one.alyx
        globus.add_endpoint(repository, alyx=alyx)
        return globus

    def setUp(self, handler, **_):
        """Download any missing input datasets from flatiron using globus-sdk."""
        lab = get_lab(handler.session_path, handler.one.alyx)
        repository = f'flatiron_{lab}'
        globus = self._globus_client(handler, repository)
        if lab == 'cortexlab' and 'cortexlab' in handler.one.alyx.base_url:
            one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_rest=handler.one.alyx.cache_mode)
            df = handler.getData(one=one)
        else:
            df = handler.getData()

        if len(df) == 0:
            # If no datasets found in the cache only work off local file system do not attempt to
            # download any missing data using Globus
            return

        # Check for space on local server. If less that 500 GB don't download new data
        space_free = shutil.disk_usage(globus.endpoints['local']['root_path'])[2]
        if space_free < 500e9:
            _logger.warning("Space left on server is < 500GB, won't re-download new data")
            return

        rel_sess_path = handler.session_path.session_path_short()
        target_paths = []
        source_paths = []
        handler.local_paths = []
        for i, d in df.iterrows():
            sess_path = Path(rel_sess_path).joinpath(d['rel_path'])
            full_local_path = Path(globus.endpoints['local']['root_path']).joinpath(sess_path)
            if not full_local_path.exists():
                uuid = i
                handler.local_paths.append(full_local_path)
                target_paths.append(sess_path)
                source_paths.append(add_uuid_string(sess_path, uuid))

        if len(target_paths) != 0:
            ts = time()
            for sp, tp in zip(source_paths, target_paths):
                _logger.info(f'Downloading {sp} to {tp}')
            globus.mv(repository, 'local', source_paths, target_paths)
            _logger.debug(f'Complete. Time elapsed {time() - ts}')

    def uploadData(self, handler, outputs, version, **kwargs):
        """
        Register output datasets on the server repositories and transfer them to flatiron.

        Parameters
        ----------
        handler : DataHandler
            The handler this transfer is acting on behalf of.
        outputs : list of pathlib.Path
            A set of ALF paths to register to Alyx.
        version : str, list of str
            The version of ibllib used to generate these output files.

        Returns
        -------
        list of dicts, dict
            The newly created/updated Alyx dataset records.
        """
        versions = super().uploadData(handler, outputs, version)
        response = register_dataset(outputs, one=handler.one, server_only=True, versions=versions, **kwargs)
        if kwargs.get('dry', False):
            return response
        items = []
        for dset, out in zip(ensure_list(response), ensure_list(outputs)):
            assert Path(out).name == dset['name']
            fr = next(fr for fr in dset['file_records'] if repository_store(fr['data_repository']) == 'flatiron')
            items.append((out, fr, dset))
        self.transfer(handler, items)
        return response

    def transfer(self, handler, items):
        """Transfer registered files over Globus, verifying each before flagging it as existing.

        See Also
        --------
        Transfer.transfer
        """
        by_repository = defaultdict(list)
        for path, fr, dset in items:
            by_repository[fr['data_repository']].append((path, fr, dset))

        for repository, group in by_repository.items():
            globus = self._globus_client(handler, repository)
            source_paths, target_paths, collections = [], [], {}
            for path, fr, dset in group:
                collection = '/'.join(fr['relative_path'].split('/')[:-1])
                details = {dset['name']: {'fr_id': fr['id'], 'size': dset['file_size']}}
                collections.setdefault(collection, {}).update(details)
                # Set exists status to false until the transfer is verified below
                handler.one.alyx.rest('files', 'partial_update', id=fr['id'], data={'exists': False})
                source_paths.append(path)
                target_paths.append(add_uuid_string(fr['relative_path'], dset['id']))

            if len(target_paths) != 0:
                ts = time()
                for sp, tp in zip(source_paths, target_paths):
                    _logger.info(f'Uploading {sp} to {tp}')
                globus.mv('local', repository, source_paths, target_paths)
                _logger.debug(f'Complete. Time elapsed {time() - ts}')

            for collection, files in collections.items():
                globus_files = globus.ls(repository, collection, remove_uuid=True, return_size=True)
                file_names = [str(gl[0]) for gl in globus_files]
                file_sizes = [gl[1] for gl in globus_files]

                for name, details in files.items():
                    try:
                        idx = file_names.index(name)
                        size = file_sizes[idx]
                        if size == details['size']:
                            # update the file record if sizes match
                            handler.one.alyx.rest('files', 'partial_update', id=details['fr_id'], data={'exists': True})
                        else:
                            _logger.warning(f'File {name} found on {repository} but sizes do not match')
                    except ValueError:
                        _logger.warning(f'File {name} not found on {repository}')

    def cleanUp(self, handler, **_):
        """Clean up, remove the files that were downloaded from Globus once task has completed."""
        for file in getattr(handler, 'local_paths', []):
            os.unlink(file)


class HTTPTransfer(Transfer):
    """Download missing input datasets via HTTP, using ONE's own web client."""

    def setUp(self, handler, check_hash=True, **_):
        """Function to download necessary data to run tasks using ONE."""
        df = handler.getData()
        handler.one._check_filesystem(df, check_hash=check_hash)


class FTPTransfer(Transfer):
    """Register and upload output datasets via FTP, to the DMZ repository."""

    def uploadData(self, handler, outputs, version, **kwargs):
        """Function to upload and register data of completed task via FTP patcher."""
        versions = super().uploadData(handler, outputs, version)
        ftp_patcher = FTPPatcher(one=handler.one)
        return ftp_patcher.create_dataset(path=outputs, created_by=handler.one.alyx.user, versions=versions, **kwargs)


class S3Transfer(Transfer):
    """Download input datasets from, and/or upload output datasets to, the IBL S3 bucket."""

    def setUp(self, handler, **_):
        """Function to download necessary data to run tasks using AWS boto3."""
        df = handler.getData()
        handler.local_paths = handler.one._download_aws(map(lambda x: x[1], df.iterrows()))

    def uploadData(self, handler, outputs, version, **kwargs):
        """Function to upload and register data of completed task via S3 patcher."""
        versions = super().uploadData(handler, outputs, version)
        s3_patcher = S3Patcher(one=handler.one)
        return s3_patcher.patch_dataset(outputs, created_by=handler.one.alyx.user, versions=versions, **kwargs)

    def transfer(self, handler, items):
        """Upload registered files to their S3 bucket, then flag them as existing.

        See Also
        --------
        Transfer.transfer
        """
        from one.remote.aws import get_s3_from_alyx  # noqa

        by_repository = defaultdict(list)
        for path, fr, dset in items:
            by_repository[fr['data_repository']].append((path, fr, dset))

        for repository, group in by_repository.items():
            s3, bucket = get_s3_from_alyx(handler.one.alyx, repo_name=repository)
            for path, fr, dset in group:
                # The bucket key is the repository path joined with the file record's relative
                # path, with the dataset UUID added, i.e. the same key ONE resolves on download.
                key = PurePosixPath(fr['data_repository_path'], fr['relative_path'])
                key = add_uuid_string(key, dset['id']).as_posix().lstrip('/')
                _logger.info('Uploading %s to s3://%s/%s', path, bucket, key)
                s3.Bucket(bucket).upload_file(str(path), key)
                handler.one.alyx.rest('files', 'partial_update', id=fr['id'], data={'exists': True})

    def cleanUp(self, handler, task=None, **_):
        """Clean up, remove the files downloaded from S3, but only once the task has completed."""
        if task is not None and task.status == 0:
            for file in getattr(handler, 'local_paths', []):
                os.unlink(file)


class DataHandler(abc.ABC):
    """Map of str to Transfer: the protocol to use to reach each data repository store.

    Keys are the store part of a data repository name (see :func:`repository_store`), e.g.
    'flatiron' for the 'flatiron_cortexlab' repository. This is declared per location, as not
    every location is set up to reach every store, e.g. a machine may not hold S3 credentials.
    Used by :meth:`transferData` to push data to each repository registration created a file
    record for. An empty map means this handler never transfers data itself.
    """

    protocols = {}

    def __init__(self, session_path, signature, one=None, download=None, upload=None):
        """
        Base data handler class.

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        :param download: a :class:`Transfer` instance used to stage missing input datasets.
            Defaults to a no-op transfer that assumes inputs are already present.
        :param upload: a :class:`Transfer` instance used to register/upload output datasets.
            Defaults to a no-op transfer that registers nothing.
        """
        self.session_path = ensure_alf_path(session_path)
        self.signature = _parse_signature(signature)
        self.one = one
        self.processed = {}  # Map of filepaths and their processed records (e.g. upload receipts or Alyx records)
        self.download = download or Transfer()
        self.upload = upload or Transfer()

    def setUp(self, **kwargs):
        """Download/stage any missing input datasets required to run the task."""
        return self.download.setUp(self, **kwargs)

    def getData(self, one=None):
        """Finds the datasets required for task based on input signatures.

        Parameters
        ----------
        one : one.api.One, optional
            An instance of ONE to use.

        Returns
        -------
        pandas.DataFrame, None
            A data frame of required datasets. An empty frame is returned if no registered datasets are required,
            while None is returned if no instance of ONE is set.
        """
        if self.one is None and one is None:
            return
        one = one or self.one
        session_datasets = one.list_datasets(one.path2eid(self.session_path), details=True)
        dfs = [file.filter(session_datasets)[1] for file in self.signature['input_files']]
        return one._cache.datasets.iloc[0:0] if len(dfs) == 0 else pd.concat(dfs).drop_duplicates()

    def getOutputFiles(self, session_path=None):
        """
        Return a data frame of output datasets found on disk.

        Returns
        -------
        pandas.DataFrame
            A dataset data frame of datasets on disk that were specified in signature['output_files'].
        """
        session_path = self.session_path if session_path is None else session_path
        assert session_path
        # Next convert datasets to frame
        # Create dataframe of all ALF datasets
        df = _make_datasets_df(session_path, hash_files=False).set_index(['eid', 'id'])
        # Filter outputs
        if len(self.signature['output_files']) == 0:
            return pd.DataFrame()
        present = [file.filter(df)[1] for file in self.signature['output_files']]
        return pd.concat(present).droplevel('eid')

    def uploadData(self, outputs, version, **kwargs):
        """
        Upload and/or register output data.

        This is typically called by :meth:`ibllib.pipes.tasks.Task.register_datasets`. The
        actual work is delegated to this handler's `upload` Transfer instance.

        Parameters
        ----------
        outputs : list of pathlib.Path
            A set of ALF paths to register to Alyx.
        version : str, list of str
            The version of ibllib used to generate these output files.
        kwargs
            Optional keyword arguments passed through to the `upload` Transfer.

        Returns
        -------
        list of dicts, dict
            A list of newly created Alyx dataset records or the registration data if dry.
        """
        return self.upload.uploadData(self, outputs, version, **kwargs)

    def transferData(self, datasets):
        """
        Transfer registered datasets to each repository that is awaiting the data.

        The destination repositories are taken from the file records returned by the registration
        endpoint: Alyx creates one per repository associated with the session's lab, and those
        that do not yet hold the data have exists=False. Each is transferred with the protocol
        this handler declares for its store in :attr:`protocols`; repositories this handler has no
        protocol for are skipped, as another process is expected to handle them (e.g. Alyx's
        nightly Globus transfer).

        Parameters
        ----------
        datasets : dict
            A map of local file path to its Alyx dataset record, as returned by registration.

        Returns
        -------
        dict
            A map of store name to the list of file records transferred to that store.
        """
        pending = defaultdict(list)
        for path, dset in datasets.items():
            for fr in filter(lambda x: not x['exists'], dset['file_records']):
                store = repository_store(fr['data_repository'])
                if store in self.protocols:
                    pending[store].append((path, fr, dset))
                else:
                    _logger.debug(
                        'No transfer protocol for repository "%s"; skipping %s', fr['data_repository'], fr['relative_path']
                    )
        for store, items in pending.items():
            self.protocols[store].transfer(self, items)
        return {store: [fr for _, fr, _ in items] for store, items in pending.items()}

    def cleanUp(self, **kwargs):
        """Clean up any local files staged by `setUp`."""
        return self.download.cleanUp(self, **kwargs)


class LocalDataHandler(DataHandler):
    def __init__(self, session_path, signatures, one=None):
        """
        Data handler for running tasks locally, with no architecture or db connection
        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signatures, one=one)


class ServerDataHandler(DataHandler):
    """Data handler for running tasks on lab local servers when all data is available locally.

    Output datasets are registered against the server's own data repository. When `mode` is
    'delayed' (the default) the data are left for Alyx to transfer: registration creates a file
    record with exists=False on each of the lab's other repositories, Alyx's nightly Globus job
    moves the data to flatiron, then a separate cron syncs flatiron to the S3 mirror. When `mode`
    is 'immediate' this handler instead pushes the data to each of those repositories itself, as
    soon as they are registered.
    """

    protocols = {'flatiron': GlobusTransfer(), 'aws': S3Transfer()}

    def __init__(self, session_path, signatures, one=None, mode='delayed'):
        """
        Data handler for running tasks on lab local servers when all data is available locally

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        :param mode: whether to leave the transfer to Alyx ('delayed', the default) or to transfer
            the data to the other repositories immediately after registering them ('immediate').
        """
        if mode not in ('delayed', 'immediate'):
            raise ValueError(f'Unknown mode "{mode}"')
        super().__init__(session_path, signatures, one=one, upload=LocalTransfer())
        self.mode = mode

    def uploadData(self, outputs, version, **kwargs):
        """
        Register output data, and transfer it if this handler's mode is 'immediate'.

        See Also
        --------
        DataHandler.uploadData
        DataHandler.transferData
        """
        records = super().uploadData(outputs, version, **kwargs)
        if self.mode == 'immediate' and not kwargs.get('dry', False):
            # NB: `processed` maps each local file path to its dataset record, so unlike the
            # returned records it is guaranteed to pair each file with its own registration.
            self.transferData({k: v for k, v in self.processed.items() if k in ensure_list(outputs)})
        return records

    def cleanUp(self, **_):
        """Empties and returns the processed dataset mep."""
        super().cleanUp()
        processed = self.processed
        self.processed = {}
        return processed


class ServerGlobusDataHandler(ServerDataHandler):
    def __init__(self, session_path, signatures, one=None, mode='delayed'):
        """
        Data handler for running tasks on lab local servers. Will download missing data from SDSC using Globus

        :param session_path: path to session
        :param signatures: input and output file signatures
        :param one: ONE instance
        :param mode: whether to leave the upload transfer to Alyx ('delayed', the default) or to
            transfer the data to the other repositories immediately after registering them.
        """
        super().__init__(session_path, signatures, one=one, mode=mode)
        self.download = GlobusTransfer()


class RemoteEC2DataHandler(DataHandler):
    def __init__(self, session_path, signature, one=None):
        """
        Data handler for running tasks on remote compute node. Downloads missing data via HTTP
        using ONE, and uploads output data via the S3 patcher.

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signature, one=one, download=HTTPTransfer(), upload=S3Transfer())


class RemoteHttpDataHandler(DataHandler):
    def __init__(self, session_path, signature, one=None):
        """
        Data handler for running tasks on remote compute node. Will download missing data via http using ONE

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signature, one=one, download=HTTPTransfer(), upload=FTPTransfer())


class RemoteAwsDataHandler(DataHandler):
    def __init__(self, session_path, signature, one=None):
        """
        Data handler for running tasks on remote compute node.

        This will download missing data from the private IBL S3 AWS data bucket.  New datasets are
        uploaded via Globus, immediately (rather than waiting for the nightly transfer), since this
        compute node's local repository is not otherwise synced by Alyx's nightly job.

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signature, one=one, download=S3Transfer(), upload=GlobusTransfer())


class RemoteGlobusDataHandler(DataHandler):
    """
    Data handler for running tasks on remote compute node. Will download missing data using Globus.

    :param session_path: path to session
    :param signature: input and output file signatures
    :param one: ONE instance
    """

    def __init__(self, session_path, signature, one=None):
        # NB: downloading via Globus here is not yet implemented (matches previous behaviour)
        super().__init__(session_path, signature, one=one, upload=FTPTransfer())


class SDSCDataHandler(DataHandler):
    """
    Data handler for running tasks on SDSC compute node

    :param session_path: path to session
    :param signature: input and output file signatures
    :param one: ONE instance
    """

    def __init__(self, session_path, signatures, one=None):
        super().__init__(session_path, signatures, one=one)
        self.patch_path = os.getenv('SDSC_PATCH_PATH', SDSC_PATCH_PATH)
        self.root_path = SDSC_ROOT_PATH
        self.linked_files = []  # List of symlinks created to run tasks

    def setUp(self, task, **_):
        """Function to create symlinks to necessary data to run tasks."""
        df = self.getData()

        SDSC_TMP = ensure_alf_path(self.patch_path.joinpath(task.__class__.__name__))
        session_path = Path(get_alf_path(self.session_path))
        for uuid, d in df.iterrows():
            file_path = session_path / d['rel_path']
            file_uuid = add_uuid_string(file_path, uuid)
            file_link = Path(SDSC_TMP.joinpath(file_path))
            file_link.parent.mkdir(exist_ok=True, parents=True)
            try:  # TODO append link to task attribute
                file_link.symlink_to(Path(self.root_path.joinpath(file_uuid)))
                self.linked_files.append(file_link)
            except FileExistsError:
                pass
        task.session_path = Path(SDSC_TMP.joinpath(session_path))
        # If one of the symlinked input files is also an expected output, raise here to avoid overwriting
        # In the future we may instead copy the data under this condition
        assert self.getOutputFiles(session_path=task.session_path).shape[0] == 0, (
            'On SDSC patcher, output files should be distinct from input files to avoid overwriting'
        )

    def uploadData(self, outputs, version, **kwargs):
        """Function to upload and register data of completed task via SDSC patcher."""
        versions = super().uploadData(outputs, version)
        sdsc_patcher = SDSCPatcher(one=self.one)
        return sdsc_patcher.patch_datasets(outputs, dry=False, versions=versions, **kwargs)

    def cleanUp(self, task):
        """Function to clean up symlinks created to run task."""
        assert self.patch_path.parts[0:4] == task.session_path.parts[0:4]
        shutil.rmtree(task.session_path)


class PopeyeDataHandler(SDSCDataHandler):
    def __init__(self, session_path, signatures, one=None):
        super().__init__(session_path, signatures, one=one)
        self.patch_path = Path(os.getenv('SDSC_PATCH_PATH', '/mnt/sdceph/users/ibl/data/quarantine/tasks/'))
        self.root_path = Path('/mnt/sdceph/users/ibl/data')

    def uploadData(self, outputs, version, **kwargs):
        raise NotImplementedError(
            'Cannot register data from Popeye. Login as Datauser and use the RegisterSpikeSortingSDSC task.'
        )

    def cleanUp(self, **_):
        """Symlinks are preserved until registration."""
        pass

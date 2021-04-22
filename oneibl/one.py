import abc
import concurrent.futures
import json
import logging
import os
import fnmatch
import re
from functools import wraps
from pathlib import Path, PurePath
import shutil
from typing import Any, Sequence, Union, Optional, List, Dict
from uuid import UUID

import requests
import tqdm
import pandas as pd
import numpy as np

import oneibl.params
import oneibl.webclient as wc
import alf.io as alfio
from alf.files import is_valid, alf_parts
# from ibllib.misc.exp_ref import is_exp_ref
from ibllib.exceptions import \
    ALFMultipleObjectsFound, ALFObjectNotFound, ALFMultipleCollectionsFound
from ibllib.io import hashfile, spikeglx
from ibllib.misc import pprint
from oneibl.dataclass import SessionDataInfo
from brainbox.io import parquet
from brainbox.numerical import ismember, ismember2d, find_first_2d

_logger = logging.getLogger('ibllib')


def Listable(t): return Union[t, Sequence[t]]  # noqa


NTHREADS = 4  # number of download threads

_ENDPOINTS = {  # keynames are possible input arguments and values are actual endpoints
    'data': 'dataset-types',
    'dataset': 'datasets',
    'datasets': 'datasets',
    'dataset-types': 'dataset-types',
    'dataset_types': 'dataset-types',
    'dataset-type': 'dataset-types',
    'dataset_type': 'dataset-types',
    'dtypes': 'dataset-types',
    'dtype': 'dataset-types',
    'users': 'users',
    'user': 'users',
    'subject': 'subjects',
    'subjects': 'subjects',
    'labs': 'labs',
    'lab': 'labs'}

_SESSION_FIELDS = {  # keynames are possible input arguments and values are actual fields
    'subjects': 'subject',
    'subject': 'subject',
    'user': 'users',
    'users': 'users',
    'lab': 'lab',
    'labs': 'lab',
    'type': 'type',
    'start_time': 'start_time',
    'start-time': 'start_time',
    'end_time': 'end_time',
    'end-time': 'end_time'}

SEARCH_TERMS = {  # keynames are possible input arguments and values are actual fields
    'data': 'dataset_types',
    'dataset': 'dataset_types',
    'datasets': 'dataset_types',
    'dataset-types': 'dataset_types',
    'dataset_types': 'dataset_types',
    'users': 'users',
    'user': 'users',
    'subject': 'subject',
    'subjects': 'subject',
    'date_range': 'date_range',
    'date-range': 'date_range',
    'date': 'date_range',
    'labs': 'lab',
    'lab': 'lab',
    'task': 'task_protocol',
    'task_protocol': 'task_protocol',
    'number': 'number',
    'location': 'location',
    'lab_location': 'location',
    'performance_lte': 'performance_lte',
    'performance_gte': 'performance_gte',
    'project': 'project',
}


def _ses2pandas(ses, dtypes=None):
    """
    :param ses: session dictionary from rest endpoint
    :param dtypes: list of dataset types
    :return:
    """
    # selection: get relevant dtypes only if there is an url associated
    rec = list(filter(lambda x: x['url'], ses['data_dataset_session_related']))
    if dtypes == ['__all__'] or dtypes == '__all__':
        dtypes = None
    if dtypes is not None:
        rec = list(filter(lambda x: x['dataset_type'] in dtypes, rec))
    include = ['id', 'hash', 'dataset_type', 'name', 'file_size', 'collection']
    uuid_fields = ['id', 'eid']
    join = {'subject': ses['subject'], 'lab': ses['lab'], 'eid': ses['url'][-36:],
            'start_time': np.datetime64(ses['start_time']), 'number': ses['number'],
            'task_protocol': ses['task_protocol']}
    col = parquet.rec2col(rec, include=include, uuid_fields=uuid_fields, join=join,
                          types={'file_size': np.double}).to_df()
    return col


def parse_id(method):
    """
    Ensures the input experiment identifier is an experiment UUID string
    :param method: An ONE method whose second arg is an experiment id
    :return: A wrapper function that parses the id to the expected string
    """
    @wraps(method)
    def wrapper(self, id, *args, **kwargs):
        id = self.to_eid(id)
        return method(self, id, *args, **kwargs)
    return wrapper


class OneAbstract(abc.ABC):

    def __init__(self, username=None, password=None, base_url=None, cache_dir=None, silent=None):
        # get parameters override if inputs provided
        self._par = oneibl.params.get(silent=silent)
        # can delete those 2 lines from mid January 2021
        if self._par.HTTP_DATA_SERVER == 'http://ibl.flatironinstitute.org':
            self._par = self._par.set("HTTP_DATA_SERVER", "https://ibl.flatironinstitute.org")
        self._par = self._par.set('ALYX_LOGIN', username or self._par.ALYX_LOGIN)
        self._par = self._par.set('ALYX_URL', base_url or self._par.ALYX_URL)
        self._par = self._par.set('ALYX_PWD', password or self._par.ALYX_PWD)
        self._par = self._par.set('CACHE_DIR', cache_dir or self._par.CACHE_DIR)
        # init the cache file
        self._cache_file = Path(self._par.CACHE_DIR).joinpath('.one_cache.parquet')
        if self._cache_file.exists():
            # we need to keep this part fast enough for transient objects
            self._cache = parquet.load(self._cache_file)
        else:
            self._cache = pd.DataFrame()

    def _load(self, eid, dataset_types=None, dclass_output=False, download_only=False,
              offline=False, **kwargs):
        """
        From a Session ID and dataset types, queries Alyx database, downloads the data
        from Globus, and loads into numpy array. Single session only
        """
        if alfio.is_uuid_string(eid):
            eid = '/sessions/' + eid
        eid_str = eid[-36:]
        # if no dataset_type is provided:
        # a) force the output to be a dictionary that provides context to the data
        # b) download all types that have a data url specified whithin the alf folder
        dataset_types = [dataset_types] if isinstance(dataset_types, str) else dataset_types
        if not dataset_types or dataset_types == ['__all__']:
            dclass_output = True
        if offline:
            dc = self._make_dataclass_offline(eid_str, dataset_types, **kwargs)
        else:
            dc = self._make_dataclass(eid_str, dataset_types, **kwargs)
        # load the files content in variables if requested
        if not download_only:
            for ind, fil in enumerate(dc.local_path):
                dc.data[ind] = alfio.load_file_content(fil)
        # parse output arguments
        if dclass_output:
            return dc
        # if required, parse the output as a list that matches dataset_types requested
        list_out = []
        for dt in dataset_types:
            if dt not in dc.dataset_type:
                _logger.warning('dataset ' + dt + ' not found for session: ' + eid_str)
                list_out.append(None)
                continue
            for i, x, in enumerate(dc.dataset_type):
                if dt == x:
                    if dc.data[i] is not None:
                        list_out.append(dc.data[i])
                    else:
                        list_out.append(dc.local_path[i])
        return list_out

    def _get_cache_dir(self, cache_dir):
        if not cache_dir:
            cache_dir = self._par.CACHE_DIR
        # if empty in parameter file, do not allow and set default
        if not cache_dir:
            cache_dir = str(PurePath(Path.home(), "Downloads", "FlatIron"))
        assert cache_dir
        return cache_dir

    def _make_dataclass_offline(self, eid, dataset_types=None, cache_dir=None, **kwargs):
        if self._cache.size == 0:
            return SessionDataInfo()
        # select the session
        npeid = parquet.str2np(eid)[0]
        df = self._cache[self._cache['eid_0'] == npeid[0]]
        df = df[df['eid_1'] == npeid[1]]
        # select datasets
        df = df[ismember(df['dataset_type'], dataset_types)[0]]
        return SessionDataInfo.from_pandas(df, self._get_cache_dir(cache_dir))

    def path_from_eid(self, eid: str) -> Optional[Listable(Path)]:
        """
        From an experiment id or a list of experiment ids, gets the local cache path
        :param eid: eid (UUID) or list of UUIDs
        :return: eid or list of eids
        """
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            path_list = []
            for p in eid:
                path_list.append(self.path_from_eid(p))
            return path_list
        # If not valid return None
        if not alfio.is_uuid_string(eid):
            print(eid, " is not a valid eID/UUID string")
            return
        if self._cache.size == 0:
            return

        # load path from cache
        ic = find_first_2d(
            self._cache[['eid_0', 'eid_1']].to_numpy(), parquet.str2np(eid))
        if ic is not None:
            ses = self._cache.iloc[ic]
            return Path(self._par.CACHE_DIR).joinpath(
                ses['lab'], 'Subjects', ses['subject'], ses['start_time'].isoformat()[:10],
                str(ses['number']).zfill(3))

    def eid_from_path(self, path_obj):
        """
        From a local path, gets the experiment id
        :param path_obj: local path or list of local paths
        :return: eid or list of eids
        """
        # If path_obj is a list recurse through it and return a list
        if isinstance(path_obj, list):
            path_obj = [Path(x) for x in path_obj]
            eid_list = []
            for p in path_obj:
                eid_list.append(self.eid_from_path(p))
            return eid_list
        # else ensure the path ends with mouse,date, number
        path_obj = Path(path_obj)
        session_path = alfio.get_session_path(path_obj)
        # if path does not have a date and a number, or cache is empty return None
        if session_path is None or self._cache.size == 0:
            return None

        # fetch eid from cache
        ind = ((self._cache['subject'] == session_path.parts[-3]) &
               (self._cache['start_time'].apply(
                   lambda x: x.isoformat()[:10] == session_path.parts[-2])) &
               (self._cache['number']) == int(session_path.parts[-1]))
        ind = np.where(ind.to_numpy())[0]
        if ind.size > 0:
            return parquet.np2str(self._cache[['eid_0', 'eid_1']].iloc[ind[0]])

    @abc.abstractmethod
    def _make_dataclass(self, eid, dataset_types=None, cache_dir=None, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, **kwargs):
        pass

    @abc.abstractmethod
    def list(self, **kwargs):
        pass

    @abc.abstractmethod
    def search(self, **kwargs):
        pass


def ONE(offline=False, **kwargs):
    if offline:
        return OneOffline(**kwargs)
    else:
        return OneAlyx(**kwargs)


class OneOffline(OneAbstract):

    def _make_dataclass(self, *args, **kwargs):
        return self._make_dataclass_offline(*args, **kwargs)

    def load(self, eid, **kwargs):
        return self._load(eid, **kwargs)

    def list(self, **kwargs):
        pass

    def search(self, **kwargs):
        pass


class OneAlyx(OneAbstract):
    def __init__(self, **kwargs):
        # get parameters override if inputs provided
        super(OneAlyx, self).__init__(**kwargs)
        try:
            self._alyxClient = wc.AlyxClient(username=self._par.ALYX_LOGIN,
                                             password=self._par.ALYX_PWD,
                                             base_url=self._par.ALYX_URL)
            # Display output when instantiating ONE
            print(f"Connected to {self._par.ALYX_URL} as {self._par.ALYX_LOGIN}", )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Can't connect to {self._par.ALYX_URL}.\n" +
                "IP addresses are filtered on IBL database servers. \n" +
                "Are you connecting from an IBL participating institution ?"
            )

    @property
    def alyx(self):
        return self._alyxClient

    def help(self, dataset_type=None):
        if not dataset_type:
            return self.alyx.rest('dataset-types', 'list')
        if isinstance(dataset_type, list):
            for dt in dataset_type:
                self.help(dataset_type=dt)
                return
        if not isinstance(dataset_type, str):
            print('No dataset_type provided or wrong type. Should be str')
            return
        out = self.alyx.rest('dataset-types', 'read', dataset_type)
        print(out['description'])

    def list(self, eid: Optional[Union[str, Path, UUID]] = None, details=False
             ) -> Union[List, Dict[str, str]]:
        """
        From a Session ID, queries Alyx database for datasets related to a session.

        :param eid: Experiment session uuid str
        :type eid: str

        :param details: If false returns a list of path, otherwise returns the REST dictionary
        :type eid: bool

        :return: list of strings or dict of lists if details is True
        :rtype:  list, dict
        """
        if not eid:
            return [x['name'] for x in self.alyx.rest('dataset-types', 'list')]

        # Session specific list
        dsets = self.alyx.rest('datasets', 'list', session=eid, exists=True)
        if not details:
            dsets = sorted([Path(dset['collection']).joinpath(dset['name']) for dset in dsets])
        return dsets

    @parse_id
    def load(self, eid, dataset_types=None, dclass_output=False, dry_run=False, cache_dir=None,
             download_only=False, clobber=False, offline=False, keep_uuid=False):
        """
        From a Session ID and dataset types, queries Alyx database, downloads the data
        from Globus, and loads into numpy array.

        :param eid: Experiment ID, for IBL this is the UUID of the Session as per Alyx
         database. Could be a full Alyx URL:
         'http://localhost:8000/sessions/698361f6-b7d0-447d-a25d-42afdef7a0da' or only the UUID:
         '698361f6-b7d0-447d-a25d-42afdef7a0da'. Can also be a list of the above for multiple eids.
        :type eid: str
        :param dataset_types: [None]: Alyx dataset types to be returned.
        :type dataset_types: list
        :param dclass_output: [False]: forces the output as dataclass to provide context.
        :type dclass_output: bool
         If None or an empty dataset_type is specified, the output will be a dictionary by default.
        :param cache_dir: temporarly overrides the cache_dir from the parameter file
        :type cache_dir: str
        :param download_only: do not attempt to load data in memory, just download the files
        :type download_only: bool
        :param clobber: force downloading even if files exists locally
        :type clobber: bool
        :param keep_uuid: keeps the UUID at the end of the filename (defaults to False)
        :type keep_uuid: bool

        :return: List of numpy arrays matching the size of dataset_types parameter, OR
         a dataclass containing arrays and context data.
        :rtype: list, dict, dataclass SessionDataInfo
        """
        # this is a wrapping function to keep signature and docstring accessible for IDE's
        return self._load_recursive(eid, dataset_types=dataset_types, dclass_output=dclass_output,
                                    dry_run=dry_run, cache_dir=cache_dir, keep_uuid=keep_uuid,
                                    download_only=download_only, clobber=clobber, offline=offline)

    @parse_id
    def load_dataset(self,
                     eid: Union[str, Path, UUID],
                     dataset: str,
                     collection: Optional[str] = None,
                     download_only: bool = False) -> Any:
        """
        Load a single dataset from a Session ID and a dataset type.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param dataset: The ALF dataset to load.  Supports asterisks as wildcards.
        :param collection:  The collection to which the object belongs, e.g. 'alf/probe01'.
        For IBL this is the relative path of the file from the session root.
        Supports asterisks as wildcards.
        :param download_only: When true the data are downloaded and the file path is returned
        :return: dataset or a Path object if download_only is true

        Examples:
            intervals = one.load_dataset(eid, '_ibl_trials.intervals.npy')
            intervals = one.load_dataset(eid, '*trials.intervals*')
            filepath = one.load_dataset(eid '_ibl_trials.intervals.npy', download_only=True)
            spikes = one.load_dataset(eid 'spikes.times.npy', collection='alf/probe01')
        """
        search_str = 'name__regex,' + dataset.replace('.', r'\.').replace('*', '.*')
        if collection:
            search_str += ',collection__regex,' + collection.replace('*', '.*')
        results = self.alyx.rest('datasets', 'list', session=eid, django=search_str, exists=True)

        # Get filenames of returned ALF files
        collection_set = {x['collection'] for x in results}
        if len(collection_set) > 1:
            raise ALFMultipleCollectionsFound('Matching dataset belongs to multiple collections:' +
                                              ', '.join(collection_set))
        if len(results) > 1:
            raise ALFMultipleObjectsFound('The following matching datasets were found: ' +
                                          ', '.join(x['name'] for x in results))
        if len(results) == 0:
            raise ALFObjectNotFound(f'Dataset "{dataset}" not found on Alyx')

        filename = self.download_dataset(results[0])
        assert filename is not None, 'failed to download dataset'

        return filename if download_only else alfio.load_file_content(filename)

    @parse_id
    def load_object(self,
                    eid: Union[str, Path, UUID],
                    obj: str,
                    collection: Optional[str] = 'alf',
                    download_only: bool = False,
                    **kwargs) -> Union[alfio.AlfBunch, List[Path]]:
        """
        Load all attributes of an ALF object from a Session ID and an object name.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param obj: The ALF object to load.  Supports asterisks as wildcards.
        :param collection:  The collection to which the object belongs, e.g. 'alf/probe01'.
        Supports asterisks as wildcards.
        :param download_only: When true the data are downloaded and the file paths are returned
        :param kwargs: Optional filters for the ALF objects, including namespace and timescale
        :return: An ALF bunch or if download_only is True, a list of Paths objects

        Examples:
        load_object(eid, '*moves')
        load_object(eid, 'trials')
        load_object(eid, 'spikes', collection='*probe01')
        """
        # Filter server-side by collection and dataset name
        search_str = 'name__regex,' + obj.replace('*', '.*')
        if collection and collection != 'all':
            search_str += ',collection__regex,' + collection.replace('*', '.*')
        results = self.alyx.rest('datasets', 'list', exists=True, session=eid, django=search_str)
        pattern = re.compile(fnmatch.translate(obj))

        # Further refine by matching object part of ALF datasets
        def match(r):
            return is_valid(r['name']) and pattern.match(alf_parts(r['name'])[1])

        # Get filenames of returned ALF files
        returned_obj = {alf_parts(x['name'])[1] for x in results if match(x)}

        # Validate result before loading
        if len(returned_obj) > 1:
            raise ALFMultipleObjectsFound('The following matching objects were found: ' +
                                          ', '.join(returned_obj))
        elif len(returned_obj) == 0:
            raise ALFObjectNotFound(f'ALF object "{obj}" not found on Alyx')
        collection_set = {x['collection'] for x in results if match(x)}
        if len(collection_set) > 1:
            raise ALFMultipleCollectionsFound('Matching object belongs to multiple collections:' +
                                              ', '.join(collection_set))

        # Download and optionally load the datasets
        out_files = self.download_datasets(x for x in results if match(x))
        assert not any(x is None for x in out_files), 'failed to download object'
        if download_only:
            return out_files
        else:
            return alfio.load_object(out_files[0].parent, obj, **kwargs)

    def _load_recursive(self, eid, **kwargs):
        """
        From a Session ID and dataset types, queries Alyx database, downloads the data
        from Globus, and loads into numpy array. Supports multiple sessions
        """
        if isinstance(eid, str):
            return self._load(eid, **kwargs)
        if isinstance(eid, list):
            # dataclass output requested
            if kwargs.get('dclass_output', False):
                for i, e in enumerate(eid):
                    if i == 0:
                        out = self._load(e, **kwargs)
                    else:
                        out.append(self._load(e, **kwargs))
            else:  # list output requested
                out = []
                for e in eid:
                    out.append(self._load(e, **kwargs)[0])
            return out

    def to_eid(self,
               id: Listable(Union[str, Path, UUID, dict]) = None,
               cache_dir: Optional[Union[str, Path]] = None) -> Listable(str):
        if isinstance(id, (list, tuple)):  # Recurse
            return [self.to_eid(i, cache_dir) for i in id]
        if isinstance(id, UUID):
            return str(id)
        # elif is_exp_ref(id):
        #     return ref2eid(id, one=self)
        elif isinstance(id, dict):
            assert {'subject', 'number', 'start_time', 'lab'}.issubset(id)
            root = Path(self._get_cache_dir(cache_dir))
            id = root.joinpath(
                id['lab'],
                'Subjects', id['subject'],
                id['start_time'][:10],
                ('%03d' % id['number']))

        if alfio.is_session_path(id):
            return self.eid_from_path(id)
        elif isinstance(id, str):
            if len(id) > 36:
                id = id[-36:]
            if not alfio.is_uuid_string(id):
                raise ValueError('Invalid experiment ID')
            else:
                return id
        else:
            raise ValueError('Unrecognized experiment ID')

    def _make_dataclass(self, eid, dataset_types=None, cache_dir=None, dry_run=False,
                        clobber=False, offline=False, keep_uuid=False):
        # if the input as an UUID, add the beginning of URL to it
        cache_dir = self._get_cache_dir(cache_dir)
        # get session json information as a dictionary from the alyx API
        try:
            ses = self.alyx.rest('sessions', 'read', id=eid)
        except requests.HTTPError:
            raise requests.HTTPError('Session ' + eid + ' does not exist')
        # filter by dataset types
        dc = SessionDataInfo.from_session_details(ses, dataset_types=dataset_types, eid=eid)
        # loop over each dataset and download if necessary
        with concurrent.futures.ThreadPoolExecutor(max_workers=NTHREADS) as executor:
            futures = []
            for ind in range(len(dc)):
                if dc.url[ind] is None or dry_run:
                    futures.append(None)
                else:
                    futures.append(executor.submit(
                        self.download_dataset, dc.url[ind], cache_dir=cache_dir, clobber=clobber,
                        offline=offline, keep_uuid=keep_uuid, file_size=dc.file_size[ind],
                        hash=dc.hash[ind]))
            concurrent.futures.wait(list(filter(lambda x: x is not None, futures)))
            for ind, future in enumerate(futures):
                if future is None:
                    continue
                dc.local_path[ind] = future.result()
        # filter by daataset types and update the cache
        self._update_cache(ses, dataset_types=dataset_types)
        return dc

    def _ls(self, table=None, verbose=False):
        """
        Queries the database for a list of 'users' and/or 'dataset-types' and/or 'subjects' fields

        :param table: the table (s) to query among: 'dataset-types','users'
         and 'subjects'; if empty or None assumes all tables
        :type table: str
        :param verbose: [False] prints the list in the current window
        :type verbose: bool

        :return: list of names to query, list of full raw output in json serialized format
        :rtype: list, list
        """
        assert (isinstance(table, str))
        table_field_names = {
            'dataset-types': 'name',
            'datasets': 'name',
            'users': 'username',
            'subjects': 'nickname',
            'labs': 'name'}
        if not table or table not in list(set(_ENDPOINTS.keys())):
            raise KeyError("The attribute/endpoint: " + table + " doesn't exist \n" +
                           "possible values are " + str(set(_ENDPOINTS.values())))

        field_name = table_field_names[_ENDPOINTS[table]]
        full_out = self.alyx.get('/' + _ENDPOINTS[table])
        list_out = [f[field_name] for f in full_out]
        if verbose:
            pprint(list_out)
        return list_out, full_out

    # def search(self, dataset_types=None, users=None, subjects=None, date_range=None,
    #            lab=None, number=None, task_protocol=None, details=False):
    def search(self, details=False, limit=None, **kwargs):
        """
        Applies a filter to the sessions (eid) table and returns a list of json dictionaries
         corresponding to sessions.

        For a list of search terms, use the methods

        >>> one.search_terms()

        :param dataset_types: list of dataset_types
        :type dataset_types: list of str

        :param date_range: list of 2 strings or list of 2 dates that define the range
        :type date_range: list

        :param details: default False, returns also the session details as per the REST response
        :type details: bool

        :param lab: a str or list of lab names
        :type lab: list or str

        :param limit: default None, limits results (if pagination enabled on server)
        :type limit: int List of possible search terms

        :param location: a str or list of lab location (as per Alyx definition) name
                         Note: this corresponds to the specific rig, not the lab geographical
                         location per se
        :type location: str

        :param number: number of session to be returned; will take the first n sessions found
        :type number: str or int

        :param performance_lte / performance_gte: search only for sessions whose performance is
        less equal or greater equal than a pre-defined threshold as a percentage (0-100)
        :type performance_gte: float

        :param subjects: a list of subjects nickname
        :type subjects: list or str

        :param task_protocol: a str or list of task protocol name (can be partial, i.e.
                              any task protocol containing that str will be found)
        :type task_protocol: str

        :param users: a list of users
        :type users: list or str

        :return: list of eids, if details is True, also returns a list of json dictionaries,
         each entry corresponding to a matching session
        :rtype: list, list


        """

        # small function to make sure string inputs are interpreted as lists
        def validate_input(inarg):
            if isinstance(inarg, str):
                return [inarg]
            elif isinstance(inarg, int):
                return [str(inarg)]
            else:
                return inarg

        # loop over input arguments and build the url
        url = '/sessions?'
        for k in kwargs.keys():
            # check that the input matches one of the defined filters
            if k not in SEARCH_TERMS:
                _logger.error(f'"{k}" is not a valid search keyword' + '\n' +
                              "Valid keywords are: " + str(set(SEARCH_TERMS.values())))
                return
            # then make sure the field is formatted properly
            field = SEARCH_TERMS[k]
            if field == 'date_range':
                query = _validate_date_range(kwargs[k])
            else:
                query = validate_input(kwargs[k])
            # at last append to the URL
            url = url + f"&{field}=" + ','.join(query)
        # the REST pagination argument has to be the last one
        if limit:
            url += f'&limit={limit}'
        # implements the loading itself
        ses = self.alyx.get(url)
        if len(ses) > 2500:
            eids = [s['url'] for s in tqdm.tqdm(ses)]  # flattens session info
        else:
            eids = [s['url'] for s in ses]
        eids = [e.split('/')[-1] for e in eids]  # remove url to make it portable
        if details:
            for s in ses:
                if all([s.get('lab'), s.get('subject'), s.get('start_time')]):
                    s['local_path'] = str(Path(self._par.CACHE_DIR, s['lab'], 'Subjects',
                                               s['subject'], s['start_time'][:10],
                                               str(s['number']).zfill(3)))
                else:
                    s['local_path'] = None
            return eids, ses
        else:
            return eids

    def download_datasets(self, dsets, **kwargs):
        """
        Download several datsets through a list of alyx REST dictionaries
        :param dset: list of dataset dictionaries from an Alyx REST query OR list of URL strings
        :return: local file path
        """
        out_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NTHREADS) as executor:
            futures = [executor.submit(self.download_dataset, dset, file_size=dset['file_size'],
                                       hash=dset['hash'], **kwargs) for dset in dsets]
            concurrent.futures.wait(futures)
            for future in futures:
                out_files.append(future.result())
        return out_files

    def download_dataset(self, dset, cache_dir=None, **kwargs):
        """
        Download a dataset from an alyx REST dictionary
        :param dset: single dataset dictionary from an Alyx REST query OR URL string
        :param cache_dir (optional): root directory to save the data in (home/downloads by default)
        :return: local file path
        """
        if isinstance(dset, str):
            url = dset
        else:
            url = next((fr['data_url'] for fr in dset['file_records']
                        if fr['data_url'] and fr['exists']), None)
        if not url:
            str_dset = Path(dset['collection']).joinpath(dset['name'])
            _logger.warning(f"{str_dset} exist flag or url not found in Alyx")
            return
        assert url.startswith(self._par.HTTP_DATA_SERVER), \
            ('remote protocol and/or hostname does not match HTTP_DATA_SERVER parameter:\n' +
             f'"{url[:40]}..." should start with "{self._par.HTTP_DATA_SERVER}"')
        relpath = Path(url.replace(self._par.HTTP_DATA_SERVER, '.')).parents[0]
        target_dir = Path(self._get_cache_dir(cache_dir), relpath)
        return self._download_file(url=url, target_dir=target_dir, **kwargs)

    def _tag_mismatched_file_record(self, url):
        fr = self.alyx.rest('files', 'list', django=f"dataset,{Path(url).name.split('.')[-2]},"
                                                    f"data_repository__globus_is_personal,False")
        if len(fr) > 0:
            json_field = fr[0]['json']
            if json_field is None:
                json_field = {'mismatch_hash': True}
            else:
                json_field.update({'mismatch_hash': True})
            self.alyx.rest('files', 'partial_update', id=fr[0]['url'][-36:],
                           data={'json': json_field})

    def _download_file(self, url, target_dir, clobber=False, offline=False, keep_uuid=False,
                       file_size=None, hash=None):
        """
        Downloads a single file from an HTTP webserver
        :param url:
        :param clobber: (bool: False) overwrites local dataset if any
        :param offline:
        :param keep_uuid:
        :param file_size:
        :param hash:
        :return:
        """
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        local_path = str(target_dir) + os.sep + os.path.basename(url)
        if not keep_uuid:
            local_path = alfio.remove_uuid_file(local_path, dry=True)
        if Path(local_path).exists():
            # the local file hash doesn't match the dataset table cached hash
            hash_mismatch = hash and hashfile.md5(Path(local_path)) != hash
            file_size_mismatch = file_size and Path(local_path).stat().st_size != file_size
            if hash_mismatch or file_size_mismatch:
                clobber = True
                _logger.warning(f" local md5 or size mismatch, re-downloading {local_path}")
        # if there is no cached file, download
        else:
            clobber = True
        if clobber and not offline:
            local_path, md5 = wc.http_download_file(
                url, username=self._par.HTTP_DATA_SERVER_LOGIN,
                password=self._par.HTTP_DATA_SERVER_PWD, cache_dir=str(target_dir),
                clobber=clobber, return_md5=True)
            # post download, if there is a mismatch between Alyx and the newly downloaded file size
            # or hash flag the offending file record in Alyx for database maintenance
            hash_mismatch = hash and md5 != hash
            file_size_mismatch = file_size and Path(local_path).stat().st_size != file_size
            if hash_mismatch or file_size_mismatch:
                self._tag_mismatched_file_record(url)
        if keep_uuid:
            return local_path
        else:
            return alfio.remove_uuid_file(local_path)

    @staticmethod
    def search_terms():
        """
        Returns possible search terms to be used in the one.search method.

        :return: a tuple containing possible search terms:
        :rtype: tuple
        """
        return sorted(list(set(SEARCH_TERMS.values())))

    @staticmethod
    def keywords():
        """
        Returns possible keywords to be used in the one.list method

        :return: a tuple containing possible search terms:
        :rtype: tuple
        """
        return sorted(list(set(_ENDPOINTS.values())))

    @staticmethod
    def setup():
        """
        Interactive command tool that populates parameter file for ONE IBL.
        """
        oneibl.params.setup()

    def path_from_eid(self, eid: str, use_cache: bool = True) -> Listable(Path):
        """
        From an experiment id or a list of experiment ids, gets the local cache path
        :param eid: eid (UUID) or list of UUIDs
        :param use_cache: if set to False, will force database connection
        :return: eid or list of eids
        """
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            path_list = []
            for p in eid:
                path_list.append(self.path_from_eid(p))
            return path_list
        # If not valid return None
        if not alfio.is_uuid_string(eid):
            print(eid, " is not a valid eID/UUID string")
            return

        # first try avoid hitting the database
        if self._cache.size > 0 and use_cache:
            cache_path = super().path_from_eid(eid)
            if cache_path:
                return cache_path

        # if it wasn't successful, query Alyx
        ses = self.alyx.rest('sessions', 'list', django=f'pk,{eid}')
        if len(ses) == 0:
            return None
        else:
            return Path(self._par.CACHE_DIR).joinpath(
                ses[0]['lab'], 'Subjects', ses[0]['subject'], ses[0]['start_time'][:10],
                str(ses[0]['number']).zfill(3))

    def eid_from_path(self, path_obj: Union[str, Path], use_cache: bool = True) -> Listable(Path):
        """
        From a local path, gets the experiment id
        :param path_obj: local path or list of local paths
        :param use_cache: if set to False, will force database connection
        :return: eid or list of eids
        """
        # If path_obj is a list recurse through it and return a list
        if isinstance(path_obj, list):
            path_obj = [Path(x) for x in path_obj]
            eid_list = []
            for p in path_obj:
                eid_list.append(self.eid_from_path(p))
            return eid_list
        # else ensure the path ends with mouse,date, number
        path_obj = Path(path_obj)
        session_path = alfio.get_session_path(path_obj)
        # if path does not have a date and a number return None
        if session_path is None:
            return None

        # try the cached info to possibly avoid hitting database
        cache_eid = super().eid_from_path(path_obj)
        if cache_eid:
            return cache_eid

        # if not search for subj, date, number XXX: hits the DB
        uuid = self.search(subjects=session_path.parts[-3],
                           date_range=session_path.parts[-2],
                           number=session_path.parts[-1])

        # Return the uuid if any
        return uuid[0] if uuid else None

    def url_from_path(self, filepath):
        """
        Given a local file path, returns the URL of the remote file.
        :param filepath: A local file path
        :return: A URL string
        """
        eid = self.eid_from_path(filepath)
        try:
            dataset, = self.alyx.rest('datasets', 'list', session=eid, name=Path(filepath).name)
        except ValueError:
            raise ALFObjectNotFound(f'File record for {filepath} not found on Alyx')
        return next(
            r['data_url'] for r in dataset['file_records'] if r['data_url'] and r['exists'])

    @parse_id
    def datasets_from_type(self, eid, dataset_type, full=False):
        """
        Get list of datasets belonging to a given dataset type for a given session
        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param dataset_type: A dataset type, e.g. camera.times
        :param full: If True, a dictionary of details is returned for each dataset
        :return: A list of datasets belonging to that session's dataset type
        """
        restriction = f'session__id,{eid},dataset_type__name,{dataset_type}'
        datasets = self.alyx.rest('datasets', 'list', django=restriction)
        return datasets if full else [d['name'] for d in datasets]

    def get_details(self, eid: str, full: bool = False):
        """ Returns details of eid like from one.search, optional return full
        session details.
        """
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            details_list = []
            for p in eid:
                details_list.append(self.get_details(p, full=full))
            return details_list
        # If not valid return None
        if not alfio.is_uuid_string(eid):
            print(eid, " is not a valid eID/UUID string")
            return
        # load all details
        dets = self.alyx.rest("sessions", "read", eid)
        if full:
            return dets
        # If it's not full return the normal output like from a one.search
        det_fields = ["subject", "start_time", "number", "lab", "project",
                      "url", "task_protocol", "local_path"]
        out = {k: v for k, v in dets.items() if k in det_fields}
        out.update({'local_path': self.path_from_eid(eid)})
        return out

    def _update_cache(self, ses, dataset_types):
        """
        :param ses: session details dictionary as per Alyx response
        :param dataset_types:
        :return: is_updated (bool): if the cache was updated or not
        """
        save = False
        pqt_dsets = _ses2pandas(ses, dtypes=dataset_types)
        # if the dataframe is empty, return
        if pqt_dsets.size == 0:
            return
        # if the cache is empty create the cache variable
        elif self._cache.size == 0:
            self._cache = pqt_dsets
            save = True
        # the cache is not empty and there are datasets in the query
        else:
            isin, icache = ismember2d(pqt_dsets[['id_0', 'id_1']].to_numpy(),
                                      self._cache[['id_0', 'id_1']].to_numpy())
            # check if the hash / filesize fields have changed on patching
            heq = (self._cache['hash'].iloc[icache].to_numpy() ==
                   pqt_dsets['hash'].iloc[isin].to_numpy())
            feq = np.isclose(self._cache['file_size'].iloc[icache].to_numpy(),
                             pqt_dsets['file_size'].iloc[isin].to_numpy(),
                             rtol=0, atol=0, equal_nan=True)
            eq = np.logical_and(heq, feq)
            # update new hash / filesizes
            if not np.all(eq):
                self._cache.iloc[icache, 4:6] = pqt_dsets.iloc[np.where(isin)[0], 4:6].to_numpy()
                save = True
            # append datasets that haven't been found
            if not np.all(isin):
                self._cache = self._cache.append(pqt_dsets.iloc[np.where(~isin)[0]])
                self._cache = self._cache.reindex()
                save = True
        if save:
            # before saving makes sure pandas did not cast uuids in float
            typs = [t for t, k in zip(self._cache.dtypes, self._cache.keys()) if 'id_' in k]
            assert (all(map(lambda t: t == np.int64, typs)))
            # if this gets too big, look into saving only when destroying the ONE object
            parquet.save(self._cache_file, self._cache)

    def download_raw_partial(self, url_cbin, url_ch, first_chunk=0, last_chunk=0):
        assert str(url_cbin).endswith('.cbin')
        assert str(url_ch).endswith('.ch')

        relpath = Path(url_cbin.replace(self._par.HTTP_DATA_SERVER, '.')).parents[0]
        target_dir = Path(self._get_cache_dir(None), relpath)
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        # First, download the .ch file if necessary
        if isinstance(url_ch, Path):
            ch_file = url_ch
        else:
            ch_file = Path(wc.http_download_file(
                url_ch,
                username=self._par.HTTP_DATA_SERVER_LOGIN,
                password=self._par.HTTP_DATA_SERVER_PWD,
                cache_dir=target_dir, clobber=True, return_md5=False))
            ch_file = alfio.remove_uuid_file(ch_file)
        ch_file_stream = ch_file.with_suffix('.stream.ch')
        # Load the .ch file.
        with open(ch_file, 'r') as f:
            cmeta = json.load(f)

        # Get the first byte and number of bytes to download.
        i0 = cmeta['chunk_bounds'][first_chunk]
        ns_stream = cmeta['chunk_bounds'][last_chunk + 1] - i0

        # if the cached version happens to be the same as the one on disk, just load it
        if ch_file_stream.exists():
            with open(ch_file_stream, 'r') as f:
                cmeta_stream = json.load(f)
            if (cmeta_stream.get('chopped_first_sample', None) == i0 and
                    cmeta_stream.get('chopped_total_samples', None) == ns_stream):
                return spikeglx.Reader(ch_file_stream.with_suffix('.cbin'))
        else:
            shutil.copy(ch_file, ch_file_stream)
        assert ch_file_stream.exists()

        # prepare the metadata file
        cmeta['chunk_bounds'] = cmeta['chunk_bounds'][first_chunk:last_chunk + 2]
        cmeta['chunk_bounds'] = [_ - i0 for _ in cmeta['chunk_bounds']]
        assert len(cmeta['chunk_bounds']) >= 2
        assert cmeta['chunk_bounds'][0] == 0

        first_byte = cmeta['chunk_offsets'][first_chunk]
        cmeta['chunk_offsets'] = cmeta['chunk_offsets'][first_chunk:last_chunk + 2]
        cmeta['chunk_offsets'] = [_ - first_byte for _ in cmeta['chunk_offsets']]
        assert len(cmeta['chunk_offsets']) >= 2
        assert cmeta['chunk_offsets'][0] == 0
        n_bytes = cmeta['chunk_offsets'][-1]
        assert n_bytes > 0

        # Save the chopped chunk bounds and ossets.
        cmeta['sha1_compressed'] = None
        cmeta['sha1_uncompressed'] = None
        cmeta['chopped'] = True
        cmeta['chopped_first_sample'] = i0
        cmeta['chopped_total_samples'] = ns_stream

        with open(ch_file_stream, 'w') as f:
            json.dump(cmeta, f, indent=2, sort_keys=True)

        # Download the requested chunks
        cbin_local_path = wc.http_download_file(
            url_cbin,
            username=self._par.HTTP_DATA_SERVER_LOGIN,
            password=self._par.HTTP_DATA_SERVER_PWD,
            cache_dir=target_dir, clobber=True, return_md5=False,
            chunks=(first_byte, n_bytes))
        cbin_local_path = alfio.remove_uuid_file(cbin_local_path)
        cbin_local_path_renamed = cbin_local_path.with_suffix('.stream.cbin')
        cbin_local_path.rename(cbin_local_path_renamed)
        assert cbin_local_path_renamed.exists()

        shutil.copy(cbin_local_path.with_suffix('.meta'),
                    cbin_local_path_renamed.with_suffix('.meta'))
        reader = spikeglx.Reader(cbin_local_path_renamed)
        return reader


def _validate_date_range(date_range):
    """
    Validates and arrange date range in a 2 elements list
    """
    if isinstance(date_range, str):
        date_range = [date_range, date_range]
    if len(date_range) == 1:
        date_range = [date_range[0], date_range[0]]
    return date_range

import concurrent.futures
import logging
import os
from pathlib import Path, PurePath

import requests
import tqdm

import oneibl.params
import oneibl.webclient as wc
from alf.io import (AlfBunch, get_session_path, is_uuid_string,
                    load_file_content, remove_uuid_file)
from ibllib.io import hashfile
from ibllib.io.one import OneAbstract
from ibllib.misc import pprint
from oneibl.dataclass import SessionDataInfo

_logger = logging.getLogger('ibllib')

NTHREADS = 4  # number of download threads

_ENDPOINTS = {  # keynames are possible input arguments and values are actual endpoints
    'data': 'dataset-types',
    'dataset': 'dataset-types',
    'datasets': 'dataset-types',
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

_LIST_KEYWORDS = dict(_SESSION_FIELDS, **{
    'all': 'all',
    'data': 'dataset-type',
    'dataset': 'dataset-type',
    'datasets': 'dataset-type',
    'dataset-types': 'dataset-type',
    'dataset_types': 'dataset-type',
    'dataset-type': 'dataset-type',
    'dataset_type': 'dataset-type',
    'dtypes': 'dataset-type',
    'dtype': 'dataset-type',
    'labs': 'lab',
    'lab': 'lab'
})

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


class ONE(OneAbstract):
    def __init__(self, username=None, password=None, base_url=None, silent=False, printout=True):
        # get parameters override if inputs provided
        self._par = oneibl.params.get(silent=silent)
        self._par = self._par.set('ALYX_LOGIN', username or self._par.ALYX_LOGIN)
        self._par = self._par.set('ALYX_URL', base_url or self._par.ALYX_URL)
        self._par = self._par.set('ALYX_PWD', password or self._par.ALYX_PWD)

        try:
            self._alyxClient = wc.AlyxClient(username=self._par.ALYX_LOGIN,
                                             password=self._par.ALYX_PWD,
                                             base_url=self._par.ALYX_URL)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Can't connect to {self._par.ALYX_URL}.\n" +
                "IP addresses are filtered on IBL database servers. \n" +
                "Are you connecting from an IBL participating institution ?"
            )
        # Init connection to Globus if needed
        # Display output when instantiating ONE
        if printout:
            print(f"Connected to {self._par.ALYX_URL} as {self._par.ALYX_LOGIN}",)

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

    def list(self, eid=None, keyword='dataset-type', details=False):
        """
        From a Session ID, queries Alyx database for datasets-types related to a session.

        :param eid: Experiment ID, for IBL this is the UUID String of the Session as per Alyx
         database. Example: '698361f6-b7d0-447d-a25d-42afdef7a0da'
         If None, returns the set of possible values. Only for the following keys:
         ('users', 'dataset-types', subjects')
        :type eid: str or list of strings

        :param keyword: The attribute to be listed.
        :type keyword: str

        :param details: returns a second argument with a full dictionary to provide context
        :type details: bool

        :return: list of strings, plus list of dictionaries if details option selected
        :rtype:  list, list

        for a list of keywords, use the methods `one.search_terms()`
        """
        # check and validate the input keyword
        if keyword.lower() not in set(_LIST_KEYWORDS.keys()):
            raise KeyError("The field: " + keyword + " doesn't exist in the Session model." +
                           "\n Here is a list of expected values: " +
                           str(set(_LIST_KEYWORDS.values())))
        keyword = _LIST_KEYWORDS[keyword.lower()]  # accounts for possible typos

        # recursive call for cases where a list is provided
        if isinstance(eid, list):
            out = []
            for e in eid:
                out.append(self.list(e, keyword=keyword, details=details))
            if details and (keyword != 'dataset-type'):
                return [[o[0] for o in out], [o[1] for o in out]]
            else:
                return out

        #  this is for basic endpoints queries: dataset-type, users, and subjects with None
        if eid is None:
            out = self._ls(table=keyword)
            if details:
                return out
            else:
                return out[0]

        # this is a query about datasets: need to unnest session info through the load function
        if keyword == 'dataset-type':
            dses = self.load(eid, dataset_types='__all__', dry_run=True)
            dlist = list(sorted(set(dses.dataset_type)))
            if details:
                return dses
            else:
                return dlist

        # get the session information
        ses = self.alyx.rest('sessions', 'read', eid)

        if keyword.lower() == 'all':
            return [ses]
        elif details:
            return ses[keyword], ses
        else:
            return ses[keyword]

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

    def load_dataset(self, eid, dataset_type, **kwargs):
        """
        Load a single dataset from a Session ID and a dataset type.

        :param eid: Experiment ID, for IBL this is the UUID of the Session as per Alyx
         database. Could be a full Alyx URL:
         'http://localhost:8000/sessions/698361f6-b7d0-447d-a25d-42afdef7a0da' or only the UUID:
         '698361f6-b7d0-447d-a25d-42afdef7a0da'.
        :type eid: str
        :param dataset_type: Alyx dataset type to be returned.
        :type dataset_types: str

        :return: A numpy array.
        :rtype: numpy array
        """
        return self._load(eid, dataset_types=[dataset_type], **kwargs)[0]

    def load_object(self, eid, obj, **kwargs):
        """
        Load all attributes of an ALF object from a Session ID and an object name.

        :param eid: Experiment ID, for IBL this is the UUID of the Session as per Alyx
         database. Could be a full Alyx URL:
         'http://localhost:8000/sessions/698361f6-b7d0-447d-a25d-42afdef7a0da' or only the UUID:
         '698361f6-b7d0-447d-a25d-42afdef7a0da'.
        :type eid: str
        :param obj: Alyx object to load.
        :type obj: str

        :return: A dictionary-like structure with one key per dataset type for the requested
         object, and a NumPy array per value.
        :rtype: AlfBunch instance
        """
        dataset_types = [dst for dst in self.list(eid) if dst.startswith(obj)]
        if len(dataset_types) == 0:
            _logger.warning(f"{eid} does not contain any {obj} object datasets")
            return
        dsets = self._load(eid, dataset_types=dataset_types, **kwargs)
        return AlfBunch({
            '.'.join(dataset_types[i].split('.')[1:]): dsets[i]
            for i in range(len(dataset_types))})

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

    def _load(self, eid, dataset_types=None, dclass_output=False, dry_run=False, cache_dir=None,
              download_only=False, clobber=False, offline=False, keep_uuid=False):
        """
        From a Session ID and dataset types, queries Alyx database, downloads the data
        from Globus, and loads into numpy array. Single session only
        """
        # if the input as an UUID, add the beginning of URL to it
        cache_dir = self._get_cache_dir(cache_dir)
        if is_uuid_string(eid):
            eid = '/sessions/' + eid
        eid_str = eid[-36:]
        # get session json information as a dictionary from the alyx API
        try:
            ses = self.alyx.get('/sessions/' + eid_str)
        except requests.HTTPError:
            raise requests.HTTPError('Session ' + eid_str + ' does not exist')
        # ses = ses[0]
        # if no dataset_type is provided:
        # a) force the output to be a dictionary that provides context to the data
        # b) download all types that have a data url specified whithin the alf folder
        dataset_types = [dataset_types] if isinstance(dataset_types, str) else dataset_types
        if not dataset_types or dataset_types == ['__all__']:
            dclass_output = True
        # this performs the filtering
        dc = SessionDataInfo.from_session_details(ses, dataset_types=dataset_types, eid=eid_str)
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
        # load the files content in variables if requested
        if not download_only:
            for ind, fil in enumerate(dc.local_path):
                dc.data[ind] = load_file_content(fil)
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
            'users': 'username',
            'subjects': 'nickname',
            'labs': 'name'}
        if not table or table not in list(set(_ENDPOINTS.keys())):
            raise KeyError("The attribute/endpoint: " + table + " doesn't exist \n" +
                           "possible values are " + str(set(_ENDPOINTS.values())))
        full_out = []
        list_out = []
        for ind, tab in enumerate(_ENDPOINTS):
            if tab == table:
                field_name = table_field_names[_ENDPOINTS[tab]]
                full_out.append(self.alyx.get('/' + _ENDPOINTS[tab]))
                list_out.append([f[field_name] for f in full_out[-1]])
        if verbose:
            pprint(list_out)
        return list_out[0], full_out[0]

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

    def _get_cache_dir(self, cache_dir):
        if not cache_dir:
            cache_dir = self._par.CACHE_DIR
        # if empty in parameter file, do not allow and set default
        if not cache_dir:
            cache_dir = str(PurePath(Path.home(), "Downloads", "FlatIron"))
        return cache_dir

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
            url = next((fr['data_url'] for fr in dset['file_records'] if fr['data_url']), None)
        if not url:
            return
        relpath = Path(url.replace(self._par.HTTP_DATA_SERVER, '.')).parents[0]
        target_dir = Path(self._get_cache_dir(cache_dir), relpath)
        return self._download_file(url=url, target_dir=target_dir, **kwargs)

    def _download_file(self, url, target_dir, clobber=False, offline=False, keep_uuid=False,
                       file_size=None, hash=None):
        """
        Downloads a single file from an HTTP webserver
        :param url:
        :param cache_dir:
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
            local_path = remove_uuid_file(local_path, dry=True)
        if Path(local_path).exists():
            # overwrites the file if the expected filesize is different from the cached filesize
            if file_size and Path(local_path).stat().st_size != file_size:
                clobber = True
            # overwrites the file if the expected hash is different from the cached hash
            if hash and hashfile.md5(Path(local_path)) != hash:
                clobber = True
        # if there is no cached file, download
        else:
            clobber = True
        if clobber:
            local_path = wc.http_download_file(url,
                                               username=self._par.HTTP_DATA_SERVER_LOGIN,
                                               password=self._par.HTTP_DATA_SERVER_PWD,
                                               cache_dir=str(target_dir),
                                               clobber=clobber,
                                               offline=offline)
        if keep_uuid:
            return local_path
        else:
            return remove_uuid_file(local_path)

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

    def path_from_eid(self, eid: str, grep_str=None) -> Path:
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            path_list = []
            for p in eid:
                path_list.append(self.path_from_eid(p, grep_str=grep_str))
            return path_list
        # If not valid return None
        if not is_uuid_string(eid):
            print(eid, " is not a valid eID/UUID string")
            return
        # Load data, if no data present on disk return None
        data = self._load(eid, download_only=True, offline=True)
        if not data.local_path:
            return None
        # If user defined a grep list of specific files return paths to files
        if grep_str is not None:
            files = [x for x in data.local_path if grep_str in str(x)]
            return files
        # If none of the above happen return the session path of the first file you find
        session_path = get_session_path(data.local_path[0])

        return session_path

    def eid_from_path(self, path_obj):
        # If path_obj is a list recurse through it and return a list
        if isinstance(path_obj, list):
            path_obj = [Path(x) for x in path_obj]
            eid_list = []
            for p in path_obj:
                eid_list.append(self.eid_from_path(p))
            return eid_list
        # else ensure the path ends with mouse,date, number
        path_obj = Path(path_obj)
        session_path = get_session_path(path_obj)
        # if path does not have a date and a number return None
        if session_path is None:
            return None
        # search for subj, date, number XXX: hits the DB
        uuid = self.search(subjects=session_path.parts[-3],
                           date_range=session_path.parts[-2],
                           number=session_path.parts[-1])
        # Return the uuid if any
        return uuid[0] if uuid else None

    def get_details(self, eid, full=False):
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
        if not is_uuid_string(eid):
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


def _validate_date_range(date_range):
    """
    Validates and arrange date range in a 2 elements list
    """
    if isinstance(date_range, str):
        date_range = [date_range, date_range]
    if len(date_range) == 1:
        date_range = [date_range[0], date_range[0]]
    return date_range

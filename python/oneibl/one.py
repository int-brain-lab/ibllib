import os
from dataclasses import dataclass, field
import abc
from pathlib import Path, PurePath
import requests

import numpy as np
import pandas as pd

import ibllib.webclient as wc
from ibllib.misc import is_uuid_string, pprint
import oneibl.params

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
    'subject': 'subjects',
    'subjects': 'subjects',
    'date_range': 'date_range',
    'date-range': 'date_range',
    'labs': 'lab',
    'lab': 'lab'
}
par = oneibl.params.get()


class OneAbstract(abc.ABC):

    @abc.abstractmethod
    def load(self, eid, **kwargs):
        return

    @abc.abstractmethod
    def list(self, **kwargs):
        return

    @abc.abstractmethod
    def search(self, **kwargs):
        return


@dataclass
class SessionDataInfo:
    """
    Dataclass that provides dataset list, dataset_id, local_path, dataset_type, url and eid fields
    """
    data: list = field(default_factory=list)
    dataset_id: list = field(default_factory=list)
    local_path: list = field(default_factory=list)
    dataset_type: list = field(default_factory=list)
    url: list = field(default_factory=list)
    eid: list = field(default_factory=list)

    def __str__(self):
        """
        This is to make print outputs more useful"
        """
        str_out = ''
        d = self.__dict__
        for k in d.keys():
            str_out += (k + '    : ' + str(type(d[k])) + ' , ' + str(len(d[k])) + ' items = ' +
                        str(d[k][0])) + '\n'
        return str_out


class ONE(OneAbstract):
    def __init__(self, username=par.ALYX_LOGIN, password=par.ALYX_PWD, base_url=par.ALYX_URL):
        # Init connection to the database
        try:
            self._alyxClient = wc.AlyxClient(username=username, password=password,
                                             base_url=base_url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Can't connect to " + base_url + '. \n' +
                                  'IP addresses are filtered on IBL database servers. \n' +
                                  'Are you connecting from an IBL participating institution ?')
        print('Connected to ' + base_url + ' as ' + username)

    @property
    def alyx(self):
        return self._alyxClient

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
            dses = self.load(eid, dry_run=True)
            dlist = list(sorted(set(dses.dataset_type)))
            if details:
                return dses
            else:
                return dlist

        # get the session information
        ses = self._alyxClient.get('/sessions?id=' + eid)

        if keyword.lower() == 'all':
            return ses
        elif details:
            return ses[0][keyword], ses
        else:
            return ses[0][keyword]

    def load(self, eid, dataset_types=None, dclass_output=False, dry_run=False, cache_dir=None,
             download_only=False):
        """
        From a Session ID and dataset types, queries Alyx database, downloads the data
        from Globus, and loads into numpy array.

        :param eid: Experiment ID, for IBL this is the UUID of the Session as per Alyx
         database. Could be a full Alyx URL:
         'http://localhost:8000/sessions/698361f6-b7d0-447d-a25d-42afdef7a0da' or only the UUID:
         '698361f6-b7d0-447d-a25d-42afdef7a0da'
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

        :return: List of numpy arrays matching the size of dataset_types parameter, OR
         a dataclass containing arrays and context data.
        :rtype: list, dict, dataclass SessionDataInfo
        """
        # TODO: feature that downloads a list of datasets from a list of sessions,
        # TODO in this case force dictionary output
        # if the input as an UUID, add the beginning of URL to it
        if not cache_dir:
            cache_dir = par.CACHE_DIR
        if is_uuid_string(eid):
            eid = '/sessions/' + eid
        eid_str = eid[-36:]
        # get session json information as a dictionary from the alyx API
        ses = self._alyxClient.get('/sessions?id=' + eid_str)
        if not ses:
            raise FileNotFoundError('Session ' + eid_str + ' does not exist')
        ses = ses[0]
        # if no dataset_type is provided:
        # a) force the output to be a dictionary that provides context to the data
        # b) download all types that have a data url specified
        dataset_types = [dataset_types] if isinstance(dataset_types, str) else dataset_types
        if not dataset_types:
            dclass_output = True
            dataset_types = [d['dataset_type'] for d in ses['data_dataset_session_related']
                             if d['data_url']]
        # loop over each dataset related to the session ID and get list of files urls
        session_dtypes = [d['dataset_type'] for d in ses['data_dataset_session_related']]
        out = SessionDataInfo()
        # this first loop only downloads the file to ease eventual refactoring
        for ind, dt in enumerate(dataset_types):
            for [i, sdt] in enumerate(session_dtypes):
                if sdt == dt:
                    urlstr = ses['data_dataset_session_related'][i]['data_url']
                    if urlstr and not dry_run:
                        rel_path = PurePath(urlstr.replace(par.HTTP_DATA_SERVER, '.')).parents[0]
                        cache_dir_file = PurePath(cache_dir, rel_path)
                        Path(cache_dir_file).mkdir(parents=True, exist_ok=True)
                        fil = wc.http_download_file(urlstr,
                                                    username=par.HTTP_DATA_SERVER_LOGIN,
                                                    password=par.HTTP_DATA_SERVER_PWD,
                                                    cache_dir=str(cache_dir_file))
                    else:
                        fil = ''
                    out.eid.append(eid_str)
                    out.dataset_type.append(dt)
                    out.url.append(urlstr)
                    out.local_path.append(fil)
                    out.dataset_id.append(ses['data_dataset_session_related'][i]['id'])
                    out.data.append([])
        # then another loop over files and load them in numpy. If not npy, just pass empty list
        # the data loading per format needs to be implemented in a generic function in ibllib/alf.
        for ind, fil in enumerate(out.local_path):
            if download_only:
                continue
            if fil and os.path.getsize(fil) == 0:
                continue
            if fil and os.path.splitext(fil)[1] == '.npy':
                out.data[ind] = np.load(file=fil)
            if fil and os.path.splitext(fil)[1] == '.json':
                pass  # FIXME would be nice to implement json read but param from matlab RIG fails
            if fil and os.path.splitext(fil)[1] == '.tsv':
                out.data[ind] = pd.read_csv(fil, delimiter='\t')
            if fil and os.path.splitext(fil)[1] == '.csv':
                out.data[ind] = pd.read_csv(fil)
        if dclass_output:
            return out
        # if required, parse the output as a list that matches dataset types provided
        list_out = []
        for dt in dataset_types:
            if dt not in out.dataset_type:
                list_out.append(None)
                continue
            for i, x, in enumerate(out.dataset_type):
                if dt == x:
                    list_out.append(out.data[i])
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
                full_out.append(self._alyxClient.get('/' + _ENDPOINTS[tab]))
                list_out.append([f[field_name] for f in full_out[-1]])
        if verbose:
            pprint(list_out)
        return list_out[0], full_out[0]

    def search(self, dataset_types=None, users=None, subjects=None, date_range=None,
               lab=None, details=False):
        """
        Applies a filter to the sessions (eid) table and returns a list of json dictionaries
         corresponding to sessions.

        :param dataset_types: list of dataset_types
        :type dataset_types: list of str
        :param users: a list of users
        :type users: list or str
        :param subjects: a list of subjects nickname
        :type subjects: list or str
        :param lab: a str or list of lab names
        :type lab: list or str
        :param date_range: list of 2 strings or list of 2 dates that define the range
        :type date_range: list
        :param details: default False, returns also the session details as per the REST response
        :type details: bool

        :return: list of eids, if details is True, also returns a list of json dictionaries,
         each entry corresponding to a matching session
        :rtype: list, list
        """
        # make sure string inputs are interpreted as lists
        def validate_input(inarg):
            return [inarg] if isinstance(inarg, str) else inarg

        dataset_types = validate_input(dataset_types)
        users = validate_input(users)
        subjects = validate_input(subjects)
        lab = validate_input(lab)
        # start creating the url
        url = '/sessions?'
        if dataset_types:
            url = url + 'dataset_types=' + ','.join(dataset_types)  # dataset_types query
        if users:
            url = url + '&users=' + ','.join(users)
        if subjects:
            url = url + '&subject=' + ','.join(subjects)
        if lab:
            url = url + '&lab=' + ','.join(lab)
        if date_range:
            date_range = _validate_date_range(date_range)
            url = url + '&date_range=' + ','.join(date_range)
        # implements the loading itself
        ses = self._alyxClient.get(url)
        eids = [s['url'] for s in ses]  # flattens session info
        eids = [e.split('/')[-1] for e in eids]  # remove url to make it portable
        if details:
            return eids, ses
        else:
            return eids

    @staticmethod
    def search_terms():
        """
        Returns possible search terms to be used as keywords in the one.search method.

        :return: a tuple containing possible search terms:
        :rtype: tuple
        """
        #  Implemented as a method to make sure this can't be changed
        return SEARCH_TERMS

    @staticmethod
    def setup():
        """
        Interactive command tool that populates parameter file for ONE IBL.
        """
        oneibl.params.setup()


def _validate_date_range(date_range):
    """
    Validates and arrange date range in a 2 elements list
    """
    if isinstance(date_range, str):
        date_range = [date_range, date_range]
    if len(date_range) == 1:
        date_range = [date_range[0], date_range[0]]
    return date_range

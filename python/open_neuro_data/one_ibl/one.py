import numpy as np
import os
from dataclasses import dataclass, field
import ibllib.webclient as wc
from ibllib.misc import is_uuid_string, pprint
from open_neuro_data.one import OneAbstract
import one_ibl.params as par


@dataclass
class SessionInfo:
    """
    Dataclass that provides dataset list, dataset_id, local_path, dataset_type, url and eid fields
    """
    data:  list = field(default_factory=list)
    dataset_id: list = field(default_factory=list)
    local_path: list = field(default_factory=list)
    dataset_type: list = field(default_factory=list)
    url: list = field(default_factory=list)
    eid: list = field(default_factory=list)


class ONE(OneAbstract):

    def __init__(self):
        # Init connection to the database
        self._alyxClient = wc.AlyxClient(username=par.ALYX_LOGIN, password=par.ALYX_PWD,
                                         base_url=par.BASE_URL)

    def load(self, eid, dataset_types=None, dclass_output=False):
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

        :return: List of numpy arrays matching the size of dataset_types parameter, OR
         a dataclass containing arrays and context data.
        :rtype: list, dict
        """
        # TODO: feature that downloads a list of datasets from a list of sessions,
        # TODO in this case force dictionary output
        # if the input as an UUID, add the beginning of URL to it
        if is_uuid_string(eid):
            eid = '/sessions/' + eid
        eid_str = eid[-36:]
        # get session json information as a dictionary from the alyx API
        ses = self._alyxClient.get(eid)
        # if no dataset_type is provided:
        # a) force the output to be a dictionary that provides context to the data
        # b) download all types that have a data url specified
        if not dataset_types:
            dclass_output = True
            dataset_types = [d['dataset_type'] for d in ses['data_dataset_session_related']
                             if d['data_url']]
        # loop over each dataset related to the session ID and get list of files urls
        session_dtypes = [d['dataset_type'] for d in ses['data_dataset_session_related']]
        out = SessionInfo()
        # this first loop only downloads the file to ease eventual refactoring
        for ind, dt in enumerate(dataset_types):
            for [i, sdt] in enumerate(session_dtypes):
                if sdt == dt:
                    urlstr = ses['data_dataset_session_related'][i]['data_url']
                    fil = wc.http_download_file(urlstr,
                                                username=par.HTTP_DATA_SERVER_LOGIN,
                                                password=par.HTTP_DATA_SERVER_PWD)
                    out.eid.append(eid_str)
                    out.dataset_type.append(dt)
                    out.url.append(urlstr)
                    out.local_path.append(fil)
                    out.dataset_id.append(ses['data_dataset_session_related'][i]['id'])
                    out.data.append([])
        # then another loop over files and load them in numpy. If not npy, just pass empty list
        for ind, fil in enumerate(out.local_path):  # this is where I miss switch case
            if fil and os.path.splitext(fil)[1] == '.npy':
                out.data[ind] = np.load(file=fil)
            if fil and os.path.splitext(fil)[1] == '.json':
                pass  # FIXME would be nice to implement json read but param from matlab RIG fails
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

    def list(self, table=None, verbose=False):
        """
        Queries the database for a list of 'users' and/or 'dataset-types' and/or 'subjects' fields

        :param table: the table (s) to query among: 'dataset-types','users'
         and 'subjects'; if empty or None assumes all tables
        :type table: str, list
        :param verbose: [False] prints the list in the current window
        :type verbose: bool

        :return: list of names to query, list of full description in json serialized format
        :rtype: list, list
        """
        tlist = ('dataset-types', 'users', 'subjects')
        field = ('name', 'username', 'nickname')
        if not table:
            table = tlist
        if isinstance(table, str):
            table = [table]
        full_out = []
        list_out = []
        for ind, tab in enumerate(tlist):
            if tab in table:
                full_out.append(self._alyxClient.get('/' + tab))
                list_out.append([f[field[ind]] for f in full_out[-1]])
        if verbose:
            pprint(list_out)
        if len(table) == 1:
            return list_out[0], full_out[0]
        else:
            return list_out, full_out

    def search(self, dataset_types=None, users=None, subject='', date_range=None):
        """
        Applies a filter to the sessions (eid) table and returns a list of json dictionaries
         corresponding to sessions.

        :param dataset_types: list of dataset_types
        :type dataset_types: list
        :param users: a list of users
        :type users: list
        :param subject: the subject nickname
        :type subject: str
        :param date_range: list of 2 strings or list of 2 dates that define the range
        :type date_range: list

        :return: list of eids
         list of json dictionaries, each entry corresponding to a matching session
        :rtype: list, list
        """
        # TODO add a lab field in the session table of ALyx to add as a query
        # make sure string inputs are interpreted as lists for dataset types and users
        dataset_types = [dataset_types] if isinstance(dataset_types, str) else dataset_types
        users = [users] if isinstance(users, str) else users
        # start creating the url
        url = '/sessions?'
        if dataset_types:
            url = url + 'dataset_types=' + ','.join(dataset_types)  # dataset_types query
        if users:
            url = url + '&users=' + ','.join(users)
        if subject:
            url = url + '&subject=' + subject
        if date_range:
            url = url + '&date_range=' + ','.join(date_range)
        # implements the loading itself
        ses = self._alyxClient.get(url)
        return [s['url'] for s in ses], ses

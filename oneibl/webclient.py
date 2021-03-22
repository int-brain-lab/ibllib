import json
import logging
import math
import os
import re
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import hashlib

import requests

from ibllib.misc import pprint, print_progress

SDSC_ROOT_PATH = PurePosixPath('/mnt/ibl')
_logger = logging.getLogger('ibllib')


class _PaginatedResponse(Mapping):
    """
    This class allows to emulate a list from a paginated response.
    Provides cache functionality
    PaginatedResponse(alyx, response)
    """

    def __init__(self, alyx, rep):
        self.alyx = alyx
        self.count = rep['count']
        self.limit = len(rep['results'])
        # warning: the offset and limit filters are not necessarily the last ones
        lquery = [q for q in rep['next'].split('&')
                  if not (q.startswith('offset=') or q.startswith('limit='))]
        self.query = '&'.join(lquery)
        # init the cache, list with None with count size
        self._cache = [None for _ in range(self.count)]
        # fill the cache with results of the query
        for i in range(self.limit):
            self._cache[i] = rep['results'][i]

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        if self._cache[item] is None:
            offset = self.limit * math.floor(item / self.limit)
            query = f'{self.query}&limit={self.limit}&offset={offset}'
            res = self.alyx._generic_request(requests.get, query)
            for i, r in enumerate(res['results']):
                self._cache[i + offset] = res['results'][i]
        return self._cache[item]

    def __iter__(self):
        for i in range(self.count):
            yield self.__getitem__(i)


def sdsc_globus_path_from_dataset(dset):
    """
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    Returns SDSC globus file path from a dset record or a list of dsets records from REST
    """
    return _path_from_dataset(dset, root_path=PurePosixPath('/'), repository=None, uuid=True)


def globus_path_from_dataset(dset, repository=None, uuid=False):
    """
    Returns local one file path from a dset record or a list of dsets records from REST
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param repository: (optional) repository name of the file record (if None, will take
     the first filerecord with an URL)
    """
    return _path_from_dataset(dset, root_path=PurePosixPath('/'), repository=repository, uuid=uuid)


def one_path_from_dataset(dset, one_cache):
    """
    Returns local one file path from a dset record or a list of dsets records from REST
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param one_cache: the one cache directory
    """
    return _path_from_dataset(dset, root_path=one_cache, uuid=False)


def sdsc_path_from_dataset(dset, root_path=SDSC_ROOT_PATH):
    """
    Returns sdsc file path from a dset record or a list of dsets records from REST
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param root_path: (optional) the prefix path such as one download directory or sdsc root
    """
    return _path_from_dataset(dset, root_path=root_path, uuid=True)


def _path_from_dataset(dset, root_path=None, repository=None, uuid=False):
    """
    returns the local file path from a dset record from a REST query
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param root_path: (optional) the prefix path such as one download directory or sdsc root
    :param repository:
    :param uuid: (optional bool) if True, will add UUID before the file extension
    :return: Path or list of Path
    """
    if isinstance(dset, list):
        return [_path_from_dataset(d) for d in dset]
    if repository:
        fr = next((fr for fr in dset['file_records'] if fr['data_repository'] == repository))
    else:
        fr = next((fr for fr in dset['file_records'] if fr['data_url']))
    uuid = dset['url'][-36:] if uuid else None
    return _path_from_filerecord(fr, root_path=root_path, uuid=uuid)


def _path_from_filerecord(fr, root_path=SDSC_ROOT_PATH, uuid=None):
    """
    Returns a data file Path constructed from an Alyx file record.  The Path type returned
    depends on the type of root_path: If root_path is a string a Path object is returned,
    otherwise if the root_path is a PurePath, the same path type is returned.
    :param fr: An Alyx file record dict
    :param root_path: An optional root path
    :param uuid: An optional UUID to add to the file name
    :return: A filepath as a pathlib object
    """
    import alf.io
    if isinstance(fr, list):
        return [_path_from_filerecord(f) for f in fr]
    repo_path = fr['data_repository_path']
    repo_path = repo_path[repo_path.startswith('/'):]  # remove starting / if any
    # repo_path = (p := fr['data_repository_path'])[p[0] == '/':]  # py3.8 Remove slash at start
    file_path = PurePosixPath(repo_path, fr['relative_path'])
    if root_path:
        # NB: By checking for string we won't cast any PurePaths
        if isinstance(root_path, str):
            root_path = Path(root_path)
        file_path = root_path / file_path
    if uuid:
        file_path = alf.io.add_uuid_string(file_path, uuid)
    return file_path


def http_download_file_list(links_to_file_list, **kwargs):
    """
    Downloads a list of files from the flat Iron from a list of links.
    Same options behaviour as http_download_file

    :param links_to_file_list: list of http links to files.
    :type links_to_file_list: list

    :return: (list) a list of the local full path of the downloaded files.
    """
    file_names_list = []
    for link_str in links_to_file_list:
        file_names_list.append(http_download_file(link_str, **kwargs))
    return file_names_list


def http_download_file(full_link_to_file, chunks=None, *, clobber=False,
                       username='', password='', cache_dir='', return_md5=False, headers=None):
    """
    :param full_link_to_file: http link to the file.
    :type full_link_to_file: str
    :param clobber: [False] If True, force overwrite the existing file.
    :type clobber: bool
    :param username: [''] authentication for password protected file server.
    :type username: str
    :param password: [''] authentication for password protected file server.
    :type password: str
    :param cache_dir: [''] directory in which files are cached; defaults to user's
     Download directory.
    :param: headers: [{}] additional headers to add to the request (auth tokens etc..)
    :type cache_dir: str

    :return: (str) a list of the local full path of the downloaded files.
    """
    from ibllib.io import hashfile
    if not full_link_to_file:
        return ''

    # default cache directory is the home dir
    if not cache_dir:
        cache_dir = str(Path.home().joinpath("Downloads"))

    # This is the local file name
    file_name = str(cache_dir) + os.sep + os.path.basename(full_link_to_file)

    # do not overwrite an existing file unless specified
    if not clobber and os.path.exists(file_name):
        return (file_name, hashfile.md5(file_name)) if return_md5 else file_name

    # This should be the base url you wanted to access.
    baseurl = os.path.split(str(full_link_to_file))[0]

    # Create a password manager
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    if (len(password) != 0) & (len(username) != 0):
        manager.add_password(None, baseurl, username, password)

    # Create an authentication handler using the password manager
    auth = urllib.request.HTTPBasicAuthHandler(manager)

    # Create an opener that will replace the default urlopen method on further calls
    opener = urllib.request.build_opener(auth)
    urllib.request.install_opener(opener)

    # Support for partial download.
    req = urllib.request.Request(full_link_to_file)
    if chunks is not None:
        first_byte, n_bytes = chunks
        req.add_header("Range", "bytes=%d-%d" % (first_byte, first_byte + n_bytes - 1))

    # add additional headers
    if headers is not None:
        for k in headers:
            req.add_header(k, headers[k])

    # Open the url and get the length
    try:
        u = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        _logger.error(f"{str(e)} {full_link_to_file}")
        raise e

    file_size = int(u.getheader('Content-length'))

    print(f"Downloading: {file_name} Bytes: {file_size}")
    file_size_dl = 0
    block_sz = 8192 * 64 * 8

    md5 = hashlib.md5()
    f = open(file_name, 'wb')
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        if return_md5:
            md5.update(buffer)
        print_progress(file_size_dl, file_size, prefix='', suffix='')
    f.close()

    return (file_name, md5.hexdigest()) if return_md5 else file_name


def file_record_to_url(file_records, urls=[]):
    """
    Translate a Json dictionary to an usable http url for downlading files.

    :param file_records: json containing a 'data_url' field
    :type file_records: dict
    :param urls: a list of strings containing previous data_urls on which new urls
     will be appended
    :type urls: list

    :return: urls: (list) a list of strings representing full data urls
    """
    for fr in file_records:
        if fr['data_url'] is not None:
            urls.append(fr['data_url'])
    return urls


def dataset_record_to_url(dataset_record):
    """
    Extracts a list of files urls from a list of dataset queries.

    :param dataset_record: dataset Json from a rest request.
    :type dataset_record: list

    :return: (list) a list of strings representing files urls corresponding to the datasets records
    """
    urls = []
    if type(dataset_record) is dict:
        dataset_record = [dataset_record]
    for ds in dataset_record:
        urls = file_record_to_url(ds['file_records'], urls)
    return urls


class UniqueSingletons(type):
    _instances: list = []

    def __call__(cls, *args, **kwargs):
        # print('args', args, '\nkwargs', kwargs)
        for inst in UniqueSingletons._instances:
            if cls in inst and inst.get(cls, None).get('args') == (args, kwargs):
                return inst[cls].get('instance')

        new_instance = super(UniqueSingletons, cls).__call__(*args, **kwargs)
        # Optional rerun of constructor
        # new_instance.__init__(*args, **kwargs)
        new_instance_record = {
            cls: {'args': (args, kwargs), 'instance': new_instance}
        }
        UniqueSingletons._instances.append(new_instance_record)

        return new_instance


class AlyxClient(metaclass=UniqueSingletons):
    """
    Class that implements simple GET/POST wrappers for the Alyx REST API
    http://alyx.readthedocs.io/en/latest/api.html
    """

    def __init__(self, **kwargs):
        """
        Create a client instance that allows to GET and POST to the Alyx server
        For oneibl, constructor attempts to authenticate with credentials in params.py
        For standalone cases, AlyxClient(username='', password='', base_url='')

        :param username: Alyx database user
        :type username: str
        :param password: Alyx database password
        :type password: str
        :param base_url: Alyx server address, including port and protocol
        :type base_url: str
        """
        self.authenticate(**kwargs)
        self._headers['Accept'] = 'application/coreapi+json'
        self._rest_schemes = self.get('/docs')
        # the mixed accept application may cause errors sometimes, only necessary for the docs
        self._headers['Accept'] = 'application/json'
        self._obj_id = id(self)

    def _generic_request(self, reqfunction, rest_query, data=None, files=None):
        # makes sure the base url is the one from the instance
        rest_query = rest_query.replace(self._base_url, '')
        if not rest_query.startswith('/'):
            rest_query = '/' + rest_query
        _logger.debug(f"{self._base_url + rest_query}, headers: {self._headers}")
        headers = self._headers.copy()
        if files is None:
            data = json.dumps(data) if isinstance(data, dict) or isinstance(data, list) else data
            headers['Content-Type'] = 'application/json'
        r = reqfunction(self._base_url + rest_query, stream=True, headers=headers,
                        data=data, files=files)
        if r and r.status_code in (200, 201):
            return json.loads(r.text)
        elif r and r.status_code == 204:
            return
        else:
            _logger.error(self._base_url + rest_query)
            _logger.error(r.text)
            raise(requests.HTTPError(r))

    def authenticate(self, username='', password='', base_url=''):
        """
        Gets a security token from the Alyx REST API to create requests headers.
        Credentials are in the params_secret_template.py file

        :param username: Alyx database user
        :type username: str
        :param password: Alyx database password
        :type password: str
        :param base_url: Alyx server address, including port and protocol
        :type base_url: str
        """
        self._base_url = base_url
        rep = requests.post(base_url + '/auth-token',
                            data=dict(username=username, password=password))
        # Assign token or raise exception on internal server error
        self._token = rep.json() if rep.ok else rep.raise_for_status()
        if not (list(self._token.keys()) == ['token']):
            _logger.error(rep)
            raise Exception('Alyx authentication error. Check your credentials')
        self._headers = {
            'Authorization': 'Token {}'.format(list(self._token.values())[0]),
            'Accept': 'application/json'}

    def delete(self, rest_query):
        """
        Sends a DELETE request to the Alyx server. Will raise an exception on any status_code
        other than 200, 201.

        :param rest_query: examples:
         '/weighings/c617562d-c107-432e-a8ee-682c17f9e698'
         'https://test.alyx.internationalbrainlab.org/weighings/c617562d-c107-432e-a8ee-682c17f9e698'.
        :type rest_query: str

        :return: (dict/list) json interpreted dictionary from response
        """
        return self._generic_request(requests.delete, rest_query)

    def download_file(self, url, **kwargs):
        """
        Downloads a file on the Alyx server from a filerecord REST field URL
        :param url: full url of the file
        :param kwargs: webclient.http_download_file parameters
        :return: local path of downloaded file
        """
        return http_download_file(url, headers=self._headers, **kwargs)

    def get(self, rest_query):
        """
        Sends a GET request to the Alyx server. Will raise an exception on any status_code
        other than 200, 201.
        For the dictionary contents and list of endpoints, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: example: '/sessions?user=Hamish'.
        :type rest_query: str

        :return: (dict/list) json interpreted dictionary from response
        """
        rep = self._generic_request(requests.get, rest_query)
        _logger.debug(rest_query)
        if isinstance(rep, dict) and list(rep.keys()) == ['count', 'next', 'previous', 'results']:
            if len(rep['results']) < rep['count']:
                rep = _PaginatedResponse(self, rep)
            else:
                rep = rep['results']
        return rep

    def patch(self, rest_query, data=None, files=None):
        """
        Sends a PATCH request to the Alyx server.
        For the dictionary contents, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: (required)the endpoint as full or relative URL
        :type rest_query: str
        :param data: json encoded string or dictionary (cf.requests)
        :type data: None, dict or str
        :param files: dictionary / tuple (cf.requests)

        :return: response object
        """
        return self._generic_request(requests.patch, rest_query, data=data, files=files)

    def post(self, rest_query, data=None, files=None):
        """
        Sends a POST request to the Alyx server.
        For the dictionary contents, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: (required)the endpoint as full or relative URL
        :type rest_query: str
        :param data: dictionary or json encoded string
        :type data: None, dict or str
        :param files: dictionary / tuple (cf.requests)

        :return: response object
        """
        return self._generic_request(requests.post, rest_query, data=data, files=files)

    def put(self, rest_query, data=None, files=None):
        """
        Sends a PUT request to the Alyx server.
        For the dictionary contents, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: (required)the endpoint as full or relative URL
        :type rest_query: str
        :param data: dictionary or json encoded string
        :type data: None, dict or str
        :param files: dictionary / tuple (cf.requests)

        :return: response object
        """
        return self._generic_request(requests.put, rest_query, data=data, files=files)

    def rest(self, url=None, action=None, id=None, data=None, files=None, **kwargs):
        """
        alyx_client.rest(): lists endpoints
        alyx_client.rest(endpoint): lists actions for endpoint
        alyx_client.rest(endpoint, action): lists fields and URL

        Example with a rest endpoint with all actions

        >>> alyx.client.rest('subjects', 'list')
            alyx.client.rest('subjects', 'list', field_filter1='filterval')
            alyx.client.rest('subjects', 'create', data=sub_dict)
            alyx.client.rest('subjects', 'read', id='nickname')
            alyx.client.rest('subjects', 'update', id='nickname', data=sub_dict)
            alyx.client.rest('subjects', 'partial_update', id='nickname', data=sub_dict)
            alyx.client.rest('subjects', 'delete', id='nickname')
            alyx.client.rest('notes', 'create', data=nd, files={'image': open(image_file, 'rb')})

        :param url: endpoint name
        :param action: 'list', 'create', 'read', 'update', 'partial_update', 'delete'
        :param id: lookup string for actions 'read', 'update', 'partial_update', and 'delete'
        :param data: data dictionary for actions 'update', 'partial_update' and 'create'
        :param files: if file upload
        :param ``**kwargs``: filter as per the Alyx REST documentation
            cf. https://alyx.internationalbrainlab.org/docs/
        :return: list of queried dicts ('list') or dict (other actions)
        """
        # if endpoint is None, list available endpoints
        if not url:
            pprint([k for k in self._rest_schemes.keys() if not k.startswith('_') and k])
            return
        # remove beginning slash if any
        if url.startswith('/'):
            url = url[1:]
        # and split to the next slash or question mark
        endpoint = re.findall("^/*[^?/]*", url)[0].replace('/', '')
        # make sure the queryied endpoint exists, if not throw an informative error
        if endpoint not in self._rest_schemes.keys():
            av = [k for k in self._rest_schemes.keys() if not k.startswith('_') and k]
            raise ValueError('REST endpoint "' + endpoint + '" does not exist. Available ' +
                             'endpoints are \n       ' + '\n       '.join(av))
        endpoint_scheme = self._rest_schemes[endpoint]
        # on a filter request, override the default action parameter
        if '?' in url:
            action = 'list'
        # if action is None, list available actions for the required endpoint
        if not action:
            pprint(list(endpoint_scheme.keys()))
            return
        # make sure the the desired action exists, if not throw an informative error
        if action not in endpoint_scheme.keys():
            raise ValueError('Action "' + action + '" for REST endpoint "' + endpoint + '" does ' +
                             'not exist. Available actions are: ' +
                             '\n       ' + '\n       '.join(endpoint_scheme.keys()))
        # the actions below require an id in the URL, warn and help the user
        if action in ['read', 'update', 'partial_update', 'delete'] and not id:
            _logger.warning('REST action "' + action + '" requires an ID in the URL: ' +
                            endpoint_scheme[action]['url'])
            return
        # the actions below require a data dictionary, warn and help the user with fields list
        if action in ['create', 'update', 'partial_update'] and not data:
            pprint(endpoint_scheme[action]['fields'])
            for act in endpoint_scheme[action]['fields']:
                print("'" + act['name'] + "': ...,")
            _logger.warning('REST action "' + action + '" requires a data dict with above keys')
            return

        if action == 'list':
            # list doesn't require id nor
            assert(endpoint_scheme[action]['action'] == 'get')
            # add to url data if it is a string
            if id:
                # this is a special case of the list where we query an uuid. Usually read is better
                if 'django' in kwargs.keys():
                    kwargs['django'] = kwargs['django'] + ','
                else:
                    kwargs['django'] = ""
                kwargs['django'] = f"{kwargs['django']}pk,{id}"
            # otherwise, look for a dictionary of filter terms
            if kwargs:
                url += '?'
                for k in kwargs.keys():
                    if isinstance(kwargs[k], str):
                        query = kwargs[k]
                    elif isinstance(kwargs[k], list):
                        query = ','.join(kwargs[k])
                    else:
                        query = str(kwargs[k])
                    url = url + f"&{k}=" + query
            return self.get('/' + url)
        if action == 'read':
            assert(endpoint_scheme[action]['action'] == 'get')
            return self.get('/' + endpoint + '/' + id.split('/')[-1])
        elif action == 'create':
            assert(endpoint_scheme[action]['action'] == 'post')
            return self.post('/' + endpoint, data=data, files=files)
        elif action == 'delete':
            assert(endpoint_scheme[action]['action'] == 'delete')
            return self.delete('/' + endpoint + '/' + id.split('/')[-1])
        elif action == 'partial_update':
            assert(endpoint_scheme[action]['action'] == 'patch')
            return self.patch('/' + endpoint + '/' + id.split('/')[-1], data=data, files=files)
        elif action == 'update':
            assert(endpoint_scheme[action]['action'] == 'put')
            return self.put('/' + endpoint + '/' + id.split('/')[-1], data=data, files=files)

    # JSON field interface convenience methods
    def _check_inputs(self, endpoint: str) -> None:
        # make sure the queryied endpoint exists, if not throw an informative error
        if endpoint not in self._rest_schemes.keys():
            av = [k for k in self._rest_schemes.keys() if not k.startswith('_') and k]
            raise ValueError('REST endpoint "' + endpoint + '" does not exist. Available ' +
                             'endpoints are \n       ' + '\n       '.join(av))
        return

    def json_field_write(
        self,
        endpoint: str = None,
        uuid: str = None,
        field_name: str = None,
        data: dict = None
    ) -> dict:
        """json_field_write [summary]
        Write data to WILL NOT CHECK IF DATA EXISTS
        NOTE: Destructive write!

        :param endpoint: Valid alyx endpoint, defaults to None
        :type endpoint: str, optional
        :param uuid: uuid or lookup name for endpoint
        :type uuid: str, optional
        :param field_name: Valid json field name, defaults to None
        :type field_name: str, optional
        :param data: data to write to json field, defaults to None
        :type data: dict, optional
        :return: Written data dict
        :rtype: dict
        """
        self._check_inputs(endpoint)
        # Prepare data to patch
        patch_dict = {field_name: data}
        # Upload new extended_qc to session
        ret = self.rest(endpoint, "partial_update", id=uuid, data=patch_dict)
        return ret[field_name]

    def json_field_update(
        self,
        endpoint: str = None,
        uuid: str = None,
        field_name: str = 'json',
        data: dict = None
    ) -> dict:
        """json_field_update
        Non destructive update of json field of endpoint for object
        Will update the field_name of the object with pk = uuid of given endpoint
        If data has keys with the same name of existing keys it will squash the old
        values (uses the dict.update() method)

        Example:
        one.alyx.json_field_update("sessions", "eid_str", "extended_qc", {"key": value})

        :param endpoint: endpoint to hit
        :type endpoint: str
        :param uuid: uuid or lookup name of object
        :type uuid: str
        :param field_name: name of the json field
        :type field_name: str
        :param data: dictionary with fields to be updated
        :type data: dict
        :return: new patched json field contents
        :rtype: dict
        """
        self._check_inputs(endpoint)
        # Load current json field contents
        current = self.rest(endpoint, "read", id=uuid)[field_name]
        if current is None:
            current = {}

        if not isinstance(current, dict):
            _logger.warning(
                f"Current json field {field_name} does not contains a dict, aborting update"
            )
            return current

        # Patch current dict with new data
        current.update(data)
        # Prepare data to patch
        patch_dict = {field_name: current}
        # Upload new extended_qc to session
        ret = self.rest(endpoint, "partial_update", id=uuid, data=patch_dict)
        return ret[field_name]

    def json_field_remove_key(
        self,
        endpoint: str = None,
        uuid: str = None,
        field_name: str = 'json',
        key: str = None
    ) -> dict:
        """json_field_remove_key
        Will remove inputted key from json field dict and reupload it to Alyx.
        Needs endpoint, uuid and json field name

        :param endpoint: endpoint to hit, defaults to None
        :type endpoint: str, optional
        :param uuid: uuid or lookup name for endpoint
        :type uuid: str, optional
        :param field_name: json field name of object, defaults to None
        :type field_name: str, optional
        :param key: key name of dictionary inside object, defaults to None
        :type key: str, optional
        :return: returns new content of json field
        :rtype: dict
        """
        self._check_inputs(endpoint)
        current = self.rest(endpoint, "read", id=uuid)[field_name]
        # If no contents, cannot remove key, return
        if current is None:
            return current
        # if contents are not dict, cannot remove key, return contents
        if isinstance(current, str):
            _logger.warning(f"Cannot remove key {key} content of json field is of type str")
            return current
        # If key not present in contents of json field cannot remove key, return contents
        if current.get(key, None) is None:
            _logger.warning(
                f"{key}: Key not found in endpoint {endpoint} field {field_name}"
            )
            return current
        _logger.info(f"Removing key from dict: '{key}'")
        current.pop(key)
        # Re-write contents without removed key
        written = self.json_field_write(
            endpoint=endpoint, uuid=uuid, field_name=field_name, data=current
        )
        return written

    def json_field_delete(
        self, endpoint: str = None, uuid: str = None, field_name: str = None
    ) -> None:
        self._check_inputs(endpoint)
        _ = self.rest(endpoint, "partial_update", id=uuid, data={field_name: None})
        return _[field_name]

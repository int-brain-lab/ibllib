import urllib.request
import os
from pathlib import Path
import requests
import json
from ibllib.misc import pprint

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


def http_download_file(full_link_to_file, *, clobber=False,
                       username='', password='', cache_dir='', verbose=True):
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
    :type cache_dir: str
    :param verbose: [True] displays a message for each download.
    :type verbose: bool

    :return: (str) a list of the local full path of the downloaded files.
    """
    if not full_link_to_file:
        return ''

    # default cache directory is the home dir
    if len(cache_dir) == 0:
        cache_dir = str(Path.home()) + os.sep + "Downloads"

    # This is the local file name
    file_name = cache_dir + os.sep + os.path.basename(full_link_to_file)

    # do not overwrite an existing file unless specified
    if not clobber and os.path.exists(file_name):
        return file_name

    # This should be the base url you wanted to access.
    baseurl = os.path.split(full_link_to_file)[0]

    # Create a password manager
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    if (len(password) != 0) & (len(username) != 0):
        manager.add_password(None, baseurl, username, password)

    # Create an authentication handler using the password manager
    auth = urllib.request.HTTPBasicAuthHandler(manager)

    # Create an opener that will replace the default urlopen method on further calls
    opener = urllib.request.build_opener(auth)
    urllib.request.install_opener(opener)

    # Open the url and get the length
    u = urllib.request.urlopen(full_link_to_file)
    file_size = int(u.getheader('Content-length'))

    if verbose:
        print("Downloading: %s Bytes: %s" % (file_name, file_size))
    file_size_dl = 0
    block_sz = 8192 * 64 * 8
    f = open(file_name, 'wb')
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        if verbose:
            print(status)
    f.close()

    return file_name


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


class AlyxClient:
    """
    Class that implements simple GET/POST wrappers for the Alyx REST API
    http://alyx.readthedocs.io/en/latest/api.html
    """
    _token = ''
    _headers = ''
    _rest_schemes = ''

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

    def _generic_request(self, reqfunction, rest_query, data=None):
        rest_query = rest_query.replace(self._base_url, '')
        r = reqfunction(self._base_url + rest_query, stream=True, headers=self._headers,
                         data=data)
        if r and r.status_code in (200, 201):
            return json.loads(r.text)
        elif r and r.status_code == 204:
            return
        else:
            print(self._base_url + rest_query)
            raise Exception(r)

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
        self._token = rep.json()
        if not (list(self._token.keys()) == ['token']):
            print(rep)
            raise Exception('Alyx authentication error. Check your ./oneibl/params.py and'
                            './oneibl/params_secret.py')
        self._headers = {
            'Authorization': 'Token {}'.format(list(self._token.values())[0]),
            'Accept': 'application/json',
            'Content-Type': 'application/json'}

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
        return self._generic_request(requests.get, rest_query)

    def post(self, rest_query, data=None):
        """
        Sends a POST request to the Alyx server.
        For the dictionary contents, refer to:
        https://alyx.internationalbrainlab.org/docs
        The preferred and safer way to interact with the REST API is by
        using the AlyxClient.rest method

        :param rest_query: (required)the endpoint as full or relative URL
        :type rest_query: str
        :param data: json encoded string
        :type data: None, dict or str

        :return: response object
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        return self._generic_request(requests.post, rest_query, data=data)

    def rest(self, endpoint=None, action=None, data=None):
        """
        alyx_client.rest()
        alyx_client.rest("sessions")
        lab_info = alyx_client.rest('labs', 'read', 'mainenlab')
        OR
        lab_info = alyx_client.rest('labs', 'read', 'https://test.alyx.internationalbrainlab.org/labs/mainenlab')

        :param endpoint:
        :param action:
        :param data:
        :return:
        """
        # if endpoint is None, list available endpoints
        if not endpoint:
            pprint([k for k in self._rest_schemes.keys() if not k.startswith('_') and k])
            return
        # allow the user to enter an endpoint beginning with a slash
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        # make sure the queryied endpoint exists, if not throw an informative error
        if endpoint not in self._rest_schemes.keys():
            av = [k for k in self._rest_schemes.keys() if not k.startswith('_') and k]
            raise ValueError('REST endpoint "' + endpoint + '" does not exist. Available ' +
                             'endpoints are \n       ' + '\n       '.join(av))
        endpoint_scheme = self._rest_schemes[endpoint]
        # if action is None, list available actions for the required endpoint
        if not action:
            pprint(list(endpoint_scheme.keys()))
            return
        # make sure the the desired action exists, if not throw an informative error
        if action not in endpoint_scheme.keys():
            raise ValueError('Action "' + action + '" for REST endpoint "' + endpoint + '" does ' +
                             'not exist. Available actions are: ' +
                             '\n       ' + '\n       '.join(endpoint_scheme.keys()))
        # if there is no data (except for list), show the user a list of fields
        if action != 'list' and not data:
            pprint(endpoint_scheme[action]['fields'])
            return
        if action == 'list':
            assert(endpoint_scheme[action]['action'] == 'get')
            return self.get('/' + endpoint)
        if action == 'read':
            assert(endpoint_scheme[action]['action'] == 'get')
            return self.get('/' + endpoint + '/' + data.split('/')[-1])
        elif action == 'create':
            assert(endpoint_scheme[action]['action'] == 'post')
            return self.post('/' + endpoint, data)
        elif action == 'delete':
            assert(endpoint_scheme[action]['action'] == 'delete')
            return self.delete('/' + endpoint + '/' + data.split('/')[-1])
        # TODO BELOW: implement and unit-tests
        elif action == 'update':
            assert(endpoint_scheme[action]['action'] == 'put')
            pass
        elif action == 'partial_update':
            assert(endpoint_scheme[action]['action'] == 'patch')
            pass



from functools import wraps
import json
import logging
import os
import os.path as op
from pathlib import Path
import tempfile

import globus_sdk as globus
from alf.io import is_uuid_string
from ibllib.io import params


logger = logging.getLogger(__name__)


# ibllib util functions
# ------------------------------------------------------------------------------------------------

def _login(globus_client_id, refresh_tokens=False):

    client = globus.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=refresh_tokens)

    authorize_url = client.oauth2_get_authorize_url()
    print('Please go to this URL and login: {0}'.format(authorize_url))
    auth_code = input(
        'Please enter the code you get after login here: ').strip()

    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

    token = dict(refresh_token=globus_transfer_data['refresh_token'],
                 transfer_token=globus_transfer_data['access_token'],
                 expires_at_s=globus_transfer_data['expires_at_seconds'],
                 )
    return token


def login(globus_client_id):
    token = _login(globus_client_id, refresh_tokens=False)
    authorizer = globus.AccessTokenAuthorizer(token['transfer_token'])
    tc = globus.TransferClient(authorizer=authorizer)
    return tc


def setup(globus_client_id, str_app='globus'):
    # Lookup and manage consents there
    # https://auth.globus.org/v2/web/consents
    gtok = _login(globus_client_id, refresh_tokens=True)
    params.write(str_app, gtok)


def login_auto(globus_client_id, str_app='globus'):
    token = params.read(str_app)
    required_fields = {'refresh_token', 'transfer_token', 'expires_at_s'}
    if not (token and required_fields.issubset(token.as_dict())):
        raise ValueError("Token file doesn't exist, run ibllib.io.globus.setup first")
    client = globus.NativeAppAuthClient(globus_client_id)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus.RefreshTokenAuthorizer(token.refresh_token, client)
    return globus.TransferClient(authorizer=authorizer)


# Login functions coming from alyx
# ------------------------------------------------------------------------------------------------

def globus_client_id():
    return params.read('one_params').GLOBUS_CLIENT_ID


def get_config_path(path=''):
    path = op.expanduser(op.join('~/.ibllib', path))
    os.makedirs(op.dirname(path), exist_ok=True)
    return path


def create_globus_client():
    client = globus.NativeAppAuthClient(globus_client_id())
    client.oauth2_start_flow(refresh_tokens=True)
    return client


def create_globus_token():
    client = create_globus_client()
    print('Please go to this URL and login: {0}'
          .format(client.oauth2_get_authorize_url()))
    get_input = getattr(__builtins__, 'raw_input', input)
    auth_code = get_input('Please enter the code here: ').strip()
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

    data = dict(transfer_rt=globus_transfer_data['refresh_token'],
                transfer_at=globus_transfer_data['access_token'],
                expires_at_s=globus_transfer_data['expires_at_seconds'],
                )
    path = get_config_path('globus-token.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_globus_transfer_rt():
    path = get_config_path('globus-token.json')
    if not op.exists(path):
        return
    with open(path, 'r') as f:
        return json.load(f).get('transfer_rt', None)


def globus_transfer_client():
    transfer_rt = get_globus_transfer_rt()
    if not transfer_rt:
        create_globus_token()
        transfer_rt = get_globus_transfer_rt()
    client = create_globus_client()
    authorizer = globus.RefreshTokenAuthorizer(transfer_rt, client)
    tc = globus.TransferClient(authorizer=authorizer)
    return tc


# Globus wrapper
# ------------------------------------------------------------------------------------------------

def local_endpoint():
    path = Path.home().joinpath(".globusonline/lta/client-id.txt")
    if path.exists():
        return path.read_text()


ENDPOINTS = {
    'test': ('2bfac104-12b1-11ea-bea5-02fcc9cdd752', '/~/mnt/xvdf/Data/'),
    'flatiron': ('15f76c0c-10ee-11e8-a7ed-0a448319c2f8', '/~/'),
    'cortexlab': ('9426178c-dc39-11e9-b5de-0ef30f6b83a8', '/cortexlab/Subjects/'),
    'local': (local_endpoint(), '/~/ssd/ephys/globus/'),
}


def _remote_path(root, path=''):
    root = str(root)
    path = str(path)
    if not root.endswith('/'):
        root += '/'
    if path.startswith('/'):
        path = path[1:]
    path = root + path
    path = path.replace('//', '/')
    assert path.startswith(root)
    return path


def _split_file_path(path):
    path = str(path)
    assert not path.endswith('/')
    if '/' in path:
        i = path.rindex('/')
        parent = path[:i]
        filename = path[i + 1:]
    else:
        parent = ''
        filename = path
    return parent, filename


def _filename_size_matches(path_size, existing):
    path, size = path_size
    path = str(path)
    if size is None:
        return path in [fn for fn, sz in existing]
    else:
        assert size >= 0
        return (path, size) in existing


def _remove_uuid_from_filename(file_path):
    file_path = Path(file_path)
    name_parts = file_path.name.split('.')
    if len(name_parts) < 2 or not is_uuid_string(name_parts[-2]):
        return str(file_path)
    name_parts.pop(-2)
    return str(file_path.parent.joinpath('.'.join(name_parts)))


class Globus:
    """Wrapper for managing files on Globus endpoints."""

    def __init__(self):
        self._tc = globus_transfer_client()

    def ls(self, endpoint, path='', remove_uuid=False):
        """Return the list of (filename, filesize) in a given endpoint directory.

        NOTE: this function automatically removes the UUIDs from the filenames.

        """
        endpoint, root = ENDPOINTS.get(endpoint, (endpoint, ''))
        assert root
        path = _remote_path(root, path)
        out = []
        try:
            for entry in self._tc.operation_ls(endpoint, path=path):
                fn = entry['name']
                if remove_uuid:
                    fn = _remove_uuid_from_filename(fn)
                size = entry['size'] if entry['type'] == 'file' else None
                out.append((fn, size))
        except Exception as e:
            logger.error(str(e))
        return out

    def file_exists(self, endpoint, path, size=None, remove_uuid=False):
        """Return whether a given file exists on a given endpoint, optionally with a specified
        file size."""
        parent, filename = _split_file_path(path)
        existing = self.ls(endpoint, parent, remove_uuid=remove_uuid)
        return _filename_size_matches((filename, size), existing)

    def dir_contains_files(self, endpoint, dir_path, filenames, remove_uuid=False):
        """Return whether a directory contains a list of filenames. Returns a list of boolean,
        one for each input file."""
        files = self.ls(endpoint, dir_path, remove_uuid=remove_uuid)
        existing = [fn for fn, size in files]
        out = []
        for filename in filenames:
            out.append(filename in existing)
        return out

    def files_exist(self, endpoint, paths, sizes=None, remove_uuid=False):
        """Return whether a list of files exist on an endpoint, optionally with specified
        file sizes."""
        if not paths:
            return []
        parents = sorted(set(_split_file_path(path)[0] for path in paths))
        existing = []
        for parent in parents:
            filenames_sizes = self.ls(endpoint, parent, remove_uuid=remove_uuid)
            existing.extend([(parent + '/' + fn, size) for fn, size in filenames_sizes])
        if sizes is None:
            sizes = [None] * len(paths)
        return [_filename_size_matches((path, size), existing) for (path, size) in zip(paths, sizes)]

    def blocking(f):
        """Add two keyword arguments to a method, blocking (boolean) and timeout."""
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            blocking = kwargs.pop('blocking', None)
            timeout = kwargs.pop('timeout', None)
            task_id = f(self, *args, **kwargs)
            if blocking:
                return self.wait_task(task_id, timeout=timeout)
            else:
                return task_id
        return wrapped

    @blocking
    def rm(self, endpoint, path, blocking=False):
        """Delete a single file on an endpoint."""
        endpoint, root = ENDPOINTS.get(endpoint, (endpoint, ''))
        assert root
        path = _remote_path(root, path)

        ddata = globus.DeleteData(self._tc, endpoint, recursive=False)
        ddata.add_item(path)
        delete_result = self._tc.submit_delete(ddata)
        task_id = delete_result["task_id"]
        message = delete_result["message"]
        return task_id

    @blocking
    def move_files(
        self, source_endpoint, target_endpoint,
        source_paths, target_paths):
        """Move files from one endpoint to another."""
        source_endpoint, source_root = ENDPOINTS.get(source_endpoint, (source_endpoint, ''))
        target_endpoint, target_root = ENDPOINTS.get(target_endpoint, (target_endpoint, ''))

        source_paths = [_remote_path(source_root, str(_)) for _ in source_paths]
        target_paths = [_remote_path(target_root, str(_)) for _ in target_paths]

        tdata = globus.TransferData(
            self._tc, source_endpoint, target_endpoint, verify_checksum=True, sync_level='checksum',
        )
        for source_path, target_path in zip(source_paths, target_paths):
            tdata.add_item(source_path, target_path)
        response = self._tc.submit_transfer(tdata)
        task_id = response.get('task_id', None)
        message = response.get('message', None)
        return task_id

    def add_text_file(self, endpoint, path, contents):
        """Create a text file on a remote endpoint."""
        local = ENDPOINTS.get('local', None)
        if not local or not local[0]:
            raise IOError(
            "Can only add a text file on a remote endpoint "
            "if the current computer is a Globus endpoint")
        local_endpoint, local_root = local
        assert local_endpoint
        assert local_root
        local_root = Path(local_root.replace('/~', ''))
        assert local_root.exists()
        fn = '_tmp_text_file.txt'
        local_path = local_root / fn
        local_path.write_text(contents)
        task_id = self.move_files(local_endpoint, endpoint, [local_path], [path], blocking=True)
        os.remove(local_path)

    def wait_task(self, task_id, timeout=None):
        """Block until a Globus task finishes."""
        if timeout is None:
            timeout = 300
        i = 0
        while not self._tc.task_wait(task_id, timeout=1, polling_interval=1):
            print('.', end='')
            i += 1
            if i >= timeout:
                raise IOError(
                    "The task %s has not finished after %d seconds." % (task_id, timeout))

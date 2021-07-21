import abc
import ftplib
from pathlib import Path, PurePosixPath, WindowsPath
import subprocess
import logging
from getpass import getpass

import globus_sdk
import iblutil.io.params as iopar
from one.alf.spec import is_uuid_string
from one.alf.files import get_session_path, add_uuid_string
from one import params

from ibllib.io import globus
from oneibl.registration import register_dataset

_logger = logging.getLogger('ibllib')

FLAT_IRON_GLOBUS_ID = 'ab2d064c-413d-11eb-b188-0ee0d5d9299f'
FLATIRON_HOST = 'ibl.flatironinstitute.org'
FLATIRON_PORT = 61022
FLATIRON_USER = 'datauser'
FLATIRON_MOUNT = '/mnt/ibl'
FTP_HOST = 'test.alyx.internationalbrainlab.org'
FTP_PORT = 21
DMZ_REPOSITORY = 'ibl_patcher'  # in alyx, the repository name containing the patched filerecords


def _run_command(cmd, dry=True):
    _logger.info(cmd)
    if dry:
        return 0, '', ''
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info, error = p.communicate()
    if p.returncode != 0:
        _logger.error(error)
        raise RuntimeError(error)
    return p.returncode, info, error


class Patcher(abc.ABC):
    def __init__(self, one=None):
        assert one
        self.one = one

    def _patch_dataset(self, path, dset_id=None, dry=False, ftp=False):
        """
        This private methods gets the dataset information from alyx, computes the local
        and remote paths and initiates the file copy
        """
        path = Path(path)
        if dset_id is None:
            dset_id = path.name.split('.')[-2]
            if not is_uuid_string(dset_id):
                dset_id = None
        assert dset_id
        assert is_uuid_string(dset_id)
        assert path.exists()
        dset = self.one.alyx.rest('datasets', "read", id=dset_id)
        fr = next(fr for fr in dset['file_records'] if 'flatiron' in fr['data_repository'])
        remote_path = Path(fr['data_repository_path']).joinpath(fr['relative_path'])
        remote_path = add_uuid_string(remote_path, dset_id).as_posix()
        if remote_path.startswith('/'):
            full_remote_path = PurePosixPath(FLATIRON_MOUNT + remote_path)
        else:
            full_remote_path = PurePosixPath(FLATIRON_MOUNT, remote_path)
        if isinstance(path, WindowsPath) and not ftp:
            # On Windows replace drive map with Globus uri, e.g. C:/ -> /~/C/
            path = '/~/' + path.as_posix().replace(':', '')
        status = self._scp(path, full_remote_path, dry=dry)[0]
        return status

    def register_dataset(self, file_list, **kwargs):
        """
        Registers a set of files belonging to a session only on the server
        :param file_list: (list of pathlib.Path)
        :param created_by: (string) name of user in Alyx (defaults to 'root')
        :param repository: optional: (string) name of the server repository in Alyx
        :param versions: optional (list of strings): versions tags (defaults to ibllib version)
        :param dry: (bool) False by default
        :return:
        """
        return register_dataset(file_list, one=self.one, server_only=True, **kwargs)

    def register_datasets(self, file_list, **kwargs):
        """
        Same as register_dataset but works with files belonging to different sessions
        """
        register_dict = {}
        # creates a dictionary of sessions with one file list per session
        for f in file_list:
            session_path = get_session_path(f)
            label = '_'.join(session_path.parts[-3:])
            if label in register_dict:
                register_dict[label]['files'].append(f)
            else:
                register_dict[label] = {'session_path': session_path, 'files': [f]}
        responses = []
        nses = len(register_dict)
        for i, label in enumerate(register_dict):
            _files = register_dict[label]['files']
            _logger.info(f"{i}/{nses} {label}, registering {len(_files)} files")
            responses.append(self.register_dataset(_files, **kwargs))
        return responses

    def patch_dataset(self, file_list, dry=False, ftp=False, **kwargs):
        """
        Creates a new dataset on FlatIron and uploads it from arbitrary location.
        Rules for creation/patching are the same that apply for registration via Alyx
        as this uses the registration endpoint to get the dataset.
        An existing file (same session and path relative to session) will be patched.
        :param path: full file path. Must be whithin an ALF session folder (subject/date/number)
        can also be a list of full file pathes belonging to the same session.
        :param server_repository: Alyx server repository name
        :param created_by: alyx username for the dataset (optional, defaults to root)
        :param ftp: flag for case when using ftppatcher. Don't adjust windows path in
        _patch_dataset when ftp=True
        :return: the registrations response, a list of dataset records
        """
        # first register the file
        if not isinstance(file_list, list):
            file_list = [Path(file_list)]
        assert len(set([get_session_path(f) for f in file_list])) == 1
        assert all([Path(f).exists() for f in file_list])
        response = self.register_dataset(file_list, dry=dry, **kwargs)
        if dry:
            return
        # from the dataset info, set flatIron flag to exists=True
        for p, d in zip(file_list, response):
            self._patch_dataset(p, dset_id=d['id'], dry=dry, ftp=ftp)
        return response

    def patch_datasets(self, file_list, **kwargs):
        """
        Same as create_dataset method but works with several sessions
        """
        register_dict = {}
        # creates a dictionary of sessions with one file list per session
        for f in file_list:
            session_path = get_session_path(f)
            label = '_'.join(session_path.parts[-3:])
            if label in register_dict:
                register_dict[label]['files'].append(f)
            else:
                register_dict[label] = {'session_path': session_path, 'files': [f]}
        responses = []
        nses = len(register_dict)
        for i, label in enumerate(register_dict):
            _files = register_dict[label]['files']
            _logger.info(f"{i}/{nses} {label}, registering {len(_files)} files")
            responses.extend(self.patch_dataset(_files, **kwargs))
        return responses

    @abc.abstractmethod
    def _scp(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _rm(self, *args, **kwargs):
        pass


class GlobusPatcher(Patcher):
    """
    Requires GLOBUS keys access

    """

    def __init__(self, one=None, globus_client_id=None, local_endpoint=None, label='ibllib patch'):
        assert globus_client_id
        assert one
        self.local_endpoint = local_endpoint or globus.get_local_endpoint()
        self.label = label
        self.transfer_client = globus.login_auto(
            globus_client_id=globus_client_id, str_app='globus/admin')
        # transfers/delete from the current computer to the flatiron: mandatory and executed first
        self.globus_transfer = globus_sdk.TransferData(
            self.transfer_client, self.local_endpoint, FLAT_IRON_GLOBUS_ID, verify_checksum=True,
            sync_level='checksum', label=label)
        self.globus_delete = globus_sdk.DeleteData(
            self.transfer_client, FLAT_IRON_GLOBUS_ID, verify_checksum=True,
            sync_level='checksum', label=label)
        # get a dictionary of data repositories from Alyx (with globus ids)
        self.repos = {r['name']: r for r in one.alyx.rest('data-repository', 'list')}
        # transfers/delete from flatiron to optional third parties to synchronize / delete
        self.globus_transfers_locals = {}
        self.globus_deletes_locals = {}
        super().__init__(one=one)

    def _scp(self, local_path, remote_path, dry=True):
        remote_path = PurePosixPath('/').joinpath(
            remote_path.relative_to(PurePosixPath(FLATIRON_MOUNT))
        )
        _logger.info(f"Globus copy {local_path} to {remote_path}")
        if not dry:
            if isinstance(self.globus_transfer, globus_sdk.transfer.data.TransferData):
                self.globus_transfer.add_item(local_path, remote_path)
            else:
                self.globus_transfer.path_src.append(local_path)
                self.globus_transfer.path_dest.append(remote_path)
        return 0, ''

    def _rm(self, flatiron_path, dry=True):
        flatiron_path = Path('/').joinpath(flatiron_path.relative_to(Path(FLATIRON_MOUNT)))
        _logger.info(f"Globus del {flatiron_path}")
        if not dry:
            if isinstance(self.globus_delete, globus_sdk.transfer.data.DeleteData):
                self.globus_delete.add_item(flatiron_path)
            else:
                self.globus_delete.path.append(flatiron_path)
        return 0, ''

    def patch_datasets(self, file_list, **kwargs):
        """
        Calls the super method that registers and updates the current computer to Python transfer
        Then, creates individual transfer items for each local server so that after the
        update on Flatiron, local server files are also updated
        :param file_list:
        :param kwargs:
        :return:
        """
        responses = super().patch_datasets(file_list, **kwargs)
        for dset in responses:
            # get the flatiron path
            fr = next(fr for fr in dset['file_records'] if 'flatiron' in fr['data_repository'])
            flatiron_path = self.repos[fr['data_repository']]['globus_path']
            flatiron_path = Path(flatiron_path).joinpath(fr['relative_path'])
            flatiron_path = add_uuid_string(flatiron_path, dset['id']).as_posix()
            # loop over the remaining repositories (local servers) and create a transfer
            # from flatiron to the local server
            for fr in dset['file_records']:
                if fr['data_repository'] == DMZ_REPOSITORY:
                    continue
                repo_gid = self.repos[fr['data_repository']]['globus_endpoint_id']
                if repo_gid == FLAT_IRON_GLOBUS_ID:
                    continue
                # if there is no transfer already created, initialize it
                if repo_gid not in self.globus_transfers_locals:
                    self.globus_transfers_locals[repo_gid] = globus_sdk.TransferData(
                        self.transfer_client, FLAT_IRON_GLOBUS_ID, repo_gid, verify_checksum=True,
                        sync_level='checksum', label=f"{self.label} on {fr['data_repository']}")
                # get the local server path and create the transfer item
                local_server_path = self.repos[fr['data_repository']]['globus_path']
                local_server_path = Path(local_server_path).joinpath(fr['relative_path'])
                self.globus_transfers_locals[repo_gid].add_item(flatiron_path, local_server_path)
        return responses

    def launch_transfers(self, local_servers=False):
        """
        patcher.launch_transfers()
        Launches the globus transfer and delete from the local patch computer to the flat-rion
        :param: local_servers (False): if True, sync the local servers after the main transfer
        :return: None
        """
        gtc = self.transfer_client

        def _wait_for_task(resp):
            # patcher.transfer_client.get_task(task_id='364fbdd2-4deb-11eb-8ffb-0a34088e79f9')
            # on a good status:
            # Out[22]: TransferResponse({'bytes_checksummed': 377736912, 'bytes_transferred': 3011090432, 'canceled_by_admin': None, 'canceled_by_admin_message': None, 'command': 'API 0.10', 'completion_time': None, 'deadline': '2021-01-06T18:10:05+00:00', 'delete_destination_extra': False, 'destination_endpoint': 'simonsfoundation#ibl', 'destination_endpoint_display_name': 'IBL Flatiron SDSC Data', 'destination_endpoint_id': 'ab2d064c-413d-11eb-b188-0ee0d5d9299f', 'directories': 0, 'effective_bytes_per_second': 873268, 'encrypt_data': False, 'fatal_error': None, 'faults': 6, 'files': 186, 'files_skipped': 12, 'files_transferred': 76, 'history_deleted': False, 'is_ok': True, 'is_paused': False, 'key': 'active,2021-01-03T17:52:34.427087', 'label': '3B analog sync patch', 'nice_status': 'OK', 'nice_status_details': None, 'nice_status_expires_in': -1, 'nice_status_short_description': 'OK', 'owner_id': 'e633663a-8561-4a5d-ac92-f198d43b14dc', 'preserve_timestamp': False, 'recursive_symlinks': 'ignore', 'request_time': '2021-01-03T17:52:34+00:00', 'source_endpoint': 'internationalbrainlab#916c2766-bd2a-11ea-8f22-0a21f750d19b', 'source_endpoint_display_name': 'olivier_laptop', 'source_endpoint_id': '916c2766-bd2a-11ea-8f22-0a21f750d19b', 'status': 'ACTIVE', 'subtasks_canceled': 0, 'subtasks_expired': 0, 'subtasks_failed': 0, 'subtasks_pending': 98, 'subtasks_retrying': 0, 'subtasks_succeeded': 274, 'subtasks_total': 372, 'symlinks': 0, 'sync_level': 3, 'task_id': '364fbdd2-4deb-11eb-8ffb-0a34088e79f9', 'type': 'TRANSFER', 'username': 'internationalbrainlab', 'verify_checksum': True})  # noqa
            # on a checksum error
            # Out[26]: TransferResponse({'bytes_checksummed': 377736912, 'bytes_transferred': 3715901232, 'canceled_by_admin': None, 'canceled_by_admin_message': None, 'command': 'API 0.10', 'completion_time': None, 'deadline': '2021-01-06T18:10:05+00:00', 'delete_destination_extra': False, 'destination_endpoint': 'simonsfoundation#ibl', 'destination_endpoint_display_name': 'IBL Flatiron SDSC Data', 'destination_endpoint_id': 'ab2d064c-413d-11eb-b188-0ee0d5d9299f', 'directories': 0, 'effective_bytes_per_second': 912410, 'encrypt_data': False, 'fatal_error': None, 'faults': 7, 'files': 186, 'files_skipped': 12, 'files_transferred': 102, 'history_deleted': False, 'is_ok': False, 'is_paused': False, 'key': 'active,2021-01-03T17:52:34.427087', 'label': '3B analog sync patch', 'nice_status': 'VERIFY_CHECKSUM', 'nice_status_details': None, 'nice_status_expires_in': -1, 'nice_status_short_description': 'checksum verification failed', 'owner_id': 'e633663a-8561-4a5d-ac92-f198d43b14dc', 'preserve_timestamp': False, 'recursive_symlinks': 'ignore', 'request_time': '2021-01-03T17:52:34+00:00', 'source_endpoint': 'internationalbrainlab#916c2766-bd2a-11ea-8f22-0a21f750d19b', 'source_endpoint_display_name': 'olivier_laptop', 'source_endpoint_id': '916c2766-bd2a-11ea-8f22-0a21f750d19b', 'status': 'ACTIVE', 'subtasks_canceled': 0, 'subtasks_expired': 0, 'subtasks_failed': 0, 'subtasks_pending': 72, 'subtasks_retrying': 0, 'subtasks_succeeded': 300, 'subtasks_total': 372, 'symlinks': 0, 'sync_level': 3, 'task_id': '364fbdd2-4deb-11eb-8ffb-0a34088e79f9', 'type': 'TRANSFER', 'username': 'internationalbrainlab', 'verify_checksum': True})  # noqa
            # on a finished task
            # Out[4]: TransferResponse({'bytes_checksummed': 377736912, 'bytes_transferred': 4998806664, 'canceled_by_admin': None, 'canceled_by_admin_message': None, 'command': 'API 0.10', 'completion_time': '2021-01-03T20:04:50+00:00', 'deadline': '2021-01-06T19:11:00+00:00', 'delete_destination_extra': False, 'destination_endpoint': 'simonsfoundation#ibl', 'destination_endpoint_display_name': 'IBL Flatiron SDSC Data', 'destination_endpoint_id': 'ab2d064c-413d-11eb-b188-0ee0d5d9299f', 'directories': 0, 'effective_bytes_per_second': 629960, 'encrypt_data': False, 'fatal_error': None, 'faults': 15, 'files': 186, 'files_skipped': 12, 'files_transferred': 174, 'history_deleted': False, 'is_ok': None, 'is_paused': False, 'key': 'complete,2021-01-03T20:04:49.540956', 'label': '3B analog sync patch', 'nice_status': None, 'nice_status_details': None, 'nice_status_expires_in': None, 'nice_status_short_description': None, 'owner_id': 'e633663a-8561-4a5d-ac92-f198d43b14dc', 'preserve_timestamp': False, 'recursive_symlinks': 'ignore', 'request_time': '2021-01-03T17:52:34+00:00', 'source_endpoint': 'internationalbrainlab#916c2766-bd2a-11ea-8f22-0a21f750d19b', 'source_endpoint_display_name': 'olivier_laptop', 'source_endpoint_id': '916c2766-bd2a-11ea-8f22-0a21f750d19b', 'status': 'SUCCEEDED', 'subtasks_canceled': 0, 'subtasks_expired': 0, 'subtasks_failed': 0, 'subtasks_pending': 0, 'subtasks_retrying': 0, 'subtasks_succeeded': 372, 'subtasks_total': 372, 'symlinks': 0, 'sync_level': 3, 'task_id': '364fbdd2-4deb-11eb-8ffb-0a34088e79f9', 'type': 'TRANSFER', 'username': 'internationalbrainlab', 'verify_checksum': True})  # noqa
            # on an errored task
            # Out[10]: TransferResponse({'bytes_checksummed': 0, 'bytes_transferred': 0, 'canceled_by_admin': None, 'canceled_by_admin_message': None, 'command': 'API 0.10', 'completion_time': '2021-01-03T17:39:00+00:00', 'deadline': '2021-01-04T17:37:34+00:00', 'delete_destination_extra': False, 'destination_endpoint': 'simonsfoundation#ibl', 'destination_endpoint_display_name': 'IBL Flatiron SDSC Data', 'destination_endpoint_id': 'ab2d064c-413d-11eb-b188-0ee0d5d9299f', 'directories': 0, 'effective_bytes_per_second': 0, 'encrypt_data': False, 'fatal_error': {'code': 'CANCELED', 'description': 'canceled'}, 'faults': 2, 'files': 6, 'files_skipped': 0, 'files_transferred': 0, 'history_deleted': False, 'is_ok': None, 'is_paused': False, 'key': 'complete,2021-01-03T17:38:59.697413', 'label': 'test 3B analog sync patch', 'nice_status': None, 'nice_status_details': None, 'nice_status_expires_in': None, 'nice_status_short_description': None, 'owner_id': 'e633663a-8561-4a5d-ac92-f198d43b14dc', 'preserve_timestamp': False, 'recursive_symlinks': 'ignore', 'request_time': '2021-01-03T17:37:34+00:00', 'source_endpoint': 'internationalbrainlab#916c2766-bd2a-11ea-8f22-0a21f750d19b', 'source_endpoint_display_name': 'olivier_laptop', 'source_endpoint_id': '916c2766-bd2a-11ea-8f22-0a21f750d19b', 'status': 'FAILED', 'subtasks_canceled': 6, 'subtasks_expired': 0, 'subtasks_failed': 0, 'subtasks_pending': 0, 'subtasks_retrying': 0, 'subtasks_succeeded': 6, 'subtasks_total': 12, 'symlinks': 0, 'sync_level': 3, 'task_id': '5706dd2c-4dea-11eb-8ffb-0a34088e79f9', 'type': 'TRANSFER', 'username': 'internationalbrainlab', 'verify_checksum': True})  # noqa
            while True:
                tinfo = gtc.get_task(task_id=resp['task_id'])['completion_time']
                if tinfo['completion_time'] is not None:
                    break
                _ = gtc.task_wait(task_id=resp['task_id'], timeout=30)
            if tinfo['fatal_error'] is not None:
                raise ConnectionError(f"Globus transfer failed \n {tinfo}")

        # handles the transfers first
        if len(self.globus_transfer['DATA']) > 0:
            # launch the transfer
            _wait_for_task(gtc.submit_transfer(self.globus_transfer))
            # re-initialize the globus_transfer property
            self.globus_transfer = globus_sdk.TransferData(
                gtc,
                self.globus_transfer['source_endpoint'],
                self.globus_transfer['destination_endpoint'],
                label=self.globus_transfer['label'],
                verify_checksum=True, sync_level='checksum')

        # do the same for deletes
        if len(self.globus_delete['DATA']) > 0:
            _wait_for_task(gtc.submit_delete(self.globus_delete))
            self.globus_delete = globus_sdk.DeleteData(
                gtc,
                endpoint=self.globus_delete['endpoint'],
                label=self.globus_delete['label'],
                verify_checksum=True, sync_level='checksum')

        # launch the local transfers and local deletes
        if local_servers:
            self.launch_transfers_secondary()

    def launch_transfers_secondary(self):
        """
        patcher.launch_transfer_secondary()
        Launches the globus transfers from flatiron to third-party repositories (local servers)
        This should run after the the main transfer from patch computer to the flatiron
        :return: None
        """
        for lt in self.globus_transfers_locals:
            transfer = self.globus_transfers_locals[lt]
            if len(transfer['DATA']) > 0:
                self.transfer_client.submit_transfer(transfer)
        for ld in self.globus_deletes_locals:
            delete = self.globus_deletes_locals[ld]
            if len(transfer['DATA']) > 0:
                self.transfer_client.submit_delete(delete)


class SSHPatcher(Patcher):
    """
    Requires SSH keys access on the FlatIron
    """
    def __init__(self, one=None, globus_client=None):
        res = _run_command(f"ssh -p {FLATIRON_PORT} {FLATIRON_USER}@{FLATIRON_HOST} ls")
        if res[0] != 0:
            raise PermissionError("Could not connect to the Flatiron via SSH. Check your RSA keys")
        super().__init__(one=one)

    def _scp(self, local_path, remote_path, dry=True):
        cmd = f"ssh -p {FLATIRON_PORT} {FLATIRON_USER}@{FLATIRON_HOST}" \
              f" mkdir -p {remote_path.parent}; "
        cmd += f"scp -P {FLATIRON_PORT} {local_path} {FLATIRON_USER}@{FLATIRON_HOST}:{remote_path}"
        return _run_command(cmd, dry=dry)

    def _rm(self, flatiron_path, dry=True):
        cmd = f"ssh -p {FLATIRON_PORT} {FLATIRON_USER}@{FLATIRON_HOST} rm {flatiron_path}"
        return _run_command(cmd, dry=dry)


class FTPPatcher(Patcher):
    """
    This is used to register from anywhere without write access to FlatIron
    """
    def __init__(self, one=None):
        super().__init__(one=one)
        if not getattr(one.alyx._par, 'FTP_DATA_SERVER_LOGIN', False):
            self.one.alyx._par = self.setup(par=one.alyx._par)
        login, pwd = (one.alyx._par.FTP_DATA_SERVER_LOGIN, one.alyx._par.FTP_DATA_SERVER_PWD)
        self.ftp = ftplib.FTP_TLS(host=FTP_HOST, user=login, passwd=pwd)
        # self.ftp.ssl_version = ssl.PROTOCOL_TLSv1
        # self.ftp.auth()
        self.ftp.prot_p()
        self.ftp.login(login, pwd)
        # pre-fetch the repositories so as not to query them for every file registered
        self.repositories = self.one.alyx.rest("data-repository", "list")

    @staticmethod
    def setup(par=None, silent=False):
        """
        Set up (and save) FTP login parameters
        :param par: A parameters object to modify, if None the default Webclient parameters are
        loaded
        :param silent: If true, the defaults are used with no user input prompt
        :return: the modified parameters object
        """
        DEFAULTS = {
            "FTP_DATA_SERVER": "ftp://ibl.flatironinstitute.org",
            "FTP_DATA_SERVER_LOGIN": "iblftp",
            "FTP_DATA_SERVER_PWD": None
        }
        if par is None:
            par = params.get(silent=silent)
        par = iopar.as_dict(par)

        if silent:
            par = DEFAULTS.update(par)
        else:
            for k in DEFAULTS.keys():
                cpar = par.get(k, DEFAULTS[k])
                # Iterate through non-password pars; skip url if client url already provided
                if 'PWD' not in k:
                    par[k] = input(f'Param {k}, current value is ["{cpar}"]:') or cpar
                else:
                    prompt = f'Param {k} (leave empty to leave unchanged):'
                    par[k] = getpass(prompt) or cpar

        # Get the client key
        client = par.get('ALYX_URL', None)
        client_key = params._key_from_url(client) if client else params.get_default_client()
        # Save the parameters
        params.save(par, client_key)  # Client params
        return iopar.from_dict(par)

    def create_dataset(self, path, created_by='root', dry=False, repository=DMZ_REPOSITORY):
        # overrides the superclass just to remove the server repository argument
        response = super().patch_dataset(path, created_by=created_by, dry=dry,
                                         repository=repository, ftp=True)
        # need to patch the file records to be consistent
        for ds in response:
            frs = ds['file_records']
            fr_server = next(filter(lambda fr: 'flatiron' in fr['data_repository'], frs))
            fr_ftp = next(filter(lambda fr: fr['data_repository'] == DMZ_REPOSITORY and
                                 fr['relative_path'] == fr_server['relative_path'], frs))
            reposerver = next(filter(lambda rep: rep['name'] == fr_server['data_repository'],
                                     self.repositories))
            relative_path = str(PurePosixPath(reposerver['globus_path']).joinpath(
                PurePosixPath(fr_ftp['relative_path'])))[1:]
            # 1) if there was already a file, the registration created a duplicate
            fr_2del = list(filter(lambda fr: fr['data_repository'] == DMZ_REPOSITORY and
                                             fr['relative_path'] == relative_path, frs))  # NOQA
            if len(fr_2del) == 1:
                self.one.alyx.rest('files', 'delete', id=fr_2del[0]['id'])
            # 2) the patch ftp file needs to be prepended with the server repository path
            self.one.alyx.rest('files', 'partial_update', id=fr_ftp['id'],
                               data={'relative_path': relative_path, 'exists': True})
            # 3) the server file is labeled as not existing
            self.one.alyx.rest('files', 'partial_update', id=fr_server['id'],
                               data={'exists': False})
        return response

    def _scp(self, local_path, remote_path, dry=True):
        # remote_path = '/mnt/ibl/zadorlab/Subjects/flowers/2018-07-13/001
        remote_path = PurePosixPath('/').joinpath(
            remote_path.relative_to(PurePosixPath(FLATIRON_MOUNT))
        )
        # local_path
        self.mktree(remote_path.parent)
        self.ftp.pwd()
        _logger.info(f"FTP upload {local_path}")
        with open(local_path, 'rb') as fid:
            self.ftp.storbinary(f'STOR {local_path.name}', fid)
        return 0, ''

    def mktree(self, remote_path):
        """ Browse to the tree on the ftp server, making directories on the way"""
        if str(remote_path) != '.':
            try:
                self.ftp.cwd(str(remote_path))
            except ftplib.error_perm:
                self.mktree(PurePosixPath(remote_path.parent))
                self.ftp.mkd(str(remote_path))
                self.ftp.cwd(str(remote_path))

    def _rm(self, flatiron_path, dry=True):
        raise PermissionError("This Patcher does not have admin permissions to remove data "
                              "from the FlatIron server. ")

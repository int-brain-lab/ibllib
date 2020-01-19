import os.path as op
from pathlib import Path
import subprocess
import logging

from ibllib.io.hashfile import md5
from alf.io import is_uuid_string, get_session_path, add_uuid_string, is_session_path
from oneibl.one import ONE

_logger = logging.getLogger('ibllib')
FLATIRON_HOST = 'ibl.flatironinstitute.org'
FLATIRON_PORT = 61022
FLATIRON_USER = 'datauser'
FLATIRON_MOUNT = '/mnt/ibl'


def _add_uuid_to_filename(fn, uuid):
    if uuid in fn:
        return fn
    dpath, ext = op.splitext(fn)
    return dpath + '.' + str(uuid) + ext


def _globus_scp(local_path, remote_path, dry=True):
    pass


def _globus_rm(flatiron_path, dry=True):
    pass


def _ssh_scp(local_path, remote_path, dry=True):
    cmd = f"ssh -p {FLATIRON_PORT} {FLATIRON_USER}@{FLATIRON_HOST} mkdir -p {remote_path.parent}"
    cmd += f"; scp -P {FLATIRON_PORT} {local_path} {FLATIRON_USER}@{FLATIRON_HOST}:{remote_path}"
    return _run_command(cmd, dry=dry)


def _ssh_rm(flatiron_path, dry=True):
    cmd = f"ssh -p {FLATIRON_PORT} {FLATIRON_USER}@{FLATIRON_HOST} rm {flatiron_path}"
    return _run_command(cmd, dry=dry)


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


class Patcher:
    """
    Requires SSH keys access to the target server
    """
    def __init__(self, one=None, globus_client=None):
        # one object
        if one is None:
            self.one = ONE()
        else:
            self.one = one
        # handles the globus objects
        self.globus = globus_client
        if globus_client is None:
            self.globus_transfer = None
            self.globus_delete = None
        else:
            pass

    def patch_dataset(self, path, dset_id=None, dry=False):
        """
        Uploads a dataset from an arbitrary location to FlatIron.
        :param path:
        :param dset_id:
        :param dry:
        :return:
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
        remote_path = op.join(fr['data_repository_path'], fr['relative_path'])
        remote_path = _add_uuid_to_filename(remote_path, dset_id)
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        self._scp(path, Path(FLATIRON_MOUNT) / remote_path, dry=dry)
        if not dry:
            self.one.alyx.rest('datasets', 'partial_update', id=dset_id,
                               data={'md5': md5(path), 'file_size': path.stat().st_size})

    def create_dataset(self, path, server_repository=None, created_by='root', dry=False):
        """
        Creates a new dataset on FlatIron and uploads it from arbitrary location.
        Rules for creation/patching are the same that apply for registration via Alyx
        as this uses the registration endpoint to get the dataset.
        An existing file (same session and path relative to session) will be patched.
        :param path: full file path. Must be whithin an ALF session folder (subject/date/number)
        :param server_repository: Alyx server repository name
        :param created_by: alyx username for the dataset (optional, defaults to root)
        :return:
        """
        path = Path(path)
        assert path.exists()
        session_path = get_session_path(path)
        assert is_session_path(session_path)
        # first register the file
        r = {'created_by': created_by,
             'path': str(session_path.relative_to((session_path.parents[2]))),
             'filenames': [str(path.relative_to(session_path))],
             'name': server_repository,
             'server_only': True,
             'hash': md5(path),
             'filesizes': path.stat().st_size}
        if not dry:
            dataset = self.one.alyx.rest('register-file', 'create', data=r)[0]
        else:
            print(r)
            return
        # from the dataset info, set flatIron flag to exists=True
        self.patch_dataset(path, dset_id=dataset['id'], dry=dry)

    def delete_dataset(self, dset_id, dry=False):
        dset = self.one.alyx.rest('datasets', "read", id=dset_id)
        assert dset
        for fr in dset['file_records']:
            if 'flatiron' in fr['data_repository']:
                flatiron_path = Path(FLATIRON_MOUNT).joinpath(fr['data_repository_path'][1:],
                                                              fr['relative_path'])
                flatiron_path = add_uuid_string(flatiron_path, dset_id)
                self._rm(flatiron_path, dry=dry)
                self.one.alyx.rest('datasets', 'delete', id=dset_id)

    def delete_session_datasets(self, eid, dry=True):
        """
            Deletes all datasets attached to the session from database and flatiron but leaves
            the session on the database.
            Useful for a full re-extraction
        """
        ses_details = self.one.alyx.rest('sessions', 'read', id=eid)
        raise NotImplementedError("Code below only removes existing files. Need to search"
                                  "for datasets in a better way")
        # first delete attached datasets from the database
        dataset_details = self.one.list(eid, details=True)
        for n in range(len(dataset_details.dataset_id)):
            print(dataset_details.dataset_id[n])
            if dry:
                continue
            try:
                self.one.alyx.rest('datasets', 'delete', id=dataset_details.dataset_id[n])
            except Exception as e:
                print(e)

        # then delete the session folder from flatiron
        flatiron_path = Path(FLATIRON_MOUNT).joinpath(ses_details['lab'],
                                                      'Subjects',
                                                      ses_details['subject'],
                                                      ses_details['start_time'][:10],
                                                      str(ses_details['number']).zfill(3))

        cmd = f"ssh -p {FLATIRON_PORT} {FLATIRON_USER}@{FLATIRON_HOST} rm -fR {flatiron_path}"
        print(cmd)

    def launch_globus_jobs(self):
        pass

    def _scp(self, *args, **kwargs):
        if self.globus is None:
            return _ssh_scp(*args, **kwargs)
        else:
            return _globus_scp(*args, **kwargs)

    def _rm(self, *args, **kwargs):
        if self.globus is None:
            return _ssh_rm(*args, **kwargs)
        else:
            return _globus_rm(*args, **kwargs)

import os.path as op
from pathlib import Path
import subprocess

from alf.io import is_uuid_string, get_session_path
from oneibl.one import ONE

FLATIRON_HOST = 'ibl.flatironinstitute.org'
FLATIRON_PORT = 61022
FLATIRON_USER = 'datauser'
FLATIRON_MOUNT = '/mnt/ibl'


def _add_uuid_to_filename(fn, uuid):
    if uuid in fn:
        return fn
    dpath, ext = op.splitext(fn)
    return dpath + '.' + str(uuid) + ext


def scp(local_path, remote_path, dry=False):
    cmd = f"scp -P {FLATIRON_PORT} {local_path} {FLATIRON_USER}@{FLATIRON_HOST}:{remote_path}"
    if dry:
        print(cmd)
        return 0,
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info, error = p.communicate()
    return p.returncode, info, error


class Patcher:
    def __init__(self, one=None):
        if one:
            self.one = one
        else:
            self.one = ONE()

    def patch_dataset(self, path, dset_id=None, dry=False):
        """
        Uploads a dataset from an arbitrary location to FlatIron. Needs admin permissions
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
        al = self.one.alyx
        dset = al.rest('datasets', "read", id=dset_id)
        fr = next(fr for fr in dset['file_records'] if 'flatiron' in fr['data_repository'])
        remote_path = op.join(fr['data_repository_path'], fr['relative_path'])
        remote_path = _add_uuid_to_filename(remote_path, dset_id)
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        returncode, info, error = scp(path, Path(FLATIRON_MOUNT) / remote_path, dry=dry)
        if returncode != 0:
            raise RuntimeError(error)

    def create_dataset(self, path, labname=None):
        """
        Creates a new dataset on FlatIron and uploads it from arbitrary location. Needs admin.
        :param path:
        :param labname:
        :return:
        """
        path = Path(path)
        assert path.exists()
        assert labname
        session_path = get_session_path(path)
        ac = self.one.alyx
        # first register the file
        r = {'created_by': 'olivier',
             'path': str(session_path.relative_to((session_path.parents[2]))),
             'filenames': [str(path.relative_to(session_path))],
             'labs': labname}
        dataset = ac.rest('register-file', 'create', data=r)
        # from the dataset info, set flatIron flag to exists=True
        for dset in dataset:
            for fr in dset['file_records']:
                if 'flatiron' in fr['data_repository']:
                    ac.rest('files', 'partial_update', id=fr['id'], data={'exists': True})
            # and scp the dataset to FlatIron
            self.patch_dataset(path, dset_id=dset['id'])

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

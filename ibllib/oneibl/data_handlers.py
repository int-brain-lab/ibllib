import logging
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os
import abc
from time import time

from one.util import filter_datasets
from one.alf.files import add_uuid_string
from iblutil.io.parquet import np2str
from ibllib.oneibl.registration import register_dataset
from ibllib.oneibl.patcher import FTPPatcher, SDSCPatcher, SDSC_ROOT_PATH, SDSC_PATCH_PATH
from ibllib.oneibl.aws import AWS

_logger = logging.getLogger('ibllib')


class DataHandler(abc.ABC):
    def __init__(self, session_path, signature, one=None):
        """
        Base data handler class
        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        self.session_path = session_path
        self.signature = signature
        self.one = one

    def setUp(self):
        """
        Function to optionally overload to download required data to run task
        :return:
        """
        pass

    def getData(self):
        """
        Finds the datasets required for task based on input signatures
        :return:
        """
        if self.one is None:
            return
        session_datasets = self.one.list_datasets(self.one.path2eid(self.session_path), details=True)
        df = pd.DataFrame(columns=self.one._cache.datasets.columns)
        for file in self.signature['input_files']:
            df = df.append(filter_datasets(session_datasets, filename=file[0], collection=file[1],
                           wildcards=True, assert_unique=False))
        return df

    def uploadData(self, outputs, version):
        """
        Function to optionally overload to upload and register data
        :param outputs: output files from task to register
        :param version: ibllib version
        :return:
        """
        if isinstance(outputs, list):
            versions = [version for _ in outputs]
        else:
            versions = [version]

        return versions

    def cleanUp(self):
        """
        Function to optionally overload to cleanup files after running task
        :return:
        """
        pass


class LocalDataHandler(DataHandler):
    def __init__(self, session_path, signatures, one=None):
        """
        Data handler for running tasks locally, with no architecture or db connection
        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signatures, one=one)


class ServerDataHandler(DataHandler):
    def __init__(self, session_path, signatures, one=None):
        """
        Data handler for running tasks on lab local servers when all data is available locally

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signatures, one=one)

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)

        return register_dataset(outputs, one=self.one, versions=versions, **kwargs)


class ServerGlobusDataHandler(DataHandler):
    def __init__(self, session_path, signatures, one=None):
        """
        Data handler for running tasks on lab local servers. Will download missing data from SDSC using Globus

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        from one.globus import Globus, get_lab_from_endpoint_id  # noqa
        super().__init__(session_path, signatures, one=one)
        self.globus = Globus(client_name='server')

        # on local servers set up the local root path manually as some have different globus config paths
        self.globus.endpoints['local']['root_path'] = '/mnt/s0/Data/Subjects'

        # Find the lab
        labs = get_lab_from_endpoint_id(one=self.one)
        if len(labs) == 2:
            # for flofer lab
            subject = self.one.path2ref(self.session_path)['subject']
            self.lab = self.one.alyx.rest('subjects', 'list', nickname=subject)[0]['lab']
        else:
            self.lab = labs[0]

        self.globus.add_endpoint(f'flatiron_{self.lab}')

    def setUp(self):
        """
        Function to download necessary data to run tasks using globus-sdk
        :return:
        """
        df = super().getData()

        if len(df) == 0:
            # If no datasets found in the cache only work off local file system do not attempt to download any missing data
            # using globus
            return

        # Check for space on local server. If less that 500 GB don't download new data
        space_free = shutil.disk_usage(self.globus.endpoints['local']['root_path'])[2]
        if space_free < 500e9:
            _logger.warning('Space left on server is < 500GB, wont redownload new data')
            return

        rel_sess_path = '/'.join(df.iloc[0]['session_path'].split('/')[-3:])
        assert (rel_sess_path.split('/')[0] == self.one.path2ref(self.session_path)['subject'])

        target_paths = []
        source_paths = []
        self.local_paths = []
        for _, d in df.iterrows():
            sess_path = Path(rel_sess_path).joinpath(d['rel_path'])
            full_local_path = Path(self.globus.endpoints['local']['root_path']).joinpath(sess_path)
            if not full_local_path.exists():
                self.local_paths.append(full_local_path)
                target_paths.append(sess_path)
                source_paths.append(add_uuid_string(sess_path, np2str(np.r_[d.name[0], d.name[1]])))

        if len(target_paths) != 0:
            ts = time()
            for sp, tp in zip(source_paths, target_paths):
                _logger.info(f'Downloading {sp} to {tp}')
            self.globus.mv(f'flatiron_{self.lab}', 'local', source_paths, target_paths)
            _logger.debug(f'Complete. Time elapsed {time() - ts}')

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)

        return register_dataset(outputs, one=self.one, versions=versions, **kwargs)

    def cleanUp(self):
        """
        Clean up, remove the files that were downloaded from globus once task has completed
        :return:
        """
        for file in self.local_paths:
            os.unlink(file)


class RemoteHttpDataHandler(DataHandler):
    def __init__(self, session_path, signature, one=None):
        """
        Data handler for running tasks on remote compute node. Will download missing data via http using ONE

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signature, one=one)

    def setUp(self):
        """
        Function to download necessary data to run tasks using ONE
        :return:
        """
        df = super().getData()
        self.one._download_datasets(df)

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via FTP patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        ftp_patcher = FTPPatcher(one=self.one)
        return ftp_patcher.create_dataset(path=outputs, created_by=self.one.alyx.user,
                                          versions=versions, **kwargs)


class RemoteAwsDataHandler(DataHandler):
    def __init__(self, session_path, signature, one=None):
        """
        Data handler for running tasks on remote compute node. Will download missing data from private ibl s3 AWS data bucket

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signature, one=one)
        self.aws = AWS(one=self.one)

    def setUp(self):
        """
        Function to download necessary data to run tasks using AWS boto3
        :return:
        """
        df = super().getData()
        self.aws._download_datasets(df)

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via FTP patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        ftp_patcher = FTPPatcher(one=self.one)
        return ftp_patcher.create_dataset(path=outputs, created_by=self.one.alyx.user,
                                          versions=versions, **kwargs)


class RemoteGlobusDataHandler(DataHandler):
    """
    Data handler for running tasks on remote compute node. Will download missing data using globus

    :param session_path: path to session
    :param signature: input and output file signatures
    :param one: ONE instance
    """
    def __init__(self, session_path, signature, one=None):
        super().__init__(session_path, signature, one=one)

    def setUp(self):
        """
        Function to download necessary data to run tasks using globus
        :return:
        """
        # TODO
        pass

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via FTP patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        ftp_patcher = FTPPatcher(one=self.one)
        return ftp_patcher.create_dataset(path=outputs, created_by=self.one.alyx.user,
                                          versions=versions, **kwargs)


class SDSCDataHandler(DataHandler):
    """
    Data handler for running tasks on SDSC compute node

    :param session_path: path to session
    :param signature: input and output file signatures
    :param one: ONE instance
    """
    def __init__(self, task, session_path, signatures, one=None):
        super().__init__(session_path, signatures, one=one)
        self.task = task

    def setUp(self):
        """
        Function to create symlinks to necessary data to run tasks
        :return:
        """
        df = super().getData()

        SDSC_TMP = Path(SDSC_PATCH_PATH.joinpath(self.task.__class__.__name__))
        for _, d in df.iterrows():
            file_path = Path(d['session_path']).joinpath(d['rel_path'])
            file_uuid = add_uuid_string(file_path, np2str(np.r_[d.name[0], d.name[1]]))
            file_link = SDSC_TMP.joinpath(file_path)
            file_link.parent.mkdir(exist_ok=True, parents=True)
            file_link.symlink_to(
                Path(SDSC_ROOT_PATH.joinpath(file_uuid)))

        self.task.session_path = SDSC_TMP.joinpath(d['session_path'])

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via SDSC patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        sdsc_patcher = SDSCPatcher(one=self.one)
        return sdsc_patcher.patch_datasets(outputs, dry=False, versions=versions, **kwargs)

    def cleanUp(self):
        """
        Function to clean up symlinks created to run task
        :return:
        """
        assert SDSC_PATCH_PATH.parts[0:4] == self.task.session_path.parts[0:4]
        shutil.rmtree(self.task.session_path)

"""
Patcher rewrite:
scope: remove or overwrite, NOT create new datasets (that would be registration)

logic:
patcher class that is agnostic to infrastructure
seperate patcher object encapsulating the infrastructures (s3 and globus)

IBLPatcher instantiates patchers and delegates the calls

general TODO
figure out local file paths / globus file paths / SDSC file paths

TODO
what are the special parts of the SDSCPatcher (UUID in filename, flatiron path?)

"""

import abc
import time
from enum import Enum
from pathlib import Path
from typing import Literal
import logging


# from one.converters import path_from_filerecord
import boto3
import globus_sdk

from one.alf.path import get_session_path, add_uuid_string, full_path_parts
from one.alf.spec import is_uuid_string, is_uuid

from one.api import ONE
from one.remote import globus
from one.remote.aws import url2uri

_logger = logging.getLogger(__name__)

# we will need those probably

# FLATIRON_HOST = 'ibl.flatironinstitute.org'
# FLATIRON_PORT = 61022
# FLATIRON_USER = 'datauser'
# FLATIRON_MOUNT = '/mnt/ibl'
# DMZ_REPOSITORY = 'ibl_patcher'  # in alyx, the repository name containing the patched filerecords
# SDSC_ROOT_PATH = PurePosixPath(FLATIRON_MOUNT)
# SDSC_PATCH_PATH = PurePosixPath('/home/datauser/temp')


class Status(Enum):
    SUCCESS = 0
    FAIL = -1
    # to be extendend as necessary (PENDING?)


def get_repository_type_by_name(name: str) -> str:
    if name.startwith('aws_'):
        return 'aws'
    elif name.startswith('flatiron_'):
        return 'flatiron'
    elif '_lab_' in name:
        return 'local_server'
    else:
        return 'other'
    # TODO and, which other?


def check_files_and_file_records(
    files: list[str] | list[Path],
    file_records: list[dict],
):
    # various checks if the files of the local file paths match those in the file records
    assert len(files) == len(file_records), 'number of files and file records must match'
    # if files is a list of str, convert to Path
    files = [Path(f) if isinstance(f, str) else f for f in files]
    # check if file name is the same name as in the file record
    for file, file_record in zip(files, file_records):
        if file.name != Path(file_record['relative_path']).name:
            raise ValueError('file name does not match file record')
    return True


"""
   ###    ########   ######
  ## ##   ##     ## ##    ##
 ##   ##  ##     ## ##
##     ## ########  ##
######### ##     ## ##
##     ## ##     ## ##    ##
##     ## ########   ######
"""


class Patcher(abc.ABC):
    # abstract Patcher class that defines the expected functionality
    # - single and multi file patching and deletion of files associated to
    # file records
    #  - optional removal of file records from alyx

    # dry mode: just log what would be done
    # on_error: controls if exceptions are raised of just logged

    # in general - I think it would be great if the multi file operations
    # just call the single file operations (which take care of all the
    # checks and logging etc)
    # - but for the globus patcher the logic is inverted - the base operation is
    # multi file and the single file operation is a special 1 element only case
    # see notes there

    def __init__(
        self,
        one: ONE,
        dry: bool = False,
        on_error: Literal['raise', 'log'] = 'raise',
    ):
        self.one = one
        self.dry = dry
        self.on_error = on_error

    @abc.abstractmethod
    def check_file(
        file_record: dict,
    ) -> bool:
        # checks if file is present on the infrastucture
        ...

    @abc.abstractmethod
    def delete_file(
        self,
        file_record: dict,
        check_exists: bool = True,
    ) -> Status:
        # deletes a single file, optionally checking if it exists beforehand
        # returns a single Status

        # GR: maybe we want to always check for existence. It might be less
        # performant but that might not be an issue
        ...

    @abc.abstractmethod
    def delete_files(
        self,
        file_records: list[dict],
        check_exists: bool = True,
        ignore_missing: bool = False,
    ) -> dict[str, Status]:
        # multi file operation
        # ignore_missing: if file is not found but a delete is attempted, just log and continue
        # returns a dict of {file_record['id']:Status}
        ...

    @abc.abstractmethod
    def patch_file(
        self,
        file_record: dict,
        file: str | Path,
        check_exists: bool = True,
    ) -> Status:
        # single file patch operation: copies the local file to the location specified
        # in the file record
        # optionally checks for file existence beforehand (again ... maybe this shouldn't be optional)
        ...

    @abc.abstractmethod
    def patch_files(
        self,
        file_records: list[dict],
        files: list[str] | list[Path],
        check_exists: bool = True,
    ) -> dict[str, Status]:
        # multi file operation of the single file
        # returns a dict of {file_record['id']:Status}
        ...


"""
   ###    ##      ##  ######
  ## ##   ##  ##  ## ##    ##
 ##   ##  ##  ##  ## ##
##     ## ##  ##  ##  ######
######### ##  ##  ##       ##
##     ## ##  ##  ## ##    ##
##     ##  ###  ###   ######
"""


class AWSPatcher(Patcher):
    def __init__(
        self,
        one: ONE,
        aws_profile: str = 'default',
        aws_region: str = 'us-east-1',
        **kwargs,
    ):
        super().__init__(one=one, **kwargs)
        self.aws_profile = aws_profile
        self.aws_region = aws_region

        # init boto3 stub
        # TODO add here: exception handling of authentification errors
        self.boto3_session = boto3.Session(profile_name=aws_profile)

    def check_file(self, file_record: dict) -> bool:
        assert self._file_record_is_aws(file_record)
        _, bucket, key = self.get_aws_info_for_file_record(file_record)

        # use the boto client to check for the file
        s3_client = self.boto3_session.client('s3', region_name=self.aws_region)
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except s3_client.exceptions.NoSuchKey:
            return False

    def delete_file(
        self,
        file_record: dict,
        check_exists: bool = True,
        ignore_missing: bool = False,
    ) -> Status:
        assert self._file_record_is_aws(file_record)

        # if a file is not found, skip it
        if check_exists:
            file_exists = self.check_file(file_record)
            if not file_exists and ignore_missing:
                _logger.warning(f'file does not exist, skipping delete: {file_record["data_url"]}')
                return Status.SUCCESS
            else:
                _logger.error(f'failed to delete non-existent file: {file_record["data_url"]}')
                return Status.FAIL

        _, bucket, key = self._get_aws_info_for_file_record(file_record)

        return self._aws_delete(bucket, key)

    def delete_files(
        self,
        file_records: list[dict],
        check_exists: bool = True,
        ignore_missing: bool = False,
    ) -> dict[str, Status]:
        # for the aws patcher simply execute the call to delete for all file records
        statuses = {}
        for file_record in file_records:
            status = self.delete_file(file_record, check_exists=check_exists, ignore_missing=ignore_missing)
            statuses[file_record['id']] = status

        return statuses

    def patch_file(
        self,
        file_record: dict,
        file: str | Path,
        check_exists: bool = True,
    ) -> Status:
        assert self._file_record_is_aws(file_record)
        # determine the bucket / key
        s3_file, bucket, key = self._get_aws_info_for_file_record(file_record)

        if check_exists:
            if not self.check_file(file_record):
                _logger.error(f'failed to patch non-existent file: {s3_file}')
                return Status.FAIL

        # launch the transfer
        return self._aws_transfer(file, bucket, key)

    def patch_files(
        self,
        file_records: list[dict],
        files: list[str] | list[Path],
        check_exists: bool = True,
    ) -> dict[str, Status]:
        assert check_files_and_file_records(files, file_records)

        # for the aws patcher simply execute the call to delete for all file records
        statuses = {}
        for file_record, local_file in zip(file_records, files):
            status = self.patch_file(file_record, local_file, check_exists=check_exists)
            statuses[file_record['id']] = status

        return statuses

    def _aws_delete(
        self,
        bucket: str,
        key: str,
    ) -> Status:
        # remove s3 file defined by bucket/key pair
        # use the boto client
        s3_file = f's3://{bucket}/{key}'
        s3_client = self.boto3_session.client('s3', region_name=self.aws_region)
        if self.dry:
            _logger.debug(f'would delete: {s3_file}')
            return Status.SUCCESS
        else:
            try:
                response = s3_client.delete_object(Bucket=bucket, Key=key)
                status_code = response['ResponseMetadata']['HTTPStatusCode']
                assert status_code == 200  # TODO verify the the codes
                _logger.info(f'deleted: {s3_file}')
                return Status.SUCCESS
            except Exception as e:
                _logger.error(f'failed to delete: {s3_file}')
                if self.on_error == 'raise':
                    raise e
                elif self.on_error == 'log':
                    return Status.FAIL

    def _aws_transfer(
        self,
        local_file: str | Path,
        bucket: str,
        key: str,
    ) -> Status:
        # just the aws transfer with everything else known
        # logging etc
        s3_file = f's3://{bucket}/{key}'
        assert self.boto3_session is not None, (
            'aws boto3 session not initialized'
        )  # this should be properly handles during init ...
        s3_client = self.boto3_session.client('s3', region_name=self.aws_region)

        if self.dry:
            _logger.debug(f'would patch: {s3_file} with {local_file}')
            return Status.SUCCESS
        else:
            try:
                s3_client.upload_file(local_file, bucket, key)  # TODO does this automatically overwrite?
                _logger.info(f'patched: {s3_file} with {local_file}')
                return Status.SUCCESS
            except Exception as e:
                _logger.error(f'Failed to patch: {s3_file} with {local_file}')
                if self.on_error == 'raise':
                    raise e
                elif self.on_error == 'log':
                    return Status.FAIL

    def _file_record_is_aws(self, file_record: dict):
        # helper to check if a file record is aws based
        # not clear if necessary, potentially remove
        return get_repository_type_by_name(file_record) == 'aws'

    def _get_aws_info_for_file_record(self, file_record: dict):
        # helper to get s3_file, bucket and key from a file record

        assert self._file_record_is_aws(file_record)  # otherwise something is really wrong

        # get s3_file uri
        s3_file = url2uri(file_record['data_url'])
        bucket = s3_file[5:].split('/')[0]
        key = s3_file.split(bucket)[1][1:]
        return s3_file, bucket, key


"""
 ######   ##        #######  ########  ##     ##  ######
##    ##  ##       ##     ## ##     ## ##     ## ##    ##
##        ##       ##     ## ##     ## ##     ## ##
##   #### ##       ##     ## ########  ##     ##  ######
##    ##  ##       ##     ## ##     ## ##     ##       ##
##    ##  ##       ##     ## ##     ## ##     ## ##    ##
 ######   ########  #######  ########   #######   ######
"""


class GlobusPatcher(Patcher):
    def __init__(
        self,
        one: ONE,
        globus_client_name: str = 'default',
        patch_label: str = 'generic patch label',
        **kwargs,
    ):
        super().__init__(one=one, **kwargs)
        self.globus = globus.Globus(client_name=globus_client_name)
        _ = globus.fetch_endpoints_from_alyx(alyx=self.one.alyx)
        self.patch_label = patch_label

    def check_file(self, file_record: dict):
        # TODO implement this function
        ...

    def delete_file(
        self,
        file_record: dict,
        ignore_missing: bool = False,
    ) -> Status:
        path_at_endpoint = self._get_path_at_endpoint(file_record)
        endpoint = self.globus.endpoints[file_record['data_repository']]['id']

        if self.dry:
            _logger.info(f'would globus delete: {path_at_endpoint} on {endpoint}')
            return Status.SUCCESS
        try:
            task_id = self.globus.delete_data(
                path_at_endpoint,
                endpoint,
                ignore_missing=ignore_missing,
            )
            self._wait_for_task(task_id)
            # TODO verify if this works, FIXME it's also pretty ugly
            globus_events = self.globus_client.task_event_list(task_id, num_results=None)
            status = self._globus_events_to_status_dict(globus_events, [file_record])[file_record['id']]
            _logger.info(f'completed globus delete for {path_at_endpoint} at {endpoint}')
        except Exception as e:
            _logger.info(f'failed globus delete for {path_at_endpoint} at {endpoint}')
            if self.on_error == 'raise':
                raise e
            # elif self.on_error == 'log':
            #     return Status.FAIL
        return status

    def delete_files(
        self,
        file_records: list[dict],
        ignore_missing: bool = False,
        per_file: bool = False,
    ) -> dict[str, Status]:
        statuses = {}
        # simple implementation: iterate over file records and process them one by one
        if per_file:
            for file_record in file_records:
                statuses[file_record['id']] = self.delete_file(file_record, ignore_missing=ignore_missing)
            return statuses

        # otherwise: launch one globus task per repository
        repositories = set(file_record['data_repository'] for file_record in file_records)
        for repository in repositories:
            _file_records = list(filter(lambda file_record: file_record[repository] == repository, file_records))
            paths_at_endpoint = [self._get_path_at_endpoint(file_record) for file_record in _file_records]
            endpoint = self.globus.endpoints[repository]['id']
            if self.dry:
                for path in paths_at_endpoint:  # FIXME explicit, but not great (log pollution)
                    _logger.info(f'would globus delete: {path} on {repository}')
                statuses[_file_records['id']] = Status.SUCCESS
            else:
                try:
                    task_id = self.globus.delete_data(paths_at_endpoint, endpoint)
                    self._wait_for_task(task_id)
                    globus_events = self.globus_client.task_event_list(task_id, num_results=None)
                    statuses.update(self._globus_events_to_status_dict(globus_events, _file_records))
                except Exception as e:
                    # how to handle the statuses if globus.delete_data raises and exception?
                    # can't get the statuses from globus as there is no task_id returend
                    # set all to fail
                    _logger.error(f'globus delete failure  on {repository}')
                    if self.on_error == 'raise':
                        raise e
                    elif self.on_error == 'log':
                        statuses.update({_file_record['id']: Status.FAIL for _file_record in _file_records})

        return statuses

    def patch_file(
        self,
        file_record: dict,
        file: str | Path,
    ) -> Status:
        source_endpoint = self.globus.endpoints['local']['id']
        destination_endpoint = self.globus.endpoints[file_record['repository']]['id']

        # the globus infrastructure was devised to move data between endpoints (not upload from disk)
        # make sure that the file path matches that what would be expected from globus
        assert file == self.globus.to_address(file, source_endpoint)
        _, root_path = self.globus._endpoint_id_root(source_endpoint)
        file_ = file.relative_to(root_path)

        if self.dry:
            _logger.info(f'would globus transfer: {file} to {file_record["repository"]}')
            return Status.SUCCESS
        try:
            task_id = self.globus.transfer_data(file_, source_endpoint, destination_endpoint)
            self._wait_for_task(task_id)  # TODO how to properly get the status here
            _logger.info(f'completed globus transfer: {file} to {file_record["repository"]}')
            return Status.SUCCESS
        except Exception as e:
            _logger.info(f'failed globus transfer: {file} to {file_record["repository"]}')
            if self.on_error == 'raise':
                raise e
            elif self.on_error == 'log':
                return Status.FAIL

    def patch_files(
        self,
        file_records: list[dict],
        files: list[str] | list[Path],
        per_file: bool = False,
    ) -> dict[str, Status]:
        # checks
        assert check_files_and_file_records(files, file_records)
        statuses = {}
        if per_file:
            for file_record, file in zip(file_records, files):
                statuses[file_record['id']] = self.patch_file(file_record, file)
            return statuses

        repositories = set(file_record['data_repository'] for file_record in file_records)
        # TODO maybe check here that all repositories are globus compatible repositories?

        source_endpoint = self.globus.endpoints['local']['id']
        _, root_path = self.globus._endpoint_id_root(source_endpoint)

        for repository in repositories:
            _file_records = []
            _files = []
            for file_record, file in zip(file_records, files):
                if file_record['data_repository'] == repository:
                    _file_records.append(file_record)
                    _files.append(file)

            # check if all files are valid
            for _file in _files:
                assert _file == self.globus.to_address(_file, source_endpoint)

            _files_rel = [_file.relative_to(root_path) for _file in _files]
            destination_endpoint = self.globus.endpoints[file_record['repository']]['id']
            try:
                task_id = self.globus.transfer_data(_files_rel, source_endpoint, destination_endpoint)
                self._wait_for_task(task_id)
                globus_events = self.globus_client.task_event_list(task_id, num_results=None)
                statuses.update(self._globus_events_to_status_dict(globus_events, _file_records))
            except Exception as e:
                _logger.error(f'failed globus transfer for {_files} from {source_endpoint} to {destination_endpoint}')
                if self.on_error == 'raise':
                    raise e
                elif self.on_error == 'log':
                    return

    def _wait_for_task(
        self,
        globus_task_id: str,
        max_wait: int = 3600,
        backoff_factor: float = 1.5,
    ):
        # blocking wait for a globus task with exponential backoff
        sleep_interval = 1.0
        t_start = time.time()
        while self.globus_client.get_task(globus_task_id)['status'] not in ['SUCCEEDED', 'FAILED']:
            time.sleep(sleep_interval)
            sleep_interval += sleep_interval * backoff_factor
            if (time.time() - t_start) > max_wait:
                msg = f'Globus task {globus_task_id} did not complete in time'
                _logger.error(msg)
                if self.on_error == 'raise':
                    raise TimeoutError(msg)
                elif self.on_error == 'log':
                    return None
        return None

    def _globus_events_to_status_dict(
        globus_events: list[str, dict],
        file_records: list[dict],
    ) -> list[str, Status]:
        # TODO proper parsing / mapping of globus statuses to our Status enum
        statuses = {}
        for file_record, event in zip(file_records, globus_events):
            if event['code'] == 'FILE_OK':
                status = Status.SUCCESS
            elif event['is_error']:
                status = Status.FAIL
            statuses[file_record['id']] = status

        return statuses

    def _get_path_at_endpoint(self, file_record: dict) -> Path:
        # to be overridden by the SDSCPatcher
        return self.globus.to_address(
            file_record['relative_path'],
            self.globus.endpoints[file_record['data_repository']]['id'],
        )


"""
 ######  ########   ######   ######
##    ## ##     ## ##    ## ##    ##
##       ##     ## ##       ##
 ######  ##     ##  ######  ##
      ## ##     ##       ## ##
##    ## ##     ## ##    ## ##    ##
 ######  ########   ######   ######
"""


class SDSCPatcher(GlobusPatcher):
    # either make this inherit from GlobusPatcher
    # of from an SSHPatcher
    def __init__(
        self,
        one: ONE,
        # backend: Literal['globus', 'ssh'] = 'globus',  # implement factory pattern?
        **kwargs,
    ):
        super().__init__(**kwargs)

    def _get_path_at_endpoint(self, file_record):
        dataset_id = file_record['dataset'].split('/')[-1]
        relative_path_w_uuid = add_uuid_string(file_record['relative_path'], dataset_id)
        return self.globus.to_address(
            relative_path_w_uuid,
            self.globus_endpoints[file_record['data_repository']],
        )

    # def delete_file(self, file_record: dict, **kwargs):
    #     return super().delete_file(file_record, **kwargs)

    # def delete_files(self, file_records: list[dict], **kwargs):
    #     return super().delete_file(file_records, **kwargs)

    # def patch_file(self, file_record: dict, file: str | Path, **kwargs):
    #     return super().patch_file(file_record, file, **kwargs)

    # def patch_files(self, file_records: list[dict], files: list[str] | list[Path], **kwargs):
    #     return super().patch_file(file_records, files, **kwargs)


"""
#### ########  ##       ########     ###    ########  ######  ##     ## ######## ########
 ##  ##     ## ##       ##     ##   ## ##      ##    ##    ## ##     ## ##       ##     ##
 ##  ##     ## ##       ##     ##  ##   ##     ##    ##       ##     ## ##       ##     ##
 ##  ########  ##       ########  ##     ##    ##    ##       ######### ######   ########
 ##  ##     ## ##       ##        #########    ##    ##       ##     ## ##       ##   ##
 ##  ##     ## ##       ##        ##     ##    ##    ##    ## ##     ## ##       ##    ##
#### ########  ######## ##        ##     ##    ##     ######  ##     ## ######## ##     ##
"""


class IBLPatcher:
    # class that combines the individual patcher classes
    # has implementations of delete_file and delete_flies that are
    # agnostic to the infrastructure by inferring the correct patcher

    # this makes  delete_dataset(s) possible - the file records associated to that dataset
    # are handled by the corresponding patchers

    # implements remove_from_alyx keyword for delete functions

    def __init__(
        self,
        one: ONE,
        dry: bool = False,
        on_error: Literal['raise', 'log'] = 'raise',
        aws_profile: str = 'default',
        aws_region: str = 'us-east-1',
        globus_client_name: str = 'default',  # this might be different for the patching from SDSC to local server
        patch_label: str = 'generic patch label',
    ):
        # patchers registry
        self.patchers = {
            'aws': AWSPatcher(
                one,
                dry=dry,
                aws_profile=aws_profile,
                aws_region=aws_region,
                on_error=on_error,
            ),
            'sdsc': SDSCPatcher(  # TODO pass through the backend
                one,
                dry=dry,
                globus_client_name=globus_client_name,
                patch_label=patch_label,
                on_error=on_error,
            ),
            'local_server': GlobusPatcher(
                one,
                dry=dry,
                globus_client_name=globus_client_name,
                patch_label=patch_label,
                on_error=on_error,
            ),
            # TODO question here: what else?
        }

    def _delete_file_from_alyx(self, file_record_id: str):
        # private helper that handles:
        # alyx rest call, logging, exception handling
        try:
            self.one.alyx.rest('files', 'delete', file_record_id)
            _logger.info(f'deleted file record {file_record_id} from alyx')
        except Exception as e:
            _logger.error(f'failed to delete file record {file_record_id} from alyx')
            if self.on_error == 'raise':
                raise e
            elif self.on_error == 'log':
                return None

    def _delete_file(
        self,
        file_record: dict,
        check_exists: bool = True,
        remove_from_alyx: bool = False,
    ) -> Status:
        # delegate call to corresponding patcher (which will handle dry mode and file checking)
        patcher = self.patchers[get_repository_type_by_name(file_record['repository'])]
        status = patcher.delete_file(file_record, check_exists=check_exists)
        # optionally remove file record from alyx
        if status == Status.SUCCESS and remove_from_alyx and not self.dry:
            self._delete_file_from_alyx(file_record['id'])
        return status

    def _delete_files(
        self,
        file_records: list[dict],
        check_exists: bool = True,
        ignore_missing: bool = False,
        remove_from_alyx: bool = False,
    ) -> dict[str, Status]:
        # get the present repositories (by their names)
        repositories = set(file_record['repository'] for file_record in file_records)
        statuses = {}
        for repository in repositories:
            # subset of file records per repository
            _file_records = list(filter(lambda file_record: file_record[repository] == repository, file_records))
            patcher = self.patchers[get_repository_type_by_name(repository)]
            _statuses = patcher.delete_files(
                _file_records,
                check_exists=check_exists,
                ignore_missing=ignore_missing,
            )
            statuses.update(_statuses)

        # handle removals from alyx
        if remove_from_alyx and not self.dry:
            for file_record_id, status in statuses.items():
                if status == Status.SUCCESS:
                    self._delete_file_from_alyx(file_record_id)

        return statuses

    def _delete_dataset_from_alyx(self, dataset_id):
        # private helper that handles:
        # alyx rest call, logging, exception handling
        try:
            self.one.alyx.rest('datasets', 'delete', dataset_id)
            _logger.info(f'deleted dataset {dataset_id} from alyx')
        except Exception as e:
            _logger.error(f'failed to delete dataset {dataset_id} from alyx')
            if self.on_error == 'raise':
                raise e
            elif self.on_error == 'log':
                return None

    def delete_dataset(
        self,
        dataset: dict,
        remove_from_alyx: bool = False,
    ) -> dict[str, Status]:
        # iterate over file records and remove them one by one
        statuses = {}
        for file_record in dataset['file_records']:
            statuses[file_record['id']] = self._delete_file(file_record, remove_from_alyx=remove_from_alyx)

        # only remove the dataset from alyx if all the file records have been processed successfully
        if all(status == Status.SUCCESS for status in statuses.values()) and not self.dry:
            if remove_from_alyx:
                self._delete_dataset_from_alyx(dataset['id'])

        return statuses

    def delete_datasets(
        self,
        datasets: list[dict],
        check_exists: bool = True,
        ignore_missing: bool = False,
        remove_from_alyx: bool = False,
        per_dataset: bool = True,
    ) -> dict[str, dict[str, Status]]:
        # per_dataset flag to switch between different implementations

        # implementation option 1: iterate over datasets and call delete_dataset
        # pro: simple and clear
        # con: potentially less efficient if many datasets share the same repository
        dataset_statuses = {}
        if per_dataset:
            for dataset in datasets:
                dataset_statuses[dataset['id']] = self.delete_dataset(
                    dataset,
                    check_exists=check_exists,
                    remove_from_alyx=remove_from_alyx,
                )

        # implementation option 2: gather all file records and 'bulk delete'
        # pro: potentially more efficient if many datasets share the same repository
        # (more efficient for globus?)
        else:
            file_records = []
            id_map = {}  # file record 2 dataset - keep this for status reordering
            for dataset in datasets:
                file_records.extent(dataset['file_records'])
                for file_record in datasets['file_records']:
                    id_map[file_record['id']] = dataset['id']

            file_record_statuses = self._delete_files(
                file_records,
                check_exists=check_exists,
                ignore_missing=ignore_missing,
                remove_from_alyx=remove_from_alyx,
            )
            # these statuses are dict[file_record_id, Status]
            # reorder these statuses back to the level of datasets
            for dataset in datasets:
                dataset_statuses[dataset['id']] = {}
            for file_record_id, status in file_record_statuses.items():
                dataset_id = id_map[file_record_id]
                dataset_statuses[dataset_id][file_record_id] = status

        if remove_from_alyx and not self.dry:
            for dataset_id, file_record_statuses in dataset_statuses.items():
                if all(status == Status.SUCCESS for status in file_record_statuses.values()):
                    self._delete_dataset_from_alyx(dataset_id)

    def patch_dataset(
        self,
        dataset: dict,
        file: str | Path,
        check_exists: bool = True,
    ) -> dict[str, Status]:
        # iterate over the file records and patch them individually
        statuses = {}
        for file_record in dataset['file_records']:
            patcher = self.patchers[get_repository_type_by_name(file_record['repository'])]
            statuses[file_record['id']] = patcher.patch_file(file_record, file, check_exists=check_exists)

        return statuses

    def patch_datasets(
        self,
        datasets: list[dict],
        files: list[str] | list[Path],
        check_exists: bool = True,
        per_dataset: bool = True,
    ) -> dict[str, dict[str, Status]]:
        # TODO have a variant here of check_file_and_file_records but adapted for datasets

        # iterate over datasets and patch one by one
        statuses = {}
        if per_dataset:
            for dataset, file in zip(datasets, files):
                statuses[dataset['id']] = {}
                for file_record in dataset['file_records']:
                    patcher = get_repository_type_by_name[file_record['repository']]
                    status = patcher.patch_file(file_record, file, check_exists=check_exists)
                    statuses[dataset['id']][file_record['id']] = status

        else:
            raise NotImplementedError
        return statuses

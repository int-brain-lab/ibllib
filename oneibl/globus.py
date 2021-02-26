import logging
import time
from datetime import datetime, timedelta

from globus_sdk.exc import TransferAPIError
import globus_sdk

from ibllib.io import globus
from oneibl.one import OneAlyx
import oneibl.params

par = oneibl.params.get()
_logger = logging.getLogger('ibllib')
POLL = (5, 60)  # min max seconds between pinging server
TIMEOUT = 24*60*60  # seconds before timeout
status_map = {
    'ACTIVE': ('QUEUED', 'ACTIVE'),
    'FAILED': ('ENDPOINT_ERROR', 'PERMISSION_DENIED', 'CONNECT_FAILED'),
    'INACTIVE': 'PAUSED_BY_ADMIN'
}


class OneGlobus(OneAlyx):

    def __init__(self, **kwargs):
        # Init connection to the database
        super(OneGlobus, self).__init__(**kwargs)
        # Init connection to Globus if needed
        self._tc = globus.login_auto(par.GLOBUS_CLIENT_ID, str_app='globus_one')

    def setup(self):
        super(OneGlobus, self).setup()
        globus.login_auto(par.GLOBUS_CLIENT_ID, str_app='globus_one')

    def download(self, eids):
        LOCAL_REPO = '0ec47586-3a19-11eb-b173-0ee0d5d9299f'
        REMOTE_REPO = par.DATA_SERVER_GLOBUS_ENDPOINT_ID
        for eid in eids:
            session_path = self.path_from_eid(eid)
            assert session_path is not None
            relative_url = '/'.join(session_path.parts[-5:])
            try:  # Check path exists
                self._tc.operation_ls(REMOTE_REPO, path=relative_url)
            except TransferAPIError as ex:
                _logger.error(f'Remote session does not exist {relative_url}')
                raise ex

            # Create the destination path if it does not exist
            session_path.mkdir(parents=True, exist_ok=True)  # Globus can't make parents
            dst_directory = globus.as_globus_path(session_path)
            try:
                self._tc.operation_ls(LOCAL_REPO, path=dst_directory)
            except TransferAPIError as ex:
                if ex.http_status == 404:
                    # Directory not found; create it
                    try:
                        self._tc.operation_mkdir(LOCAL_REPO, dst_directory)
                        _logger.info(f'Created directory: {dst_directory}')
                    except TransferAPIError as tapie:
                        _logger.error(f'Failed to create directory: {tapie.message}')
                        raise tapie
                else:
                    raise ex
            # Create transfer object
            transfer_object = globus_sdk.TransferData(
                self._tc,
                source_endpoint=REMOTE_REPO,
                destination_endpoint=LOCAL_REPO,
                verify_checksum=True,
                delete_destination_extra=False,
                sync_level='mtime',
                label=eid,
                deadline=datetime.now() + timedelta(0, TIMEOUT)
            )

            # add any number of items to the submission data
            transfer_object.add_item(relative_url, dst_directory, recursive=True)
            response = self._tc.submit_transfer(transfer_object)
            assert round(response.http_status / 100) == 2  # Check for 20x status

            # What for transfer to complete
            task_id = response.data['task_id']
            last_status = None
            files_transferred = None
            files_skipped = 0
            subtasks_failed = 0
            poll = POLL[0]
            MAX_WAIT = 60 * 60
            # while not gtc.task_wait(task_id, timeout=WAIT):
            running = True
            while running:
                """Possible statuses = ('ACTIVE', 'INACTIVE', 'FAILED', 'SUCCEEDED')
                Nice statuses = (None, 'OK', 'Queued', 'PERMISSION_DENIED',
                                 'ENDPOINT_ERROR', 'CONNECT_FAILED', 'PAUSED_BY_ADMIN')
                """
                tr = self._tc.get_task(task_id)
                detail = (
                    'ACTIVE'
                    if (tr.data['nice_status']) == 'OK'
                    else (tr.data['nice_status'] or tr.data['status']).upper()
                )
                status = next((k for k, v in status_map.items() if detail in v), tr.data['status'])
                running = tr.data['status'] == 'ACTIVE' and detail in ('ACTIVE', 'QUEUED')
                if files_skipped != tr.data['files_skipped']:
                    files_skipped = tr.data['files_skipped']
                    _logger.info(f'Skipping {files_skipped} files....')
                if last_status != status or files_transferred != tr.data['files_transferred']:
                    files_transferred = tr.data['files_transferred']
                    total_files = tr.data['files'] - tr.data['files_skipped']
                    if status == 'FAILED' or detail in status_map['FAILED']:
                        _logger.error(f'Transfer {status}: {tr.data["fatal_error"] or detail}')
                        # If still active and error unlikely to resolve by itself, cancel the task
                        if tr.data['status'] == 'ACTIVE' and detail != 'CONNECT_FAILED':
                            self._tc.cancel_task(task_id)
                            _logger.warning('Transfer CANCELLED')
                    elif status == 'INACTIVE' or detail == 'PAUSED_BY_ADMIN':
                        _logger.info(f'Transfer INACTIVE: {detail}')
                    else:
                        _logger.info(
                            f'Transfer {status}: {files_transferred} of {total_files} files transferred')
                        # Report failed subtasks
                        new_failed = tr['subtasks_expired'] + tr['subtasks_failed']
                        if new_failed != subtasks_failed:
                            _logger.warning(
                                f'{abs(new_failed - subtasks_failed)} sub-tasks expired or failed')
                            subtasks_failed = new_failed
                    last_status = status
                    poll = POLL[0]
                else:
                    poll = min((poll * 2, POLL[1]))
                time.sleep(poll)

import logging
from pathlib import Path, PosixPath
import re
import subprocess
import os

from ibllib.ephys import sync_probes
from ibllib.pipes import ephys_preprocessing as ephys
from oneibl.patcher import FTPPatcher
from one.api import ONE

_logger = logging.getLogger('ibllib')

FLATIRON_HOST = 'ibl.flatironinstitute.org'
FLATIRON_PORT = 61022
FLATIRON_USER = 'datauser'
root_path = '/mnt/s0/Data/'


def _run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    info, error = process.communicate()
    if process.returncode != 0:
        return None, error.decode('utf-8')
    else:
        return info.decode('utf-8').strip(), None


def job_transfer_ks2(probe_path):

    assert(isinstance(probe_path, str))

    def _get_volume_usage_percentage(vol):
        cmd = f'df {vol}'
        res, _ = _run_command(cmd)
        size_list = re.split(' +', res.split('\n')[-1])
        per_usage = int(size_list[4][:-1])
        return per_usage

    # First check disk availability
    space = _get_volume_usage_percentage('/mnt/s0')
    # If we are less than 80% full we can transfer more stuff
    if space < 80:
        # Transfer data from flatiron to s3
        cmd = f'ssh -i ~/.ssh/mayo_alyx.pem -p {FLATIRON_PORT} ' \
              f'{FLATIRON_USER}@{FLATIRON_HOST} ./transfer_to_aws.sh {probe_path}'
        result, error = _run_command(cmd)

        # Check that command has run as expected and output info to logger
        if not result:
            _logger.error(f'{probe_path}: Could not transfer data from FlatIron to s3 \n'
                          f'Error: {error}')
            return
        else:
            _logger.info(f'{probe_path}: Data transferred from FlatIron to s3')

        # Transfer data from s3 to /mnt/s0/Data on aws
        session = str(PosixPath(probe_path).parent.parent)
        cmd = f'aws s3 sync s3://ibl-ks2-storage/{session} "/mnt/s0/Data/{session}"'
        result, error = _run_command(cmd)

        # Check that command has run as expected and output info to logger
        if not result:
            _logger.error(f'{probe_path}: Could not transfer data from s3 to aws \n'
                          f'Error: {error}')
            return
        else:
            _logger.info(f'{probe_path}: Data transferred from s3 to aws')

        # Rename the files to get rid of eid associated with each dataset
        session_path = Path(root_path).joinpath(session)
        for file in session_path.glob('**/*'):
            if len(Path(file.stem).suffix) == 37:
                file.rename(Path(file.parent, str(Path(file.stem).stem) + file.suffix))
                _logger.info(f'Renamed dataset {file.stem} to {str(Path(file.stem).stem)}')
            else:
                _logger.warning(f'Dataset {file.stem} not renamed')
                continue

        # Create a sort_me.flag
        cmd = f'touch /mnt/s0/Data/{session}/sort_me.flag'
        result, error = _run_command(cmd)
        _logger.info(f'{session}: sort_me.flag created')

        # Remove files from s3
        cmd = f'aws s3 rm --recursive s3://ibl-ks2-storage/{session}'
        result, error = _run_command(cmd)
        if not result:
            _logger.error(f'{session}: Could not remove data from s3 \n'
                          f'Error: {error}')
            return
        else:
            _logger.info(f'{session}: Data removed from s3')

        return


def job_run_ks2():

    # Look for flag files in /mnt/s0/Data and sort them in order of date they were created
    flag_files = list(Path(root_path).glob('**/sort_me.flag'))
    flag_files.sort(key=os.path.getmtime)

    # Start with the oldest flag
    session_path = flag_files[0].parent
    session = str(PosixPath(*session_path.parts[4:]))
    flag_files[0].unlink()

    # Instantiate one
    one = ONE(cache_rest=None)

    # sync the probes
    status, sync_files = sync_probes.sync(session_path)

    if not status:
        _logger.error(f'{session}: Could not sync probes')
        return
    else:
        _logger.info(f'{session}: Probes successfully synced')

    # run ks2
    task = ephys.SpikeSorting(session_path, one=one)
    status = task.run()

    if status != 0:
        _logger.error(f'{session}: Could not run ks2')
        return
    else:
        _logger.info(f'{session}: ks2 successfully completed')

        # Run the cell qc
        # qc_file = []

        # Register and upload files to FTP Patcher
        outfiles = task.outputs
        ftp_patcher = FTPPatcher(one=one)
        ftp_patcher.create_dataset(path=outfiles, created_by=one._par.ALYX_LOGIN)

        # Remove everything apart from alf folder and spike sorter folder
        # Don't do this for now unitl we are sure it works for 3A and 3B!!
        # cmd = f'rm -r {session_path}/raw_ephys_data rm -r {session_path}/raw_behavior_data'
        # result, error = _run_command(cmd)

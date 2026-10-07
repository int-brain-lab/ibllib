"""Lab server pipeline construction and task runner.

This is the module called by the job services on the lab servers.  See
iblscripts/deploy/serverpc/crons for the service scripts that employ this module.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
import traceback
import importlib.metadata

from one.api import ONE
from one.webclient import AlyxClient
from one.remote.globus import get_lab_from_endpoint_id, get_local_endpoint_id
from one.alf.spec import is_session_path
from one.alf.path import session_path_parts

from ibllib import __version__ as ibllib_version
from ibllib.pipes import tasks
from ibllib.pipes.routing import task_env
from ibllib.time import date2isostr
from ibllib.oneibl.registration import IBLRegistrationClient
from ibllib.oneibl.data_handlers import get_local_data_repository
from ibllib.io.session_params import read_params
from ibllib.pipes.dynamic_pipeline import make_pipeline, acquisition_description_legacy_session

_logger = logging.getLogger(__name__)
LARGE_TASKS = ['EphysVideoCompress', 'TrainingVideoCompress', 'SpikeSorting', 'EphysDLC', 'MesoscopePreprocess']


def _run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info, error = process.communicate()
    if process.returncode != 0:
        return None
    else:
        return info.decode('utf-8').strip()


def _get_volume_usage(vol, label=''):
    cmd = f'df {vol}'
    res = _run_command(cmd)
    # size_list = ['/dev/sdc1', '1921802500', '1427128132', '494657984', '75%', '/datadisk']
    size_list = re.split(' +', res.split('\n')[-1])
    fac = 1024**2
    d = {
        'total': int(size_list[1]) / fac,
        'used': int(size_list[2]) / fac,
        'available': int(size_list[3]) / fac,
        'volume': size_list[5],
    }
    return {f'{label}_{k}': d[k] for k in d}


def report_health(alyx):
    """
    Get a few indicators and label the json field of the corresponding lab with them.
    """
    status = {
        'python_version': sys.version,
        'ibllib_version': ibllib_version,
        'phylib_version': importlib.metadata.version('phylib'),
        'local_time': date2isostr(datetime.now()),
    }
    status.update(_get_volume_usage('/mnt/s0/Data', 'raid'))
    status.update(_get_volume_usage('/', 'system'))

    data_repos = alyx.rest('data-repository', 'list', globus_endpoint_id=get_local_endpoint_id())

    for dr in data_repos:
        alyx.json_field_update(endpoint='data-repository', uuid=dr['name'], field_name='json', data=status)


def job_creator(root_path, one=None, dry=False, rerun=False):
    """
    Create new sessions and pipelines.

    Server function that will look for 'raw_session.flag' files and for each:
     1) create the session on Alyx
     2) create the tasks to be run on Alyx

    For legacy sessions the raw data are registered separately, instead of within a pipeline task.

    Parameters
    ----------
    root_path : str, pathlib.Path
        Main path containing sessions or a session path.
    one : one.api.OneAlyx
        An ONE instance for registering the session(s).
    dry : bool
        If true, simply log the session_path(s) found, without registering anything.
    rerun : bool
        If true and session pipeline tasks already exist, set them all to waiting.

    Returns
    -------
    list of ibllib.pipes.tasks.Pipeline
        The pipelines created.
    list of dicts
        A list of any datasets registered (only for legacy sessions)
    """
    _logger.info('Start looking for new sessions...')
    if not one:
        one = ONE(cache_rest=None)
    rc = IBLRegistrationClient(one=one)
    flag_files = Path(root_path).glob('*/????-??-??/*/raw_session.flag')
    flag_files = filter(lambda x: is_session_path(x.parent), flag_files)
    pipes = []
    all_datasets = []
    for flag_file in flag_files:
        session_path = flag_file.parent
        if session_path_parts(session_path)[1] in ('test', 'test_subject'):
            _logger.debug('skipping test session %s', session_path)
            continue
        _logger.info(f'creating session for {session_path}')
        if dry:
            continue
        try:
            # if the subject doesn't exist in the database, skip
            rc.register_session(session_path, file_list=False)

            # NB: all sessions now extracted using dynamic pipeline
            if read_params(session_path) is None:
                # Create legacy experiment description file
                acquisition_description_legacy_session(session_path, save=True)
            pipe = make_pipeline(session_path, one=one)
            if rerun:
                rerun__status__in = '__all__'
            else:
                rerun__status__in = ['Waiting']
            pipe.create_alyx_tasks(rerun__status__in=rerun__status__in)
            if pipe.planner_errors:
                # Keep the flag file so that the missing tasks are created on the next run
                _logger.error(
                    'Failed to plan %s tasks for session %s; keeping flag file to retry',
                    ', '.join(pipe.planner_errors),
                    session_path.relative_to(root_path),
                )
            else:
                flag_file.unlink()
            if pipe is not None:
                pipes.append(pipe)
        except Exception:
            _logger.error('Failed to register session %s:\n%s', session_path.relative_to(root_path), traceback.format_exc())
            continue

    return pipes, all_datasets


def list_available_envs(root=Path.home() / 'Documents/PYTHON/envs'):
    """
    List all the envs within `root` dir.

    NB: Environment labels don't necessarily match the venv directory names; use
    :func:`ibllib.pipes.routing.installed_envs` to list the installed environment labels.

    Parameters
    ----------
    root : str, pathlib.Path
        The directory containing venvs.

    Returns
    -------
    list of str
        A list of envs, including None (assumed to be base iblenv).
    """
    try:
        envs = filter(Path.is_dir, Path(root).iterdir())
        return [None, *sorted(x.name for x in envs)]
    except FileNotFoundError:
        return [None]


def _waiting_tasks(alyx, lab=None):
    """
    Query the waiting tasks of a lab for the local data repository.

    Parameters
    ----------
    alyx : one.webclient.AlyxClient
        An Alyx instance.
    lab : str
        Lab name as per Alyx, otherwise try to infer from local Globus install.

    Returns
    -------
    list of dict, None
        A list of Alyx tasks with a 'Waiting' status, or None if the lab could not be determined.
    """
    if lab is None:
        _logger.debug('Trying to infer lab from globus installation')
        lab = get_lab_from_endpoint_id(alyx=alyx)
    if lab is None:
        _logger.error('No lab provided or found')
        return  # if the lab is none, this will return empty tasks each time
    data_repo = get_local_data_repository(alyx)
    return alyx.rest(
        'tasks', 'list', status='Waiting', django=f'session__lab__name__in,{lab},data_repository__name,{data_repo}', no_cache=True
    )


def list_queued_envs(one=None, lab=None):
    """
    The set of all envs in the list of waiting tasks.

    The environment of each task is determined from its executable, without importing the task class.

    Parameters
    ----------
    one : one.api.OneAlyx
        An instance of ONE.
    lab : str
        Lab name as per Alyx, otherwise try to infer from local Globus install.

    Returns
    -------
    set
        All environments required to process waiting tasks.
    """
    one = one or ONE(mode='remote', cache_rest=None)
    return {task_env(t['executable']) for t in _waiting_tasks(one.alyx, lab=lab) or []}


def is_job_size(task, mode):
    """
    Check whether a task is of a given job size.

    NB: This imports the task class.

    Parameters
    ----------
    task : dict
        An Alyx task dictionary.
    mode : {'all', 'small', 'large'}
        The job size to check.

    Returns
    -------
    bool
        True if the task class job size matches the mode (or mode is 'all'). False if the task
        class could not be imported.
    """
    if mode == 'all':
        return True
    try:
        return tasks.str2class(task['executable']).job_size == mode
    except (ImportError, AttributeError):
        _logger.error('Task %s not found in this env', task['executable'])
        return False


def task_queue(mode='all', lab=None, alyx=None, env=(None,)):
    """
    Query waiting jobs from the specified Lab

    The environment of each task is determined from its executable (see
    :func:`ibllib.pipes.routing.task_env`), so only the task classes of the given environments are
    imported (to determine the job size).

    Parameters
    ----------
    mode : {'all', 'small', 'large'}
        Whether to return all waiting tasks, or only small or large (specified in LARGE_TASKS) jobs.
    lab : str
        Lab name as per Alyx, otherwise try to infer from local Globus install.
    alyx : one.webclient.AlyxClient
        An Alyx instance.
    env : str, list
        One or more environment labels to filter by, where None is the base environment. See
        :data:`ibllib.pipes.routing.ROUTES`.

    Returns
    -------
    list of dict
        A list of Alyx tasks associated with `lab` that have a 'Waiting' status.
    """
    env = (env,) if env is None or isinstance(env, str) else tuple(env)
    alyx = alyx or AlyxClient(cache_rest=None)
    waiting_tasks = _waiting_tasks(alyx, lab=lab)
    if waiting_tasks is None:
        return
    # Filter tasks by environment, then by size
    filtered_tasks = (t for t in waiting_tasks if task_env(t['executable']) in env)
    filtered_tasks = filter(lambda t: is_job_size(t, mode), filtered_tasks)
    # Order tasks by priority
    sorted_tasks = sorted(filtered_tasks, key=lambda d: d['priority'], reverse=True)

    return sorted_tasks


def tasks_runner(subjects_path, tasks_dict, one=None, dry=False, count=5, time_out=None, **kwargs):
    """
    Function to run a list of tasks (task dictionary from Alyx query) on a local server

    Parameters
    ----------
    subjects_path : str, pathlib.Path
        The location of the subject session folders, e.g. '/mnt/s0/Data/Subjects'.
    tasks_dict : list of dict
        A list of tasks to run. Typically the output of `task_queue`.
    one : one.api.OneAlyx
        An instance of ONE.
    dry : bool, default=False
        If true, simply prints the full session paths and task names without running the tasks.
    count : int, default=5
        The maximum number of tasks to run from the tasks_dict list.
    time_out : float, optional
        The time in seconds to run tasks before exiting. If set this will run tasks until the
        timeout has elapsed.  NB: Only checks between tasks and will not interrupt a running task.
    kwargs
        See ibllib.pipes.tasks.run_alyx_task.

    Returns
    -------
    list of pathlib.Path
        A list of datasets registered to Alyx.
    """
    if one is None:
        one = ONE(cache_rest=None)
    tstart = time.time()
    c = 0
    last_session = None
    all_datasets = []
    for tdict in tasks_dict:
        # if the count is reached or if the time_out has been elapsed, break the loop and return
        if c >= count or (time_out and time.time() - tstart > time_out):
            break
        # reconstruct the session local path. As many jobs belong to the same session
        # cache the result
        if last_session != tdict['session']:
            ses = one.alyx.rest('sessions', 'list', django=f'pk,{tdict["session"]}')[0]
            session_path = Path(subjects_path).joinpath(Path(ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3)))
            last_session = tdict['session']
        if dry:
            print(session_path, tdict['name'])
        else:
            task, dsets = tasks.run_alyx_task(tdict=tdict, session_path=session_path, one=one, **kwargs)
            if dsets:
                all_datasets.extend(dsets)
                c += 1
    return all_datasets

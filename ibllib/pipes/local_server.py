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
import importlib
import importlib.metadata

from one.api import ONE
from one.webclient import AlyxClient
from one.remote.globus import get_lab_from_endpoint_id, get_local_endpoint_id

from ibllib import __version__ as ibllib_version
from ibllib.io.extractors.base import get_pipeline, get_task_protocol, get_session_extractor_type
from ibllib.pipes import tasks, training_preprocessing, ephys_preprocessing
from ibllib.time import date2isostr
from ibllib.oneibl.registration import IBLRegistrationClient, register_session_raw_data, get_lab
from ibllib.oneibl.data_handlers import get_local_data_repository
from ibllib.io.session_params import read_params
from ibllib.pipes.dynamic_pipeline import make_pipeline, acquisition_description_legacy_session

_logger = logging.getLogger(__name__)
LARGE_TASKS = [
    'EphysVideoCompress', 'TrainingVideoCompress', 'SpikeSorting', 'EphysDLC', 'MesoscopePreprocess'
]


def _get_pipeline_class(session_path, one):
    pipeline = get_pipeline(session_path)
    if pipeline == 'training':
        PipelineClass = training_preprocessing.TrainingExtractionPipeline
    elif pipeline == 'ephys':
        PipelineClass = ephys_preprocessing.EphysExtractionPipeline
    else:
        # try and look if there is a custom extractor in the personal projects extraction class
        import projects.base
        task_type = get_session_extractor_type(session_path)
        PipelineClass = projects.base.get_pipeline(task_type)
    _logger.info(f"Using {PipelineClass} pipeline for {session_path}")
    return PipelineClass(session_path=session_path, one=one)


def _run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
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
    fac = 1024 ** 2
    d = {'total': int(size_list[1]) / fac,
         'used': int(size_list[2]) / fac,
         'available': int(size_list[3]) / fac,
         'volume': size_list[5]}
    return {f"{label}_{k}": d[k] for k in d}


def report_health(one):
    """
    Get a few indicators and label the json field of the corresponding lab with them.
    """
    status = {'python_version': sys.version,
              'ibllib_version': ibllib_version,
              'phylib_version': importlib.metadata.version('phylib'),
              'local_time': date2isostr(datetime.now())}
    status.update(_get_volume_usage('/mnt/s0/Data', 'raid'))
    status.update(_get_volume_usage('/', 'system'))

    data_repos = one.alyx.rest('data-repository', 'list', globus_endpoint_id=get_local_endpoint_id())

    for dr in data_repos:
        one.alyx.json_field_update(endpoint='data-repository', uuid=dr['name'], field_name='json', data=status)


def job_creator(root_path, one=None, dry=False, rerun=False, max_md5_size=None):
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
    max_md5_size : int
        (legacy sessions) The maximum file size to calculate the MD5 hash sum for.

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
    flag_files = list(Path(root_path).glob('**/raw_session.flag'))
    pipes = []
    all_datasets = []
    for flag_file in flag_files:
        session_path = flag_file.parent
        _logger.info(f'creating session for {session_path}')
        if dry:
            continue
        try:
            # if the subject doesn't exist in the database, skip
            rc.register_session(session_path, file_list=False)

            # See if we need to create a dynamic pipeline
            experiment_description_file = read_params(session_path)
            if experiment_description_file is not None:
                pipe = make_pipeline(session_path, one=one)
            else:
                # Create legacy experiment description file
                acquisition_description_legacy_session(session_path, save=True)
                lab = get_lab(session_path, one.alyx)  # Can be set to None to do this Alyx-side if using ONE v1.20.1
                _, dsets = register_session_raw_data(session_path, one=one, max_md5_size=max_md5_size, labs=lab)
                if dsets:
                    all_datasets.extend(dsets)
                pipe = _get_pipeline_class(session_path, one)
                if pipe is None:
                    task_protocol = get_task_protocol(session_path)
                    _logger.info(f'Session task protocol {task_protocol} has no matching pipeline pattern {session_path}')
            if rerun:
                rerun__status__in = '__all__'
            else:
                rerun__status__in = ['Waiting']
            pipe.create_alyx_tasks(rerun__status__in=rerun__status__in)
            flag_file.unlink()
            if pipe is not None:
                pipes.append(pipe)
        except Exception:
            _logger.error('Failed to register session %s:\n%s', session_path.relative_to(root_path), traceback.format_exc())
            continue

    return pipes, all_datasets


def task_queue(mode='all', lab=None, alyx=None):
    """
    Query waiting jobs from the specified Lab

    Parameters
    ----------
    mode : {'all', 'small', 'large'}
        Whether to return all waiting tasks, or only small or large (specified in LARGE_TASKS) jobs.
    lab : str
        Lab name as per Alyx, otherwise try to infer from local Globus install.
    alyx : one.webclient.AlyxClient
        An Alyx instance.

    Returns
    -------
    list of dict
        A list of Alyx tasks associated with `lab` that have a 'Waiting' status.
    """
    alyx = alyx or AlyxClient(cache_rest=None)
    if lab is None:
        _logger.debug('Trying to infer lab from globus installation')
        lab = get_lab_from_endpoint_id(alyx=alyx)
    if lab is None:
        _logger.error('No lab provided or found')
        return  # if the lab is none, this will return empty tasks each time
    data_repo = get_local_data_repository(alyx)
    # Filter for tasks
    tasks_all = alyx.rest('tasks', 'list', status='Waiting',
                          django=f'session__lab__name__in,{lab},data_repository__name,{data_repo}', no_cache=True)
    if mode == 'all':
        waiting_tasks = tasks_all
    else:
        small_jobs = []
        large_jobs = []
        for t in tasks_all:
            strmodule, strclass = t['executable'].rsplit('.', 1)
            classe = getattr(importlib.import_module(strmodule), strclass)
            job_size = classe.job_size
            if job_size == 'small':
                small_jobs.append(t)
            else:
                large_jobs.append(t)
    if mode == 'small':
        waiting_tasks = small_jobs
    elif mode == 'large':
        waiting_tasks = large_jobs

    # Order tasks by priority
    sorted_tasks = sorted(waiting_tasks, key=lambda d: d['priority'], reverse=True)

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
    **kwargs
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
            ses = one.alyx.rest('sessions', 'list', django=f"pk,{tdict['session']}")[0]
            session_path = Path(subjects_path).joinpath(
                Path(ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3)))
            last_session = tdict['session']
        if dry:
            print(session_path, tdict['name'])
        else:
            task, dsets = tasks.run_alyx_task(tdict=tdict, session_path=session_path,
                                              one=one, **kwargs)
            if dsets:
                all_datasets.extend(dsets)
                c += 1
    return all_datasets

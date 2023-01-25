import logging
import time
from datetime import datetime
from pathlib import Path
import pkg_resources
import re
import subprocess
import sys
import traceback
import importlib

from one.api import ONE

from ibllib.io.extractors.base import get_pipeline, get_task_protocol, get_session_extractor_type
from ibllib.pipes import tasks, training_preprocessing, ephys_preprocessing
from ibllib.time import date2isostr
from ibllib.oneibl.registration import IBLRegistrationClient, register_session_raw_data, get_lab
from ibllib.oneibl.data_handlers import get_local_data_repository
from ibllib.io.session_params import read_params
from ibllib.pipes.dynamic_pipeline import make_pipeline, acquisition_description_legacy_session

_logger = logging.getLogger(__name__)
LARGE_TASKS = ['EphysVideoCompress', 'TrainingVideoCompress', 'SpikeSorting', 'EphysDLC']


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
    Get a few indicators and label the json field of the corresponding lab with them
    """
    status = {'python_version': sys.version,
              'ibllib_version': pkg_resources.get_distribution("ibllib").version,
              'phylib_version': pkg_resources.get_distribution("phylib").version,
              'local_time': date2isostr(datetime.now())}
    status.update(_get_volume_usage('/mnt/s0/Data', 'raid'))
    status.update(_get_volume_usage('/', 'system'))

    lab_names = get_lab(one.alyx)
    for ln in lab_names:
        one.alyx.json_field_update(endpoint='labs', uuid=ln, field_name='json', data=status)


def job_creator(root_path, one=None, dry=False, rerun=False, max_md5_size=None):
    """
    Server function that will look for creation flags and for each:
    1) create the sessions on Alyx
    2) register the corresponding raw data files on Alyx
    3) create the tasks to be run on Alyx
    :param root_path: main path containing sessions or session path
    :param one
    :param dry
    :param rerun
    :param max_md5_size
    :return:
    """
    if not one:
        one = ONE(cache_rest=None)
    rc = IBLRegistrationClient(one=one)
    flag_files = list(Path(root_path).glob('**/raw_session.flag'))
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
                labs = ','.join(get_lab(one.alyx))
                files, dsets = register_session_raw_data(session_path, one=one, max_md5_size=max_md5_size, labs=labs)
                if dsets is not None:
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
        except Exception:
            _logger.error(traceback.format_exc())
            _logger.warning(f'Creating session / registering raw datasets {session_path} errored')
            continue

    return all_datasets


def task_queue(mode='all', lab=None, one=None):
    """
    Query waiting jobs from the specified Lab
    :param mode: Whether to return all waiting tasks, or only small or large (specified in LARGE_TASKS) jobs
    :param lab: lab name as per Alyx, otherwise try to infer from local globus install
    :param one: ONE instance
    -------

    """
    if one is None:
        one = ONE(cache_rest=None)
    if lab is None:
        _logger.debug("Trying to infer lab from globus installation")
        lab = get_lab(one.alyx)
    if lab is None:
        _logger.error("No lab provided or found")
        return  # if the lab is none, this will return empty tasks each time
    data_repo = get_local_data_repository(one)
    # Filter for tasks
    tasks_all = one.alyx.rest('tasks', 'list', status='Waiting',
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
    :param subjects_path:
    :param tasks_dict:
    :param one:
    :param dry:
    :param count: maximum number of tasks to run
    :param time_out: between each task, if time elapsed is greater than time out, returns (seconds)
    :param kwargs:
    :return: list of dataset dictionaries
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

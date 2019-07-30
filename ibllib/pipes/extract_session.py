#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, October 11th 2018, 12:11:13 pm
"""
Find task name
Check if extractors for specific task exist
Extract data OR return error to user saying that the task has no extractors
"""
import logging
from pathlib import Path
import traceback

from ibllib.io.extractors import (ephys_trials, ephys_fpga,
                                  biased_wheel, biased_trials,
                                  training_trials, training_wheel)
from ibllib.io import raw_data_loaders as raw
import ibllib.io.flags as flags

logger_ = logging.getLogger('ibllib.alf')


# this is a decorator to add a logfile to each extraction and registration on top of the logging
def log2sessionfile(func):
    def func_wrapper(sessionpath, *args, **kwargs):
        fh = logging.FileHandler(Path(sessionpath).joinpath('extract_register.log'))
        str_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        fh.setFormatter(logging.Formatter(str_format))
        logger_.addHandler(fh)
        f = func(sessionpath, *args, **kwargs)
        fh.close()
        logger_.removeHandler(fh)
        return f
    return func_wrapper


def get_task_extractor_type(task_name):
    """
    Splits the task name according to naming convention:
    -   ignores everything
    _iblrig_tasks_biasedChoiceWorld3.7.0 returns "biased"
    _iblrig_tasks_trainingChoiceWorld3.6.0 returns "training'
    :param task_name:
    :return:
    """
    if '_biasedChoiceWorld' in task_name:
        return 'biased'
    elif '_trainingChoiceWorld' in task_name:
        return 'training'
    elif 'ephysChoiceWorld' in task_name:
        return 'ephys'
    elif task_name.startswith('_iblrig_tasks_ephys_certification'):
        return 'sync_ephys'


def get_session_extractor_type(session_path):
    """
    From a session path, loads the settings file, finds the task and checks if extractors exist
    task names examples:
    :param session_path:
    :return: bool
    """
    settings = raw.load_settings(session_path)
    if settings is None:
        logger_.error(f'ABORT: No data found in "raw_behavior_data" folder {session_path}')
        return False
    extractor_type = get_task_extractor_type(settings['PYBPOD_PROTOCOL'])
    if extractor_type:
        return extractor_type
    else:
        logger_.warning(str(session_path) +
                        f" No extractors were found for {extractor_type} ChoiceWorld")
        return False


def is_extracted(session_path):
    sp = Path(session_path)
    if (sp / 'alf').exists():
        return True
    else:
        return False


@log2sessionfile
def from_path(session_path, force=False, save=True):
    """
    Extract a session from full ALF path (ex: '/scratch/witten/ibl_witten_01/2018-12-18/001')

    :param force: (False) overwrite existing files
    :param save: (True) boolean or list of ALF file names to extract
    :return: None
    """
    logger_.info('Extracting ' + str(session_path))
    extractor_type = get_session_extractor_type(session_path)
    if is_extracted(session_path) and not force:
        logger_.info(f"Session {session_path} already extracted.")
        return
    if extractor_type == 'training':
        settings, data = raw.load_bpod(session_path)
        logger_.info('training session on ' + settings['PYBPOD_BOARD'])
        training_trials.extract_all(session_path, data=data, save=save)
        training_wheel.extract_all(session_path, bp_data=data, save=save)
        logger_.info('session extracted \n')  # timing info in log
    if extractor_type == 'biased':
        settings, data = raw.load_bpod(session_path)
        logger_.info('biased session on ' + settings['PYBPOD_BOARD'])
        biased_trials.extract_all(session_path, data=data, save=save)
        biased_wheel.extract_all(session_path, bp_data=data, save=save)
        logger_.info('session extracted \n')  # timing info in log
    if extractor_type == 'ephys':
        data = raw.load_data(session_path)
        ephys_trials.extract_all(session_path, data=data, save=save)
        ephys_fpga.extract_all(session_path, save=save)
    if extractor_type == 'sync_ephys':
        ephys_fpga.extract_sync(session_path, save=save)


def bulk(subjects_folder, dry=False, glob_flag='**/extract_me.flag'):
    ses_path = Path(subjects_folder).glob(glob_flag)
    for p in ses_path:
        # the flag file may contains specific file names for a targeted extraction
        save = flags.read_flag_file(p)
        if dry:
            print(p)
            continue
        try:
            from_path(p.parent, force=True, save=save)
        except Exception as e:
            error_message = str(p.parent) + ' failed extraction' + '\n    ' + str(e)
            error_message += traceback.format_exc()
            err_file = p.parent.joinpath('extract_me.error')
            p.replace(err_file)
            with open(err_file, 'w+') as f:
                f.write(error_message)
            logger_.error(error_message)
            continue
        p.unlink()
        flags.write_flag_file(p.parent.joinpath('register_me.flag'), file_list=save)


if __name__ == "__main__":
    sess = '/home/nico/Projects/IBL/scratch/test_iblrig_data/Subjects/IBL-T1/2019-02-19/001'  # noqa
    bulk(sess)
    print('.')

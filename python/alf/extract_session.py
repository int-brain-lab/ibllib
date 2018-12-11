# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, October 11th 2018, 12:11:13 pm
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 11-10-2018 12:11:26.2626
"""
Find task name
Check if extractors for specific task exist
Extract data OR return error to user saying that the task has no extractors
"""
import os
from pathlib import Path

from alf.extractors import *
from ibllib.io import raw_data_loaders as raw


def extractors_exist(session_path):
    settings = raw.load_settings(session_path)
    task_name = settings['PYBPOD_PROTOCOL']
    task_name = task_name.split('_')[-1]
    extractor_type = task_name[:task_name.find('ChoiceWorld')]
    if any([extractor_type in x for x in globals()]):
        return extractor_type
    else:
        print(f"No extrators were found for {extractor_type}ChoiceWorld")
        return False


def is_extracted(session_path):
    sp = Path(session_path)
    if (sp / 'alf').exists():
        return True
    else:
        return False


def from_path(session_path, force=False):
    extractor_type = extractors_exist(session_path)
    if is_extracted(session_path) and not force:
        print(f"Session {session_path} already extracted.")
        return

    if extractor_type == 'training':
        training_trials.extract_all(session_path, save=True)
        training_wheel.extract_all(session_path, save=True)



if __name__ == '__main__':
    main_data_path = "/home/nico/GoogleDriveNeuro/IBL/PRIVATE/iblrig_data/"
    session_name = "6814/2018-12-05/001"
    session_path = main_data_path + session_name
    force = True

    from_path(session_path, force=force)
    print(".")

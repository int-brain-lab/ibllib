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

import alf.extractors.basic_trials as basic_trials
import alf.extractors.basic_wheel as basic_wheel
from ibllib.io import raw_data_loaders as raw


def get_task_name(session_path):
    settings = raw.load_settings(session_path)
    task_name = settings['PYBPOD_PROTOCOL']
    extractor_type = task_name[:task_name.find('ChoiceWorld')]
    if any([extractor_type in x for x in dir()]):
        pass
    else:
        print("No extrators were found for {}ChoiceWorld".format(extractor_type))
    os.getcwd()

if __name__ == '__main__':
    session_path = "/home/nico/Projects/IBL/IBL-github/iblrig/test_dataset/\
test_mouse/2018-10-02/1"
    settings = raw.load_settings(session_path)
    task_name = settings['PYBPOD_PROTOCOL']
    os.getcwd()

# -*- coding: utf-8 -*-
# @Author: nico
# @Date:   2018-02-07 19:03:05
# @Last Modified by:   nico
# @Last Modified time: 2018-02-15 13:38:12
import os
import json


DATA_FOLDER = '../pybpod_projects/IBL/data/'


def list_subjects(folder):
    """Gets any file or folder in data dir if its name has any digits in it"""
    out = [x for x in os.listdir(folder) if any([i.isdigit() for i in x])]
    return out


def list_sessions(subject):
    out = [x for x in os.listdir(DATA_FOLDER + subject)]
    return out


def session_files(subject, session):
    out = os.listdir(os.join(DATA_FOLDER, subject, session))
    return out


def load_data(session_path):
    data = []
    with open(session_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def make_cwFeedback(data):
    if 'iti' in data[0]['behavior_data']['States timestamps'].keys():
        first_state = 'stim_on'
        last_state = 'iti'
    else:
        first_state = 'trial_start'
        # last_state =

    trial_durs = []
    for trial in data:
        states = trial['behavior_data']['States timestamps']
        trial_durs.append(states[last_state][-1][-1] -
                          states[first_state][0][0])


if __name__ == '__main__':
    subject_list = list_subjects(DATA_FOLDER)
    session_list = {x: list_sessions(x) for x in subject_list}
    # file_lists = session_files(k, session_list[k]) for k in
    main_data_folder = '../pybpod_projects/IBL/data/'
    test_session_folder = '4579/2018-1-23_16-22-50/'
    data_file_name = '4579_2018-1-23_16-22-50.data.json'
    file_path = os.path.join(main_data_folder,
                             test_session_folder,
                             data_file_name)

    data = load_data(file_path)

    # trial = data[0]
    if 'iti' in data[0]['behavior_data']['States timestamps'].keys():
        first_state = 'stim_on'
        last_state = 'iti'

    trial_durs = []
    for trial in data:
        states = trial['behavior_data']['States timestamps']
        trial_durs.append(states[last_state][-1][-1] -
                          states[last_state][0][0])

    trial_start_times = []
    for trial in data:
        trial_start_times.append(
            trial['behavior_data']['Bpod start timestamp'])
    # data_file_name = 'vanillaChoiceWorld_pybpod.trial_data.json'
    # with open(test_session_folder + data_file_name, 'r') as f:
    #     trial_data = json.load(f)

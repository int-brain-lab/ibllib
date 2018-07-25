# -*- coding: utf-8 -*-
# @Author: nico
# @Date:   2018-02-13 18:02:33
# @Last Modified by:   nico
# @Last Modified time: 2018-05-16 17:32:32
import os
import json
import dateutil.parser
from ibl_utils import flatten


DATA_FOLDER = '/home/nico/Projects/IBL/IBL-github/analysis/'


def subject_list(folder=DATA_FOLDER):
    """Gets any file or folder in data dir if its name has any digits in it"""
    out = [x for x in os.listdir(folder) if any([i.isdigit() for i in x])]
    return out


def find_sessions(subject=None):
    if subject is not None:
        out = {subject: [x for x in os.listdir(DATA_FOLDER + subject)]}
        return out
    elif subject is None:
        all_sessions = {}
        for sub in subject_list():
            all_sessions[sub] = find_sessions(sub)
        return all_sessions


def all_sessions():
    out = flatten(find_sessions().values())
    return out


def _get_files(ftype=None, subject=None):

    paths = []
    all_sessions = find_sessions(subject=subject)
    for sub in all_sessions:
        for sess in all_sessions[sub]:
            folder_path = os.path.join(DATA_FOLDER, sub, sess)
            if ftype is None:
                file_names = [x for x in os.listdir(folder_path)
                              if 'settings' in x or 'data' in x]
                paths.append(os.path.join(folder_path, x) for x in file_names)
            else:
                file_name = [x for x in os.listdir(folder_path)
                             if ftype in x and 'json' in x][0]

                paths.append(os.path.join(folder_path, file_name))
    return paths


def settings_files(subject=None):
    return _get_files(ftype='settings', subject=subject)


def data_files(subject=None):
    return _get_files(ftype='data', subject=subject)


def load_raw_data(data_file):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_settings(settings_file):
    settings = []
    with open(settings_file, 'r') as s:
        for line in s:
            settings.append(json.loads(line))
    return settings


def session_datetime(sdf):
    if 'json' in sdf:
        datetime_str = [x for x in sdf.split('/') if x][-2]
    else:
        datetime_str = [x for x in sdf.split('/') if x][-1]
    if 'T' in datetime_str:
        out = dateutil.parser.parse(datetime_str)
    else:
        date_time = datetime_str.split('_')
        date = date_time[0]
        time = date_time[1].replace('-', ':')
        out = dateutil.parser.parse(date + ' ' + time)
    return out


if __name__ == '__main__':
    ftype = 'data'
    subject = None
    sdf = '/home/nico/Projects/IBL/IBL-github/analysis/4579/2018-1-23_16-22-50/4579_2018-1-23_16-22-50.data.json'

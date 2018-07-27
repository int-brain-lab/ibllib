# -*- coding: utf-8 -*-
# @Author: nico
# @Date:   2018-05-14 16:08:51
# @Last Modified by:   Niccolò Bonacchi
# @Last Modified time: 2018-07-03 13:08:07
import os
import re
import json
import dateutil.parser
from ibllib.misc import flatten


class Subject(object):
    """
    Subject object scrapes folder structure given
    ROOT_DATA_FOLDER for subjects i.e. ../lab/Subjects folder
    Properties:
        all_names: (str) all subject names in ROOT_DATA_FOLDER
        all_foders: (str) all subject folder paths
    Methods:
        folder: in: (str) mouse_name
                out: (str) absolute folder path for mouse_name
    Usage:

        subj = Subject(ROOT_DATA_FOLDER)
        >>> sub.all_names
         ['test_mouse2', 'test_mouse']
        >>> sub.all_folders
         ['/absolute/path/to/Subjects/test_mouse2',
          '/absolute/path/to/Subjects/test_mouse']
        >>> sub.folder()
         I need a mouse name...
        >>> sub.folder("test_mouse")
         /home/nico/Projects/IBL/IBL-github/IBL_root/pybpod_data/test_mouse
    """
    def __init__(self, ROOT_DATA_FOLDER):
        self.root_data_folder = ROOT_DATA_FOLDER

    @property
    def all_names(self):
        """Subject names are only defined by the folder in the main Subjects
        folder.
        TODO: check alyx for all subject names to match folder names"""
        return [x for x in os.listdir(self.root_data_folder)]

    @property
    def all_folders(self):
        return [os.path.join(self.root_data_folder, x) for x in self.all_names]

    def folder(self, mouse_name=None):
        if mouse_name is None:
            return 'I need a mouse name...'
        elif mouse_name not in self.all_names:
            return 'Unknown mouse...'
        else:
            mid = re.compile('^.+' + os.path.sep + mouse_name + os.path.sep +
                             '.+$')
            end = re.compile('^.+' + os.path.sep + mouse_name + '$')
            return [x for x in self.all_folders
                    if mid.match(x) or end.match(x)][0]


# TODO: decorate methods with mouse_name check
class Session(object):
    def __init__(self, ROOT_DATA_FOLDER):
        self.subj = Subject(ROOT_DATA_FOLDER)

    @property
    def all_dates(self):
        return flatten([os.listdir(x) for x in self.subj.all_folders])

    def dates(self, mouse_name=None):
        if mouse_name is None:
            return 'I need a mouse name...'
        elif mouse_name not in self.subj.all_names:
            return 'Unknown mouse...'
        else:
            return flatten([os.listdir(self.subj.folder(mouse_name))])

    @property
    def all_dates_paths(self):
        return [os.path.join(self.subj.folder(x), y)
                for x in self.subj.all_names for y in self.dates(x)]

    def dates_paths(self, mouse_name=None):
        if mouse_name is None:
            return 'I need a mouse name...'
        elif mouse_name not in self.subj.all_names:
            return 'Unknown mouse...'
        else:
            mid = re.compile('^.+' + os.path.sep + mouse_name + os.path.sep +
                             '.+$')
            end = re.compile('^.+' + os.path.sep + mouse_name + '$')
            return list(set([x for x in self.all_dates_paths
                             if mid.match(x) or end.match(x)]))

    @property
    def all_paths(self):
        return [os.path.join(x, y)
                for x in self.all_dates_paths for y in os.listdir(x)]

    @staticmethod
    def name_from_folder(folder_or_list):
        if isinstance(folder_or_list, str):
            folder_or_list = [folder_or_list]
        return ['{0}{2}{1}'.format(*y[-2:], os.path.sep) for y in
                [x.split('/') for x in folder_or_list]]

    @property
    def all_names(self):
        return ['{0}{3}{1}{3}{2}'.format(*y[-3:], os.path.sep) for y in
                [x.split(os.path.sep) for x in self.all_paths]]

    def paths(self, mouse_name=None):
        if mouse_name is None:
            return 'I need a mouse name...'
        elif mouse_name not in self.subj.all_names:
            return 'Unknown mouse...'
        else:
            mid = re.compile('^.+' + os.path.sep + mouse_name + os.path.sep +
                             '.+$')
            return list(set([x for x in self.all_paths
                             if mid.match(x)]))

    def names(self, mouse_name=None):
        if mouse_name is None:
            return 'I need a mouse name...'
        elif mouse_name not in self.subj.all_names:
            return 'Unknown mouse...'
        else:
            return Session.name_from_folder(self.paths(mouse_name))


# XXX: BROKEN SINCE INTRODUCTION OF raw_behavior_folder
class File(object):
    def __init__(self, ROOT_DATA_FOLDER):
        self.subj = Subject(ROOT_DATA_FOLDER)
        self.sess = Session(ROOT_DATA_FOLDER)

    @property
    def all_file_paths(self):
        """All files, all mice, all sessions"""
        out_paths = []
        for x in self.sess.all_paths:
            out_paths.extend(os.path.join(x, y) for y in os.listdir(x))
        return out_paths

    def mouse_file_paths(self, mouse_name=None):
        if mouse_name is None:
            return 'I need a mouse name...'
        elif mouse_name not in self.subj.all_names:
            return 'Unknown mouse...'
        else:
            out_paths = []
            for path in self.sess.paths(mouse_name):
                out_paths.extend([os.path.join(path, x)
                                  for x in os.listdir(path)])
            return out_paths

    def session_file_paths(self, session_name=None):
        path = os.path.join(ROOT_DATA_FOLDER, session_name)
        return [os.path.join(path, x) for x in os.listdir(path)]

    @staticmethod
    def paths_between(path_list, interval=[None, None]):
        path_list = sorted(path_list)
        if not all(interval):
            return path_list
        if interval[0] is None:
            interval[0] = path_list[0]
        if interval[1] is None:
            interval[1] = path_list[-1]
        idxs = [i for i, x in enumerate(path_list)
                if interval[0] in x or interval[1] in x]
        start = min(idxs)
        stop = max(idxs)

        return path_list[start:stop]

    @property
    def all_raw_files(self):
        return [x for x in file.all_file_paths if os.path.sep + 'raw' in x]

    def mouse_raw_files(self, mouse_name=None):
        return

    def session_raw_files(self, session_name=None):
        return

    @property
    def all_raw_behavior_file_paths(self):
        return

    def mouse_raw_behavior_file_paths(self, mouse_name=None):
        return

    def session_raw_behavior_file_paths(self, session_name=None):
        return

    def session_file_objects(self, session_name=None):
        """Returns a list of all ALF objects from a particular session """
        return

    def session_file_attributes(self, session_name=None, include_obj=False):
        """Returns a list of all ALF attributes from a particular session
        include_obj set to True will return list of unique obj.attrib
                    set to False will retur only list od attribs"""
        return

    def session_file_extensions(self, session_name=None):
        """Returns a list of all ALF estensions from a particular session """
        return

    @property
    def all_alf_file_paths(self):
        return

    def mouse_alf_file_paths(self, mouse_name=None):
        return

    def session_alf_file_paths(self, mouse_name=None):
        return

    @property
    def all_unextracted_sessions(self):
        return


class Data(object):
    """docstring for Data"""
    def __init__(self, ROOT_DATA_FOLDER):
        self.subj = Subject(ROOT_DATA_FOLDER)
        self.sess = Session(ROOT_DATA_FOLDER)
        self.file = File(ROOT_DATA_FOLDER)


if __name__ == '__main__':
    ROOT_DATA_FOLDER = '/home/nico/Projects/IBL/IBL-github/IBL_root/pybpod_data'
    sub = Subject(ROOT_DATA_FOLDER)
    print('\n')
    print('# SUBJECT #\n')
    print('>>> sub = Subject(folder)')
    print('>>> sub.all_names\n', sub.all_names)
    print('>>> sub.all_folders\n', sub.all_folders)
    print('>>> sub.folder()\n', sub.folder())
    print('>>> sub.folder("test_mouse")\n', sub.folder('test_mouse'))

    sess = Session(ROOT_DATA_FOLDER)
    print('\n')
    print('# SESSION #\n')
    print('>>> sess = Session(ROOT_DATA_FOLDER)')
    print('>>> sess.all_dates\n', sess.all_dates)
    print('>>> sess.dates()\n', sess.dates())
    print('>>> sess.dates("test_mouse")\n', sess.dates('test_mouse'))
    print('>>> sess.all_dates_paths\n', sess.all_dates_paths)
    print('>>> sess.dates_paths("test_mouse")\n',
          sess.dates_paths('test_mouse'))
    print('>>> sess.all_paths\n', sess.all_paths)
    print('>>> sess.all_names\n', sess.all_names)
    print('>>> sess.paths("test_mouse")\n', sess.paths('test_mouse'))
    print('>>> sess.names("test_mouse")\n', sess.names('test_mouse'))

    file = File(ROOT_DATA_FOLDER)
    print('\n')
    print('# FILE #\n')
    print('>>> file = File(ROOT_DATA_FOLDER)')
    print('>>> file.all_file_paths\n', file.all_file_paths)
    print('>>> file.mouse_file_paths("test_mouse")\n',
          file.mouse_file_paths('test_mouse'))
    print('>>> file.session_file_paths("test_mouse/2018-07-11/11")\n',
          file.session_file_paths('test_mouse/2018-07-11/11'))

# Experiment reference
# validator

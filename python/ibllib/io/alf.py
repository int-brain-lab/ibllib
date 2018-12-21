import os
import re
from ibllib.misc import flatten


class Scraper(object):
    """
    Scrapes folder structure for subjects.

    Requires a ROOT_DATA_FOLDER for subject data i.e.
    ../lab_name/Subjects folder

    alf.Scraper(root_data_folder)
    """

    def __init__(self, root_data_folder):
        self.root_data_folder = root_data_folder

    @property
    def subjects(self):
        """
        Get all folders from main subjects folder.

        Subject names are defined by being a folder in the main Subjects
        folder.
        """
        return [x for x in os.listdir(self.root_data_folder) if
                os.path.isdir(os.path.join(self.root_data_folder, x))]

    @property
    def subject_folders(self):
        """
        Get all subject folder paths found in root data folder.

        :return: All path strings of all subject folders found.
        :rtype: list
        """
        return [os.path.join(self.root_data_folder, x) for x in self.all_names]

    def path(self, root_path=None, subject=None, session=None, date_time=None,
             session_number=None):
        """
        Get path

        :param mouse_name: Name of subject.
        :type mouse_name: str
        :return: Absolute folder path for mouse_name
        :rtype: str
        """
        if subject is None:
            return 'I need a mouse name...'
        elif subject not in self.all_names:
            return 'Unknown mouse...'
        else:
            mid = re.compile('^.+' + os.path.sep + subject + os.path.sep +
                             '.+$')
            end = re.compile('^.+' + os.path.sep + subject + '$')
            return [x for x in self.all_folders
                    if mid.match(x) or end.match(x)][0]


class Subject(object):
    def __init__(self):
        pass


# TODO: decorate methods with mouse_name check
class Session(object):
    def __init__(self, root_data_folder):
        self.subj = Subject(root_data_folder)

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
        return ['{0}{3}{1}{3}{2}'.format(*y[-3:], os.path.sep) for y in
                [x.split(os.path.sep) for x in folder_or_list]]

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
    def __init__(self, root_data_folder):
        self.subj = Subject(root_data_folder)
        self.sess = Session(root_data_folder)

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
        path = os.path.join(self.subj.root_data_folder, session_name)
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
        """
        Returns a list of all ALF attributes from a particular session
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

    def __init__(self, root_data_folder):
        self.subj = Subject(root_data_folder)
        self.sess = Session(root_data_folder)
        self.file = File(root_data_folder)


if __name__ == '__main__':
    ROOT_DATA_FOLDER = '/home/nico/Projects/IBL/IBL-github/iblrig/Subjects'
    sub = Subject(ROOT_DATA_FOLDER)
    sub.all_names
    sub.all_folders
    sub.folder('test_mouse')

    sess = Session(ROOT_DATA_FOLDER)
    sess.all_dates
    sess.dates()
    sess.dates('test_mouse')
    sess.all_dates_paths
    sess.dates_paths('test_mouse')
    sess.all_paths
    sess.all_names
    sess.paths('test_mouse')
    sess.names('test_mouse')

    file = File(ROOT_DATA_FOLDER)
    file.all_file_paths
    file.mouse_file_paths('test_mouse')
    file.session_file_paths('test_mouse/2018-07-11/11')

# Experiment reference
# validator

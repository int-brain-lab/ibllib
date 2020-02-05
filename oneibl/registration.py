from pathlib import Path
import json
import datetime
import logging
from dateutil import parser as dateparser

import ibllib.time
from ibllib.misc import version, log2session
import ibllib.io.raw_data_loaders as raw
from ibllib.io import flags, hashfile

from oneibl.one import ONE

logger_ = logging.getLogger('ibllib.alf')
REGISTRATION_GLOB_PATTERNS = ['alf/**/*.*',
                              'logs/**/_ibl_log.*.log',
                              'raw_behavior_data/**/_iblrig_*.*',
                              'raw_passive_data/**/_iblrig_*.*',
                              'raw_behavior_data/**/_iblmic_*.*',
                              'raw_video_data/**/_iblrig_*.*',
                              'raw_video_data/**/_ibl_*.*',
                              'raw_ephys_data/**/_iblrig_*.*',
                              'raw_ephys_data/**/_spikeglx_*.*',
                              'raw_ephys_data/**/_iblqc_*.*',
                              ]


class RegistrationClient:
    """
    Object that keeps the ONE instance and provides method to create sessions and register data.
    """
    def __init__(self, one=None):
        self.one = one
        if not one:
            self.one = ONE()
        self.dtypes = self.one.alyx.rest('dataset-types', 'list')
        self.file_extensions = [df['file_extension'] for df in
                                self.one.alyx.rest('data-formats', 'list')]

    def create_sessions(self, root_data_folder, dry=False):
        """
        Create sessions looking recursively for flag files

        :param root_data_folder: folder to look for create_me.flag
        :param dry: bool. Dry run if True
        :return: None
        """
        flag_files = Path(root_data_folder).glob('**/create_me.flag')
        for flag_file in flag_files:
            if dry:
                print(flag_file)
                continue
            logger_.info('creating session for ' + str(flag_file.parent))
            # providing a false flag stops the registration after session creation
            self.register_session(flag_file.parent, file_list=False)
            flag_file.unlink()

    def register_sync(self, root_data_folder, dry=False):
        """
        Register sessions looking recursively for flag files

        :param root_data_folder: folder to look for register_me.flag
        :param dry: bool. Dry run if True
        :return:
        """
        flag_files = Path(root_data_folder).glob('**/register_me.flag')
        for flag_file in flag_files:
            if dry:
                print(flag_file)
                continue
            file_list = flags.read_flag_file(flag_file)
            logger_.info('registering ' + str(flag_file.parent))
            self.register_session(flag_file.parent, file_list=file_list)
            flags.write_flag_file(flag_file.parent.joinpath('flatiron.flag'), file_list=file_list)
            flag_file.unlink()
            if flag_file.parent.joinpath('create_me.flag').exists():
                flag_file.parent.joinpath('create_me.flag').unlink()
            logger_.info('registered' + '\n')

    @log2session('register')
    def register_session(self, ses_path, file_list=True):
        """
        Register session in Alyx

        :param ses_path: path to the session
        :param file_list: bool. Set to False will only create the session and skip registration
        :param repository_name: Optional, repository on which to register the data
        :return: Status string on error
        """
        if isinstance(ses_path, str):
            ses_path = Path(ses_path)
        # read meta data from the rig for the session from the task settings file
        settings_json_file = list(ses_path.glob(
            '**/raw_behavior_data/_iblrig_taskSettings.raw*.json'))
        if not settings_json_file:
            settings_json_file = list(ses_path.glob('**/_iblrig_taskSettings.raw*.json'))
            if not settings_json_file:
                logger_.error(['could not find _iblrig_taskSettings.raw.json. Abort.'])
                return
            logger_.warning([f'Settings found in a strange place: {settings_json_file}'])
        else:
            settings_json_file = settings_json_file[0]
        md = _read_settings_json_compatibility_enforced(settings_json_file)
        # query alyx endpoints for subject, error if not found
        try:
            subject = self.one.alyx.rest('subjects?nickname=' + md['SUBJECT_NAME'], 'list')[0]
        except IndexError as e:
            logger_.error(f"Subject: {md['SUBJECT_NAME']} doesn't exist in Alyx. ABORT.")
            raise e

        # look for a session from the same subject, same number on the same day
        session_id, session = self.one.search(subjects=subject['nickname'],
                                              date_range=md['SESSION_DATE'],
                                              number=md['SESSION_NUMBER'],
                                              details=True)
        try:
            user = self.one.alyx.rest('users', 'read', id=md["PYBPOD_CREATOR"][0])
        except Exception as e:
            logger_.error(f"User: {md['PYBPOD_CREATOR'][0]} doesn't exist in Alyx. ABORT")
            raise e

        username = user['username'] if user else subject['responsible_user']

        # load the trials data to get information about session duration and performance
        ses_data = raw.load_data(ses_path)
        start_time, end_time = _get_session_times(ses_path, md, ses_data)
        n_trials, n_correct_trials = _get_session_performance(md, ses_data)

        # this is the generic relative path: subject/yyyy-mm-dd/NNN
        gen_rel_path = Path(subject['nickname'], md['SESSION_DATE'],
                            '{0:03d}'.format(int(md['SESSION_NUMBER'])))

        # if nothing found create a new session in Alyx
        if not session:
            ses_ = {'subject': subject['nickname'],
                    'users': [username],
                    'location': md['PYBPOD_BOARD'],
                    'procedures': ['Behavior training/tasks'],
                    'lab': subject['lab'],
                    # 'project': project['name'],
                    'type': 'Experiment',
                    'task_protocol': md['PYBPOD_PROTOCOL'] + md['IBLRIG_VERSION_TAG'],
                    'number': md['SESSION_NUMBER'],
                    'start_time': ibllib.time.date2isostr(start_time),
                    'end_time': ibllib.time.date2isostr(end_time) if end_time else None,
                    'n_correct_trials': n_correct_trials,
                    'n_trials': n_trials,
                    'json': json.dumps(md, indent=1),
                    }
            session = self.one.alyx.rest('sessions', 'create', data=ses_)
            if md['SUBJECT_WEIGHT']:
                wei_ = {'subject': subject['nickname'],
                        'date_time': ibllib.time.date2isostr(start_time),
                        'weight': md['SUBJECT_WEIGHT'],
                        'user': username
                        }
                self.one.alyx.rest('weighings', 'create', data=wei_)
        else:  # TODO: if session exists and no json partial_upgrade it
            session = self.one.alyx.rest('sessions', 'read', id=session_id[0])

        logger_.info(session['url'] + ' ')
        # create associated water administration if not found
        if not session['wateradmin_session_related'] and ses_data:
            wa_ = {
                'subject': subject['nickname'],
                'date_time': ibllib.time.date2isostr(end_time),
                'water_administered': ses_data[-1]['water_delivered'] / 1000,
                'water_type': md.get('REWARD_TYPE') or 'Water',
                'user': username,
                'session': session['url'][-36:],
                'adlib': False}
            self.one.alyx.rest('water-administrations', 'create', data=wa_)
        # at this point the session has been created. If create only, exit
        if not file_list:
            return
        # register all files that match the Alyx patterns, warn user when files are encountered
        rename_files_compatibility(ses_path, md['IBLRIG_VERSION_TAG'])
        F = []  # empty list whose keys will be relative paths and content filenames
        md5s = []
        file_sizes = []
        for fn in _glob_session(ses_path):
            if fn.suffix in ['.flag', '.error', '.avi']:
                logger_.debug('Excluded: ', str(fn))
                continue
            if not self._match_filename_dtypes(fn):
                logger_.warning('No matching dataset type for: ' + str(fn))
                continue
            if fn.suffix not in self.file_extensions:
                logger_.warning('No matching dataformat (ie. file extension) for: ' + str(fn))
                continue
            if not _register_bool(fn.name, file_list):
                logger_.debug('Not in filelist: ' + str(fn))
                continue
            try:
                assert (str(gen_rel_path) in str(fn))
            except AssertionError as e:
                strerr = 'ALF folder mismatch: data is in wrong subject/date/number folder. \n'
                strerr += ' Expected ' + str(gen_rel_path) + ' actual was ' + str(fn)
                logger_.error(strerr)
                raise e
            # extract the relative path of the file
            rel_path = Path(str(fn)[str(fn).find(str(gen_rel_path)):])
            F.append(str(rel_path.relative_to(gen_rel_path)))
            file_sizes.append(fn.stat().st_size)
            md5s.append(hashfile.md5(fn) if fn.stat().st_size < 1024 ** 3 else None)
            logger_.info('Registering ' + str(fn))

        r_ = {'created_by': username,
              'path': str(gen_rel_path),
              'filenames': F,
              'hashes': md5s,
              'filesizes': file_sizes,
              'versions': [version.ibllib() for _ in F]
              }
        self.one.alyx.post('/register-file', data=r_)

    def _match_filename_dtypes(self, full_file):
        import re
        patterns = [dt['filename_pattern'] for dt in self.dtypes if dt['filename_pattern']]
        for pat in patterns:
            reg = pat.replace('.', r'\.').replace('_', r'\_').replace('*', r'.*')
            if re.match(reg, Path(full_file).name, re.IGNORECASE):
                return True
        return False


def _register_bool(fn, file_list):
    if isinstance(file_list, bool):
        return file_list
    if isinstance(file_list, str):
        file_list = [file_list]
    return any([str(fil) in fn for fil in file_list])


def _read_settings_json_compatibility_enforced(json_file):
    with open(json_file) as js:
        md = json.load(js)
    if 'IBLRIG_VERSION_TAG' not in md.keys():
        md['IBLRIG_VERSION_TAG'] = '3.2.3'
    if not md['IBLRIG_VERSION_TAG']:
        logger_.warning("You appear to be on an untagged version...")
        return md
    # 2018-12-05 Version 3.2.3 fixes (permanent fixes in IBL_RIG from 3.2.4 on)
    if version.le(md['IBLRIG_VERSION_TAG'], '3.2.3'):
        if 'LAST_TRIAL_DATA' in md.keys():
            md.pop('LAST_TRIAL_DATA')
        if 'weighings' in md['PYBPOD_SUBJECT_EXTRA'].keys():
            md['PYBPOD_SUBJECT_EXTRA'].pop('weighings')
        if 'water_administration' in md['PYBPOD_SUBJECT_EXTRA'].keys():
            md['PYBPOD_SUBJECT_EXTRA'].pop('water_administration')
        if 'IBLRIG_COMMIT_HASH' not in md.keys():
            md['IBLRIG_COMMIT_HASH'] = 'f9d8905647dbafe1f9bdf78f73b286197ae2647b'
        #  parse the date format to Django supported ISO
        dt = dateparser.parse(md['SESSION_DATETIME'])
        md['SESSION_DATETIME'] = ibllib.time.date2isostr(dt)
        # add the weight key if it doesn't already exists
        if 'SUBJECT_WEIGHT' not in md.keys():
            md['SUBJECT_WEIGHT'] = None
    return md


def rename_files_compatibility(ses_path, version_tag):
    if not version_tag:
        return
    if version.le(version_tag, '3.2.3'):
        task_code = ses_path.glob('**/_ibl_trials.iti_duration.npy')
        for fn in task_code:
            fn.replace(fn.parent.joinpath('_ibl_trials.itiDuration.npy'))
    task_code = ses_path.glob('**/_iblrig_taskCodeFiles.raw.zip')
    for fn in task_code:
        fn.replace(fn.parent.joinpath('_iblrig_codeFiles.raw.zip'))


def _get_session_times(fn, md, ses_data):
    """
    Get session start and end time from the Bpod data
    """
    start_time = ibllib.time.isostr2date(md['SESSION_DATETIME'])
    if not ses_data:
        return start_time, None
    c = 0
    for sd in reversed(ses_data):
        ses_duration_secs = (sd['behavior_data']['Trial end timestamp'] -
                             sd['behavior_data']['Bpod start timestamp'])
        if ses_duration_secs < (6 * 3600):
            break
        c += 1
    if c:
        logger_.warning((f'Trial end timestamps of last {c} trials above 6 hours '
                        f'(most likely corrupt): ') + str(fn))
    end_time = start_time + datetime.timedelta(seconds=ses_duration_secs)
    return start_time, end_time


def _get_session_performance(md, ses_data):
    """Get performance about the session from bpod data"""
    if not ses_data:
        return None, None
    n_trials = ses_data[-1]['trial_num']
    # checks that the number of actual trials and labeled number of trials check out
    assert (len(ses_data) == n_trials)
    # task specific logic
    if 'habituationChoiceWorld' in md['PYBPOD_PROTOCOL']:
        n_correct_trials = 0
    else:
        n_correct_trials = ses_data[-1]['ntrials_correct']
    return n_trials, n_correct_trials


def _glob_session(ses_path):
    """
    Glob for files to be registered on an IBL session
    :param ses_path: pathlib.Path of the session
    :return: a list of files to potentially be registered
    """
    fl = []
    for gp in REGISTRATION_GLOB_PATTERNS:
        fl.extend(list(ses_path.glob(gp)))
    return fl

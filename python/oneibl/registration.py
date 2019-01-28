from pathlib import Path
import json
import datetime
import logging
import traceback

import ibllib.time
from ibllib.misc import version
import ibllib.io.raw_data_loaders as raw
import ibllib.io.flags as flags

from oneibl.one import ONE

logger_ = logging.getLogger('ibllib.alf')


# this is a decorator to add a logfile to each extraction and registration on top of the logging
def log2sessionfile(func):
    def func_wrapper(self, sessionpath, *args, **kwargs):
        fh = logging.FileHandler(Path(sessionpath).joinpath('extract_register.log'))
        str_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        fh.setFormatter(logging.Formatter(str_format))
        logger_.addHandler(fh)
        f = func(self, sessionpath, *args, **kwargs)
        logger_.removeHandler(fh)
        return f
    return func_wrapper


class RegistrationClient:
    def __init__(self, one=None):
        self.one = one
        if not one:
            self.one = ONE()
        self.dtypes = self.one.alyx.rest('dataset-types', 'list')
        self.file_extensions = [df['file_extension'] for df in
                                self.one.alyx.rest('data-formats', 'list')]

    def register_sync(self, root_data_folder, dry=False):
        flag_files = Path(root_data_folder).glob('**/register_me.flag')
        for flag_file in flag_files:
            if dry:
                print(flag_file)
                continue
            file_list = flags.read_flag_file(flag_file)
            logger_.info('registering' + str(flag_file.parent))
            status_str = self.register_session(flag_file.parent, file_list=file_list)
            if status_str:
                error_message = str(flag_file.parent) + ' failed registration'
                error_message += '\n' + ' ' * 8 + status_str
                error_message += traceback.format_exc()
                logger_.error(error_message)
                err_file = flag_file.parent.joinpath('register_me.error')
                flag_file.rename(err_file)
                with open(err_file, 'w+') as f:
                    f.write(error_message)
                continue
            flag_file.rename(flag_file.parent.joinpath('flatiron.flag'))

    @log2sessionfile
    def register_session(self, ses_path, file_list=True):
        if isinstance(ses_path, str):
            ses_path = Path(ses_path)
        # read meta data from the rig for the session from the task settings file
        settings_json_file = [f for f in ses_path.glob('**/_iblrig_taskSettings.raw*.json')]
        if not settings_json_file:
            logger_.error(['could not find _iblrig_taskSettings.raw.json. Abort.'])
            return
        else:
            settings_json_file = settings_json_file[0]
        md = _read_settings_json_compatibility_enforced(settings_json_file)
        # query alyx endpoints for subject, projects and repository information
        try:
            subject = self.one.alyx.rest('subjects?nickname=' + md['SUBJECT_NAME'], 'list')[0]
        except IndexError:
            return 'Subject: ' + md['SUBJECT_NAME'] + " doesn't exist in Alyx"

        # find the first ibl matching project for the subject
        pname = [p for p in subject['projects'] if 'ibl' in p.lower()]
        project = self.one.alyx.rest('projects', 'read', pname[0])
        repository_name = [r for r in project['repositories'] if 'flatiron' not in r.lower()][0]
        repository = self.one.alyx.rest('data-repository', 'read', repository_name)
        # look for a session from the same subject, same number on the same day
        _, session = self.one.search(subjects=subject['nickname'],
                                     date_range=md['SESSION_DATE'],
                                     number=md['SESSION_NUMBER'],
                                     details=True)
        try:
            user = self.one.alyx.rest('users', 'read', md["PYBPOD_CREATOR"][0])
        except Exception:
            return 'Subject: ' + md["PYBPOD_CREATOR"][0] + " doesn't exist in Alyx"

        username = user['username'] if user else subject['responsible_user']

        # load the trials data to get information about session duration
        ses_data = raw.load_data(ses_path)
        ses_duration_secs = ses_data[-1]['behavior_data']['Trial end timestamp'] - \
            ses_data[-1]['behavior_data']['Bpod start timestamp']
        start_time = ibllib.time.isostr2date(md['SESSION_DATETIME'])
        end_time = start_time + datetime.timedelta(seconds=ses_duration_secs)

        # this is the generic relative path: subject/yyyy-mm-dd/NNN
        gen_rel_path = Path(subject['nickname'], md['SESSION_DATE'],
                            '{0:03d}'.format(int(md['SESSION_NUMBER'])))

        # checks that the number of actual trials and labeled number of trials check out
        assert(len(ses_data) == ses_data[-1]['trial_num'])

        # task specific logic
        if 'habituationChoiceWorld' in md['PYBPOD_PROTOCOL']:
            n_correct_trials = 0
        else:
            n_correct_trials = ses_data[-1]['ntrials_correct']

        # if nothing found create a new session in Alyx
        if not session:
            logger_.info('creating session' + str(gen_rel_path))
            ses_ = {'subject': subject['nickname'],
                    'users': [username],
                    'location': md['PYBPOD_BOARD'],
                    'procedures': ['Behavior training/tasks'],
                    'lab': subject['lab'],
                    'project': project['name'],
                    'type': 'Experiment',
                    'task_protocol': md['PYBPOD_PROTOCOL'] + md['IBLRIG_VERSION_TAG'],
                    'number': md['SESSION_NUMBER'],
                    'start_time': ibllib.time.date2isostr(start_time),
                    'end_time': ibllib.time.date2isostr(end_time),
                    'n_correct_trials': n_correct_trials,
                    'n_trials': ses_data[-1]['trial_num'],
                    'json': json.dumps(md, indent=1),
                    }
            session = self.one.alyx.rest('sessions', 'create', data=ses_)
            if md['SUBJECT_WEIGHT']:
                wei_ = {'subject': subject['nickname'],
                        'date_time': ibllib.time.date2isostr(start_time),
                        'weight': md['SUBJECT_WEIGHT'],
                        'user': username
                        }
                self.one.alyx.rest('weighings', 'create', wei_)
        else:
            session = session[0]

        # create associated water administration if not found
        if not session['wateradmin_session_related']:
            wa_ = {
                'subject': subject['nickname'],
                'date_time': ibllib.time.date2isostr(end_time),
                'water_administered': ses_data[-1]['water_delivered'] / 1000,
                'water_type': md['REWARD_TYPE'],
                'user': username,
                'session': session['url'][-36:],
                'adlib': False}
            self.one.alyx.rest('water-administrations', 'create', wa_)

        # register all files that match the Alyx patterns, warn user when files are encountered
        rename_files_compatibility(ses_path, md['IBLRIG_VERSION_TAG'])
        F = {}  # empty dict whose keys will be relative paths and content filenames
        for fn in ses_path.glob('**/*.*'):
            if fn.suffix in ['.flag', '.error', '.avi', '.log']:
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
            except AssertionError:
                strerr = 'ALF folder mismatch: data is in wrong subject/date/number folder. \n'
                strerr += ' Expected ' + str(gen_rel_path) + ' actual was ' + str(fn)
                return strerr
            # extract the relative path of the file
            rel_path = Path(str(fn)[str(fn).find(str(gen_rel_path)):]).parent
            if str(rel_path) not in F.keys():
                F[str(rel_path)] = [fn.name]
            else:
                F[str(rel_path)].append(fn.name)
            logger_.info('Registering ' + str(fn))

        for rpath in F:
            r_ = {'created_by': username,
                  'name': repository['name'],
                  'path': rpath,
                  'filenames': F[rpath],
                  }
            self.one.alyx.post('/register-file', data=r_)

    def _match_filename_dtypes(self, full_file):
        import re
        patterns = [dt['filename_pattern'] for dt in self.dtypes if dt['filename_pattern']]
        for pat in patterns:
            reg = pat.replace('.', r'\.').replace('_', r'\_').replace('*', r'.+')
            if re.match(reg, Path(full_file).name, re.IGNORECASE):
                return True
        return False


def _register_bool(fn, file_list):
    if isinstance(file_list, bool):
        return file_list
    if isinstance(file_list, str):
        file_list = [file_list]
    return any([fil in fn for fil in file_list])


def _read_settings_json_compatibility_enforced(json_file):
    with open(json_file) as js:
        md = json.load(js)
    if 'IBLRIG_VERSION_TAG' not in md.keys():
        md['IBLRIG_VERSION_TAG'] = '3.2.3'
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
        #  change the date format to proper ISO
        dt = datetime.datetime.strptime(md['SESSION_DATETIME'], '%Y-%m-%d %H:%M:%S.%f')
        md['SESSION_DATETIME'] = ibllib.time.date2isostr(dt)
        if 'SUBJECT_WEIGHT' not in md.keys():
            md['SUBJECT_WEIGHT'] = None
    return md


def rename_files_compatibility(ses_path, version_tag):
    if version.le(version_tag, '3.2.3'):
        task_code = ses_path.glob('**/_ibl_trials.iti_duration.npy')
        for fn in task_code:
            fn.rename(fn.parent.joinpath('_ibl_trials.itiDuration.npy'))
    task_code = ses_path.glob('**/_iblrig_taskCodeFiles.raw.zip')
    for fn in task_code:
        fn.rename(fn.parent.joinpath('_iblrig_codeFiles.raw.zip'))

from pathlib import Path
import json
import datetime
import logging

import ibllib.time
from ibllib.misc import version
from ibllib.io.raw_data_loaders import load_data
from oneibl.one import ONE

logger_ = logging.getLogger('ibllib.one')


class RegistrationClient:
    def __init__(self, one=None):
        self.one = one
        if not one:
            self.one = ONE()
        self.dtypes = self.one.alyx.rest('dataset-types', 'list')

    def register_sync(self, root_data_folder):
        flag_files = Path(root_data_folder).glob('**/register_me.flag')
        for flag_file in flag_files:
            logger_.info('registering' + str(flag_file.parent))
            status_str = self.register_session(flag_file.parent)
            if status_str:
                error_message = str(flag_file.parent) + ' failed registration'
                error_message += '\n' + ' ' * 8 + status_str
                logging.error(error_message)
                err_file = flag_file.parent.joinpath('register_me.error')
                flag_file.rename(err_file)
                with open(err_file, 'w+') as f:
                    f.write(error_message)
                continue
            flag_file.unlink()

    def register_session(self, ses_path):
        if isinstance(ses_path, str):
            ses_path = Path(ses_path)
        # read meta data from the rig for the session from the task settings file
        settings_json_file = [f for f in ses_path.glob('**/_iblrig_taskSettings.raw.json')][0]
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

        user = self.one.alyx.rest('users', 'read', md["PYBPOD_CREATOR"][0])
        username = user['username'] if user else subject['responsible_user']

        # load the trials data to get information about session duration
        ses_data = load_data(ses_path)
        ses_duration_secs = ses_data[-1]['behavior_data']['Trial end timestamp'] - \
            ses_data[-1]['behavior_data']['Bpod start timestamp']
        start_time = ibllib.time.isostr2date(md['SESSION_DATETIME'])
        end_time = start_time + datetime.timedelta(seconds=ses_duration_secs)

        # this is the generic relative path: subject/yyyy-mm-dd/NNN
        gen_rel_path = Path(subject['nickname'], md['SESSION_DATE'],
                            '{0:03d}'.format(int(md['SESSION_NUMBER'])))

        # checks that the number of actual trials and labeled number of trials check out
        assert(len(ses_data) == ses_data[-1]['trial_num'])

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
                    'n_correct_trials': ses_data[-1]['ntrials_correct'],
                    'n_trials': ses_data[-1]['trial_num'],
                    'json': json.dumps(md, indent=1),
                    }
            session = self.one.alyx.rest('sessions', 'create', data=ses_)
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
            if fn.suffix == '.flag' or fn.suffix == '.error':
                continue
            if not self._match_filename_dtypes(fn):
                logger_.warning('No matching dataset type for: ' + str(fn))
                continue
            # enforces that the session information is consistent with the current generic path
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
    return md


def rename_files_compatibility(ses_path, version_tag):
    if version.le(version_tag, '3.2.3'):
        task_code = ses_path.glob('**/_iblrig_TaskCodeFiles.raw.zip')
        for fn in task_code:
            fn.rename(fn.parent.joinpath('_iblrig_codeFiles.raw.zip'))
        task_code = ses_path.glob('**/_ibl_trials.iti_duration.npy')
        for fn in task_code:
            fn.rename(fn.parent.joinpath('_ibl_trials.itiDuration.npy'))

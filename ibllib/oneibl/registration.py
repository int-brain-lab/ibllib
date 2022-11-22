from pathlib import Path
import json
import datetime
import logging
import re
import shutil
from requests import HTTPError

from pkg_resources import parse_version
from dateutil import parser as dateparser
from iblutil.io import hashfile
from one.alf.files import get_session_path, folder_parts
import one.alf.exceptions as alferr
from one.api import ONE

import ibllib
import ibllib.io.extractors.base
import ibllib.time
import ibllib.io.raw_data_loaders as raw
from ibllib.io import flags

_logger = logging.getLogger(__name__)
EXCLUDED_EXTENSIONS = ['.flag', '.error', '.avi']
REGISTRATION_GLOB_PATTERNS = ['alf/**/*.*',
                              'raw_behavior_data/**/_iblrig_*.*',
                              'raw_passive_data/**/_iblrig_*.*',
                              'raw_behavior_data/**/_iblmic_*.*',
                              'raw_video_data/**/_iblrig_*.*',
                              'raw_video_data/**/_ibl_*.*',
                              'raw_ephys_data/**/_iblrig_*.*',
                              'raw_ephys_data/**/_spikeglx_*.*',
                              'raw_ephys_data/**/_iblqc_*.*',
                              'spikesorters/**/_kilosort_*.*'
                              'spikesorters/**/_kilosort_*.*',
                              'raw_widefield_data/**/_ibl_*.*',
                              'raw_photometry_data/**/_neurophotometrics_*.*'
                              ]


def _check_filename_for_registration(full_file, patterns):
    for pat in patterns:
        reg = pat.replace('.', r'\.').replace('_', r'\_').replace('*', r'.*')
        if Path(full_file).suffix in EXCLUDED_EXTENSIONS:
            return False
        elif re.match(reg, Path(full_file).name, re.IGNORECASE):
            return True
    return False


def register_dataset(file_list, one=None, created_by=None, repository=None, server_only=False,
                     versions=None, default=True, dry=False, max_md5_size=None, exists=False):
    """
    Registers a set of files belonging to a session only on the server
    :param file_list: (list of pathlib.Path or pathlib.Path)
    :param one: optional (one.api.One), current one object, will create an instance if not provided
    :param created_by: (string) name of user in Alyx (defaults to 'root')
    :param repository: optional: (string) name of the repository in Alyx
    :param server_only: optional: (bool) if True only creates on the Flatiron (defaults to False)
    :param versions: optional (list of strings): versions tags (defaults to ibllib version)
    :param default: optional (bool) whether to set as default dataset (defaults to True)
    :param dry: (bool) False by default
    :param max_md5_size: (int) maximum file in bytes to compute md5 sum (always compute if None)
    defaults to None
    :return:
    """
    # If the repository is specified then for the registration client we want server_only=True to make sure we don't make
    # any other repositories for the lab
    if repository and not server_only:
        server_only = True

    if created_by is None:
        created_by = one.alyx.user
    if file_list is None or file_list == '' or file_list == []:
        return
    elif not isinstance(file_list, list):
        file_list = [Path(file_list)]

    assert len(set([get_session_path(f) for f in file_list])) == 1
    assert all([Path(f).exists() for f in file_list])
    if versions is None:
        versions = ibllib.__version__
    if isinstance(versions, str):
        versions = [versions for _ in file_list]
    assert isinstance(versions, list) and len(versions) == len(file_list)

    # computing the md5 can be very long, so this is an option to skip if the file is bigger
    # than a certain threshold
    if max_md5_size:
        hashes = [hashfile.md5(p) if
                  p.stat().st_size < max_md5_size else None for p in file_list]
    else:
        hashes = [hashfile.md5(p) for p in file_list]

    session_path = get_session_path(file_list[0])

    # first register the file
    r = {'created_by': created_by,
         'path': session_path.relative_to((session_path.parents[2])).as_posix(),
         'filenames': [p.relative_to(session_path).as_posix() for p in file_list],
         'name': repository,
         'server_only': server_only,
         'hashes': hashes,
         'filesizes': [p.stat().st_size for p in file_list],
         'versions': versions,
         'default': default,
         'exists': exists,
         'check_protected': True}  # flag to see if any datasets are protected

    if not dry:
        if one is None:
            one = ONE(cache_rest=None)
        try:
            response = one.alyx.rest('register-file', 'create', data=r, no_cache=True)
            for p in file_list:
                _logger.info(f"ALYX REGISTERED DATA: {p}")
            return response
        except HTTPError as err:
            err_message = json.loads(err.response.text)
            if err_message['status_code'] == 403 and err_message['error'] == 'One or more datasets is protected':

                response = err_message['details']
                today_revision = datetime.datetime.today().strftime('%Y-%m-%d')
                new_file_list = []

                for fl, res in zip(file_list, response):
                    (name, prot_info), = res.items()

                    # Dataset has not yet been registered
                    if prot_info == []:
                        new_file_list.append(fl)
                        continue
                    else:
                        # Check to see if the file path already has a revision in it
                        file_revision = folder_parts(fl, as_dict=True)['revision']

                        if file_revision:
                            # Find existing protected revisions
                            existing_revisions = [key for pr in prot_info for key, val in pr.items() if val]
                            # If the revision explicitly defined by the user doesn't exist or is not protected, register as is
                            if file_revision not in existing_revisions:
                                revision_path = fl.parent
                            else:
                                i = 97  # equivalent to 'a'
                                new_revision = file_revision + chr(i).lower()
                                # Find the next subrevision that isn't protected
                                while new_revision in existing_revisions:
                                    i += 1
                                    new_revision = file_revision + chr(i).lower()
                                revision_path = fl.parent.parent.joinpath(f'#{new_revision}#')

                            if revision_path != fl.parent:
                                revision_path.mkdir(exist_ok=True)
                                shutil.move(fl, revision_path.joinpath(fl.name))
                            new_file_list.append(revision_path.joinpath(fl.name))
                            continue
                        else:
                            fl_path = fl.parent

                        assert name == fl_path.relative_to(session_path).joinpath(fl.name).as_posix()

                        # Find info about the latest revision, N.B on django side prot_info is sorted by latest revisions first
                        (latest_revision, protected), = prot_info[0].items()

                        # If the latest revision is the original and it is unprotected no need for revision
                        # e.g {'clusters.amp.npy': [{'': False}]}
                        if latest_revision == '' and not protected:
                            # Use original path
                            revision_path = fl_path

                        # If there already is a revision but it is unprotected, move into this revision folder
                        # e.g {'clusters.amp.npy': [{'2022-10-31': False}, {'2022-05-31': True}, {'': True}]}
                        elif not protected:
                            # Check that the latest_revision has the date naming convention we expect 'YYYY-MM-DD'
                            try:
                                _ = datetime.datetime.strptime(latest_revision[:10], '%Y-%m-%d')
                                revision_path = fl_path.joinpath(f'#{latest_revision}#')
                            # If it doesn't it probably has been made manually so we don't want to overwrite this and instead
                            # use today's date
                            except ValueError:
                                revision_path = fl_path.joinpath(f'#{today_revision}#')

                        # If protected and the latest protected revision is from today we need to make a subrevision
                        elif protected and today_revision in latest_revision:
                            if latest_revision == today_revision:
                                new_revision = today_revision + 'a'
                            else:
                                alpha = latest_revision[-1]
                                new_revision = today_revision + chr(ord(alpha) + 1).lower()

                            revision_path = fl_path.joinpath(f'#{new_revision}#')

                        # Otherwise cases move into revision from today
                        # e.g {'clusters.amp.npy': [{'': True}]}
                        # e.g {'clusters.amp.npy': [{'2022-10-31': True}, {'': True}]}
                        else:
                            revision_path = fl_path.joinpath(f'#{today_revision}#')

                        # Only move for the cases where a revision folder has been made
                        if revision_path != fl_path:
                            revision_path.mkdir(exist_ok=True)
                            shutil.move(fl, revision_path.joinpath(fl.name))
                        new_file_list.append(revision_path.joinpath(fl.name))

                file_list = new_file_list

                r = {'created_by': created_by,
                     'path': session_path.relative_to((session_path.parents[2])).as_posix(),
                     'filenames': [p.relative_to(session_path).as_posix() for p in file_list],
                     'name': repository,
                     'server_only': server_only,
                     'hashes': hashes,
                     'filesizes': [p.stat().st_size for p in file_list],
                     'versions': versions,
                     'default': default,
                     'exists': exists,
                     'check_protected': False}

                response = one.alyx.rest('register-file', 'create', data=r, no_cache=True)
                for p in file_list:
                    _logger.info(f"ALYX REGISTERED DATA: {p}")
                return response
            else:
                raise err


def register_session_raw_data(session_path, one=None, overwrite=False, dry=False, **kwargs):
    """
    Registers all files corresponding to raw data files to Alyx. It will select files that
    match Alyx registration patterns.
    :param session_path:
    :param one: one instance to work with
    :param overwrite: (False) if set to True, will patch the datasets. It will take very long.
    If set to False (default) will skip all already registered data.
    :param dry: do not register files, returns the list of files to be registered
    :return: list of file to register
    :return: Alyx response: dictionary of registered files
    """
    session_path = Path(session_path)
    one.alyx.clear_rest_cache()  # Ensure data are from database
    eid = one.path2eid(session_path, query_type='remote')  # needs to make sure we're up to date
    # query the database for existing datasets on the session and allowed dataset types
    dsets = one.alyx.rest('datasets', 'list', session=eid)
    already_registered = [
        session_path.joinpath(Path(ds['collection'] or '').joinpath(ds['name'])) for ds in dsets]
    dtypes = one.alyx.rest('dataset-types', 'list')
    registration_patterns = [dt['filename_pattern'] for dt in dtypes if dt['filename_pattern']]
    # glob all the files
    glob_patterns = [pat for pat in REGISTRATION_GLOB_PATTERNS if pat.startswith('raw')]
    files_2_register = []
    for gp in glob_patterns:
        f2r = list(session_path.glob(gp))
        files_2_register.extend(f2r)
    # filter 1/2 filter out datasets that do not match any dataset type
    files_2_register = list(filter(lambda f: _check_filename_for_registration(
        f, registration_patterns), files_2_register))
    # filter 2/2 unless overwrite is True, filter out the datasets that already exist
    if not overwrite:
        files_2_register = list(filter(lambda f: f not in already_registered, files_2_register))

    data_repo = get_local_data_repository(one)
    response = register_dataset(files_2_register, one=one, versions=None, dry=dry, repository=data_repo, **kwargs)
    return files_2_register, response


class RegistrationClient:
    """
    Object that keeps the ONE instance and provides method to create sessions and register data.
    """
    def __init__(self, one=None):
        self.one = one
        if not one:
            self.one = ONE(cache_rest=None)
        self.dtypes = self.one.alyx.rest('dataset-types', 'list')
        self.registration_patterns = [
            dt['filename_pattern'] for dt in self.dtypes if dt['filename_pattern']]
        self.file_extensions = [df['file_extension'] for df in
                                self.one.alyx.rest('data-formats', 'list', no_cache=True)]

    def create_sessions(self, root_data_folder, glob_pattern='**/create_me.flag', dry=False):
        """
        Create sessions looking recursively for flag files

        :param root_data_folder: folder to look for create_me.flag
        :param dry: bool. Dry run if True
        :param glob_pattern: bool. Dry run if True
        :return: None
        """
        flag_files = Path(root_data_folder).glob(glob_pattern)
        for flag_file in flag_files:
            if dry:
                print(flag_file)
                continue
            try:
                _logger.info('creating session for ' + str(flag_file.parent))
                # providing a false flag stops the registration after session creation
                self.create_session(flag_file.parent)
                flag_file.unlink()
            except BaseException as e:
                _logger.error(f'Error creating session for {flag_file.parent}\n{e}')
                _logger.warning(f'Skipping {flag_file.parent}')
                continue

        return [ff.parent for ff in flag_files]

    def create_session(self, session_path, **kwargs):
        """
        create_session(session_path)
        """
        return self.register_session(session_path, file_list=False, **kwargs)

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
            _logger.info('registering ' + str(flag_file.parent))
            self.register_session(flag_file.parent, file_list=file_list)
            flags.write_flag_file(flag_file.parent.joinpath('flatiron.flag'), file_list=file_list)
            flag_file.unlink()
            if flag_file.parent.joinpath('create_me.flag').exists():
                flag_file.parent.joinpath('create_me.flag').unlink()
            _logger.info('registered' + '\n')

    def register_session(self, ses_path, file_list=True, projects=None, procedures=None):
        """
        Register session in Alyx

        :param ses_path: path to the session
        :param file_list: bool. Set to False will only create the session and skip registration
        :param projects: list of strings corresponding to project names in the database. If set to
         None, defaults to the subject projects. Here is how to get a list of current projects
            >>> sorted([proj['name'] for proj in one.alyx.rest('projects', 'list')])
        :param procedures: (None) list of session procedure to label the session with. They should
        correspond to procedures in the database.
            >>> sorted([proc['name'] for proc in one.alyx.rest('procedures', 'list')])
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
                _logger.error(['could not find _iblrig_taskSettings.raw.json. Abort.'])
                raise ValueError(f'_iblrig_taskSettings.raw.json not found in {ses_path} Abort.')
            _logger.warning([f'Settings found in a strange place: {settings_json_file}'])
        else:
            settings_json_file = settings_json_file[0]
        md = _read_settings_json_compatibility_enforced(settings_json_file)
        # query alyx endpoints for subject, error if not found
        try:
            subject = self.one.alyx.rest(
                'subjects', 'list', nickname=md['SUBJECT_NAME'], no_cache=True)[0]
        except IndexError:
            _logger.error(f"Subject: {md['SUBJECT_NAME']} doesn't exist in Alyx. ABORT.")
            raise alferr.AlyxSubjectNotFound(md['SUBJECT_NAME'])

        # look for a session from the same subject, same number on the same day
        session_id, session = self.one.search(subject=subject['nickname'],
                                              date_range=md['SESSION_DATE'],
                                              number=md['SESSION_NUMBER'],
                                              details=True, query_type='remote')
        try:
            user = self.one.alyx.rest('users', 'read', id=md["PYBPOD_CREATOR"][0], no_cache=True)
        except Exception as e:
            _logger.error(f"User: {md['PYBPOD_CREATOR'][0]} doesn't exist in Alyx. ABORT")
            raise e

        username = user['username'] if user else subject['responsible_user']

        # load the trials data to get information about session duration and performance
        ses_data = raw.load_data(ses_path)
        start_time, end_time = _get_session_times(ses_path, md, ses_data)
        n_trials, n_correct_trials = _get_session_performance(md, ses_data)

        # this is the generic relative path: subject/yyyy-mm-dd/NNN
        gen_rel_path = Path(subject['nickname'], md['SESSION_DATE'],
                            '{0:03d}'.format(int(md['SESSION_NUMBER'])))

        task_protocol = md['PYBPOD_PROTOCOL'] + md['IBLRIG_VERSION_TAG']
        # unless specified label the session projects with subject projects
        projects = subject['projects'] if projects is None else projects
        # makes sure projects is a list
        projects = [projects] if isinstance(projects, str) else projects

        # unless specified label the session procedures with task protocol lookup
        procedures = procedures if procedures else _alyx_procedure_from_task(task_protocol)
        procedures = [procedures] if isinstance(procedures, str) else procedures
        json_fields_names = ['IS_MOCK', 'POOP_COUNT', 'IBLRIG_VERSION']
        json_field = {f: md[f] for f in json_fields_names if f in md}
        if not session:
            ses_ = {'subject': subject['nickname'],
                    'users': [username],
                    'location': md['PYBPOD_BOARD'],
                    'procedures': procedures,
                    'lab': subject['lab'],
                    'projects': projects,
                    'type': 'Experiment',
                    'task_protocol': task_protocol,
                    'number': md['SESSION_NUMBER'],
                    'start_time': ibllib.time.date2isostr(start_time),
                    'end_time': ibllib.time.date2isostr(end_time) if end_time else None,
                    'n_correct_trials': n_correct_trials,
                    'n_trials': n_trials,
                    'json': json_field,
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
            session = self.one.alyx.rest('sessions', 'read', id=session_id[0], no_cache=True)

        _logger.info(session['url'] + ' ')
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
            return session
        # register all files that match the Alyx patterns, warn user when files are encountered
        rename_files_compatibility(ses_path, md['IBLRIG_VERSION_TAG'])
        F = []  # empty list whose keys will be relative paths and content filenames
        md5s = []
        file_sizes = []
        for fn in _glob_session(ses_path):
            if fn.suffix in EXCLUDED_EXTENSIONS:
                _logger.debug('Excluded: ', str(fn))
                continue
            if not _check_filename_for_registration(fn, self.registration_patterns):
                _logger.warning('No matching dataset type for: ' + str(fn))
                continue
            if fn.suffix not in self.file_extensions:
                _logger.warning('No matching dataformat (ie. file extension) for: ' + str(fn))
                continue
            if not _register_bool(fn.name, file_list):
                _logger.debug('Not in filelist: ' + str(fn))
                continue
            try:
                assert (str(gen_rel_path) in str(fn))
            except AssertionError as e:
                strerr = 'ALF folder mismatch: data is in wrong subject/date/number folder. \n'
                strerr += ' Expected ' + str(gen_rel_path) + ' actual was ' + str(fn)
                _logger.error(strerr)
                raise e
            # extract the relative path of the file
            rel_path = Path(str(fn)[str(fn).find(str(gen_rel_path)):])
            F.append(str(rel_path.relative_to(gen_rel_path).as_posix()))
            file_sizes.append(fn.stat().st_size)
            md5s.append(hashfile.md5(fn) if fn.stat().st_size < 1024 ** 3 else None)
            _logger.info('Registering ' + str(fn))

        r_ = {'created_by': username,
              'path': str(gen_rel_path.as_posix()),
              'filenames': F,
              'hashes': md5s,
              'filesizes': file_sizes,
              'versions': [ibllib.__version__ for _ in F]
              }
        self.one.alyx.post('/register-file', data=r_)
        return session


def _alyx_procedure_from_task(task_protocol):
    task_type = ibllib.io.extractors.base.get_task_extractor_type(task_protocol)
    procedure = _alyx_procedure_from_task_type(task_type)
    return procedure or []


def _alyx_procedure_from_task_type(task_type):
    lookup = {'biased': 'Behavior training/tasks',
              'biased_opto': 'Behavior training/tasks',
              'habituation': 'Behavior training/tasks',
              'training': 'Behavior training/tasks',
              'ephys': 'Ephys recording with acute probe(s)',
              'ephys_biased_opto': 'Ephys recording with acute probe(s)',
              'ephys_passive_opto': 'Ephys recording with acute probe(s)',
              'ephys_replay': 'Ephys recording with acute probe(s)',
              'ephys_training': 'Ephys recording with acute probe(s)',
              'mock_ephys': 'Ephys recording with acute probe(s)',
              'sync_ephys': 'Ephys recording with acute probe(s)'}
    try:
        # look if there are tasks in the personal projects repo with proceedures
        import projects.base
        custom_tasks = Path(projects.base.__file__).parent.joinpath('task_type_procedures.json')
        with open(custom_tasks) as fp:
            lookup.update(json.load(fp))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    if task_type in lookup:
        return lookup[task_type]


def _register_bool(fn, file_list):
    if isinstance(file_list, bool):
        return file_list
    if isinstance(file_list, str):
        file_list = [file_list]
    return any([str(fil) in fn for fil in file_list])


def _read_settings_json_compatibility_enforced(json_file):
    with open(json_file) as js:
        md = json.load(js)
    if 'IS_MOCK' not in md.keys():
        md['IS_MOCK'] = False
    if 'IBLRIG_VERSION_TAG' not in md.keys():
        md['IBLRIG_VERSION_TAG'] = '3.2.3'
    if not md['IBLRIG_VERSION_TAG']:
        _logger.warning("You appear to be on an untagged version...")
        return md
    # 2018-12-05 Version 3.2.3 fixes (permanent fixes in IBL_RIG from 3.2.4 on)
    if parse_version(md['IBLRIG_VERSION_TAG']) <= parse_version('3.2.3'):
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
    if parse_version(version_tag) <= parse_version('3.2.3'):
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
        _logger.warning((f'Trial end timestamps of last {c} trials above 6 hours '
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


def get_local_data_repository(one):
    """
    Get local data repo name from globus client
    :param one:
    :return:
    """
    if one is None:
        return

    if not Path.home().joinpath(".globusonline/lta/client-id.txt").exists():
        return

    with open(Path.home().joinpath(".globusonline/lta/client-id.txt"), 'r') as fid:
        globus_id = fid.read()

    data_repo = one.alyx.rest('data-repository', 'list', globus_endpoint_id=globus_id)
    if len(data_repo):
        return [da['name'] for da in data_repo][0]

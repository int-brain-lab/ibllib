from pathlib import Path
import json
import datetime
import logging
import itertools

from pkg_resources import parse_version
from one.alf.files import get_session_path, folder_parts, get_alf_path
from one.registration import RegistrationClient, get_dataset_type
from one.remote.globus import get_local_endpoint_id
import one.alf.exceptions as alferr
from one.util import datasets2records, ensure_list

import ibllib
import ibllib.io.extractors.base
from ibllib.time import isostr2date
import ibllib.io.raw_data_loaders as raw
from ibllib.io import session_params

_logger = logging.getLogger(__name__)
EXCLUDED_EXTENSIONS = ['.flag', '.error', '.avi']
REGISTRATION_GLOB_PATTERNS = ['alf/**/*.*',
                              'raw_behavior_data/**/_iblrig_*.*',
                              'raw_task_data_*/**/_iblrig_*.*',
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
                              'raw_photometry_data/**/_neurophotometrics_*.*',
                              ]


def register_dataset(file_list, one=None, exists=False, versions=None, **kwargs):
    """
    Registers a set of files belonging to a session only on the server.

    Parameters
    ----------
    file_list : list, str, pathlib.Path
        A filepath (or list thereof) of ALF datasets to register to Alyx.
    one : one.api.OneAlyx
        An instance of ONE.
    exists : bool
        Whether files exist in the repository. May be set to False when registering files
        before copying to the repository.
    versions : str, list of str
        Optional version tags, defaults to the current ibllib version.
    kwargs
        Optional keyword arguments for one.registration.RegistrationClient.register_files.

    Returns
    -------
    list of dicts, dict
        A list of newly created Alyx dataset records or the registration data if dry.

    Notes
    -----
    - If a repository is passed, server_only will be set to True.

    See Also
    --------
    one.registration.RegistrationClient.register_files
    """
    if not file_list:
        return
    elif isinstance(file_list, (str, Path)):
        file_list = [file_list]

    assert len(set(get_session_path(f) for f in file_list)) == 1
    assert all(Path(f).exists() for f in file_list)

    client = IBLRegistrationClient(one)
    # If the repository is specified then for the registration client we want server_only=True to
    # make sure we don't make any other repositories for the lab
    if kwargs.get('repository') and not kwargs.get('server_only', False):
        kwargs['server_only'] = True

    return client.register_files(file_list, versions=versions or ibllib.__version__, exists=exists, **kwargs)


def register_session_raw_data(session_path, one=None, overwrite=False, **kwargs):
    """
    Registers all files corresponding to raw data files to Alyx. It will select files that
    match Alyx registration patterns.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The local session path.
    one : one.api.OneAlyx
        An instance of ONE.
    overwrite : bool
        If set to True, will patch the datasets. It will take very long. If set to False (default)
        will skip all already registered data.
    **kwargs
        Optional keyword arguments for one.registration.RegistrationClient.register_files.

    Returns
    -------
    list of pathlib.Path
        A list of raw dataset paths.
    list of dicts, dict
        A list of newly created Alyx dataset records or the registration data if dry.
    """
    client = IBLRegistrationClient(one)
    session_path = Path(session_path)
    eid = one.path2eid(session_path, query_type='remote')  # needs to make sure we're up to date
    if not eid:
        raise alferr.ALFError(f'Session does not exist on Alyx: {get_alf_path(session_path)}')
    # find all files that are in a raw data collection
    file_list = [f for f in client.find_files(session_path)
                 if f.relative_to(session_path).as_posix().startswith('raw')]
    # unless overwrite is True, filter out the datasets that already exist
    if not overwrite:
        # query the database for existing datasets on the session and allowed dataset types
        dsets = datasets2records(one.alyx.rest('datasets', 'list', session=eid))
        already_registered = list(map(session_path.joinpath, dsets['rel_path']))
        file_list = list(filter(lambda f: f not in already_registered, file_list))

    kwargs['repository'] = get_local_data_repository(one.alyx)
    kwargs['server_only'] = True

    response = client.register_files(file_list, versions=ibllib.__version__, exists=False, **kwargs)
    return file_list, response


class IBLRegistrationClient(RegistrationClient):
    """
    Object that keeps the ONE instance and provides method to create sessions and register data.
    """

    def register_session(self, ses_path, file_list=True, projects=None, procedures=None):
        """
        Register an IBL Bpod session in Alyx.

        Parameters
        ----------
        ses_path : str, pathlib.Path
            The local session path.
        file_list : bool, list
            An optional list of file paths to register.  If True, all valid files within the
            session folder are registered.  If False, no files are registered.
        projects: str, list
            The project(s) to which the experiment belongs (optional).
        procedures : str, list
            An optional list of procedures, e.g. 'Behavior training/tasks'.

        Returns
        -------
        dict
            An Alyx session record.

        Notes
        -----
        For a list of available projects:
        >>> sorted(proj['name'] for proj in one.alyx.rest('projects', 'list'))
        For a list of available procedures:
        >>> sorted(proc['name'] for proc in one.alyx.rest('procedures', 'list'))
        """
        if isinstance(ses_path, str):
            ses_path = Path(ses_path)

        # Read in the experiment description file if it exists and get projects and procedures from here
        experiment_description_file = session_params.read_params(ses_path)
        if experiment_description_file is None:
            collections = ['raw_behavior_data']
        else:
            projects = experiment_description_file.get('projects', projects)
            procedures = experiment_description_file.get('procedures', procedures)
            collections = ensure_list(session_params.get_task_collection(experiment_description_file))

        # read meta data from the rig for the session from the task settings file
        task_data = (raw.load_bpod(ses_path, collection) for collection in sorted(collections))
        # Filter collections where settings file was not found
        if not (task_data := list(zip(*filter(lambda x: x[0] is not None, task_data)))):
            raise ValueError(f'_iblrig_taskSettings.raw.json not found in {ses_path} Abort.')
        settings, task_data = task_data
        if len(settings) != len(collections):
            raise ValueError(f'_iblrig_taskSettings.raw.json not found in {ses_path} Abort.')

        # Do some validation
        _, subject, date, number, *_ = folder_parts(ses_path)
        assert len({x['SUBJECT_NAME'] for x in settings}) == 1 and settings[0]['SUBJECT_NAME'] == subject
        assert len({x['SESSION_DATE'] for x in settings}) == 1 and settings[0]['SESSION_DATE'] == date
        assert len({x['SESSION_NUMBER'] for x in settings}) == 1 and settings[0]['SESSION_NUMBER'] == number
        assert len({x['IS_MOCK'] for x in settings}) == 1
        assert len({md['PYBPOD_BOARD'] for md in settings}) == 1
        assert len({md.get('IBLRIG_VERSION') for md in settings}) == 1
        assert len({md['IBLRIG_VERSION_TAG'] for md in settings}) == 1

        # query Alyx endpoints for subject, error if not found
        subject = self.assert_exists(subject, 'subjects')

        # look for a session from the same subject, same number on the same day
        session_id, session = self.one.search(subject=subject['nickname'],
                                              date_range=date,
                                              number=number,
                                              details=True, query_type='remote')
        users = []
        for user in filter(None, map(lambda x: x.get('PYBPOD_CREATOR'), settings)):
            user = self.assert_exists(user[0], 'users')  # user is list of [username, uuid]
            users.append(user['username'])

        # extract information about session duration and performance
        start_time, end_time = _get_session_times(str(ses_path), settings, task_data)
        n_trials, n_correct_trials = _get_session_performance(settings, task_data)

        # TODO Add task_protocols to Alyx sessions endpoint
        task_protocols = [md['PYBPOD_PROTOCOL'] + md['IBLRIG_VERSION_TAG'] for md in settings]
        # unless specified label the session projects with subject projects
        projects = subject['projects'] if projects is None else projects
        # makes sure projects is a list
        projects = [projects] if isinstance(projects, str) else projects

        # unless specified label the session procedures with task protocol lookup
        procedures = procedures or list(set(filter(None, map(self._alyx_procedure_from_task, task_protocols))))
        procedures = [procedures] if isinstance(procedures, str) else procedures
        json_fields_names = ['IS_MOCK', 'IBLRIG_VERSION']
        json_field = {k: settings[0].get(k) for k in json_fields_names}
        # The poo count field is only updated if the field is defined in at least one of the settings
        poo_counts = [md.get('POOP_COUNT') for md in settings if md.get('POOP_COUNT') is not None]
        if poo_counts:
            json_field['POOP_COUNT'] = int(sum(poo_counts))

        if not session:  # Create session and weighings
            ses_ = {'subject': subject['nickname'],
                    'users': users or [subject['responsible_user']],
                    'location': settings[0]['PYBPOD_BOARD'],
                    'procedures': procedures,
                    'lab': subject['lab'],
                    'projects': projects,
                    'type': 'Experiment',
                    'task_protocol': '/'.join(task_protocols),
                    'number': number,
                    'start_time': self.ensure_ISO8601(start_time),
                    'end_time': self.ensure_ISO8601(end_time) if end_time else None,
                    'n_correct_trials': n_correct_trials,
                    'n_trials': n_trials,
                    'json': json_field
                    }
            session = self.one.alyx.rest('sessions', 'create', data=ses_)
            # Submit weights
            for md in filter(lambda md: md.get('SUBJECT_WEIGHT') is not None, settings):
                user = md.get('PYBPOD_CREATOR')
                user = user[0] if user[0] in users else self.one.alyx.user
                self.register_weight(subject['nickname'], md['SUBJECT_WEIGHT'],
                                     date_time=md['SESSION_DATETIME'], user=user)
        else:  # if session exists update the JSON field
            session = self.one.alyx.rest('sessions', 'read', id=session_id[0], no_cache=True)
            self.one.alyx.json_field_update('sessions', session['id'], data=json_field)

        _logger.info(session['url'] + ' ')
        # create associated water administration if not found
        if not session['wateradmin_session_related'] and any(task_data):
            for md, d in zip(settings, task_data):
                _, _end_time = _get_session_times(ses_path, md, d)
                user = md.get('PYBPOD_CREATOR')
                user = user[0] if user[0] in users else self.one.alyx.user
                if (volume := d[-1]['water_delivered'] / 1000) > 0:
                    self.register_water_administration(
                        subject['nickname'], volume, date_time=_end_time or end_time, user=user,
                        session=session['id'], water_type=md.get('REWARD_TYPE') or 'Water')
        # at this point the session has been created. If create only, exit
        if not file_list:
            return session, None

        # register all files that match the Alyx patterns and file_list
        rename_files_compatibility(ses_path, settings[0]['IBLRIG_VERSION_TAG'])
        F = filter(lambda x: self._register_bool(x.name, file_list), self.find_files(ses_path))
        recs = self.register_files(F, created_by=users[0] if users else None, versions=ibllib.__version__)
        return session, recs

    @staticmethod
    def _register_bool(fn, file_list):
        if isinstance(file_list, bool):
            return file_list
        if isinstance(file_list, str):
            file_list = [file_list]
        return any(str(fil) in fn for fil in file_list)

    @staticmethod
    def _alyx_procedure_from_task(task_protocol):
        task_type = ibllib.io.extractors.base.get_task_extractor_type(task_protocol)
        procedure = _alyx_procedure_from_task_type(task_type)
        return procedure or []

    def find_files(self, session_path):
        """Similar to base class method but further filters by name and extension.

        In addition to finding files that match Excludes files
        whose extension is in EXCLUDED_EXTENSIONS, or that don't match the patterns in
        REGISTRATION_GLOB_PATTERNS.

        Parameters
        ----------
        session_path : str, pathlib.Path
            The session path to search.

        Yields
        -------
        pathlib.Path
            File paths that match the dataset type patterns in Alyx and registration glob patterns.
        """
        files = itertools.chain.from_iterable(session_path.glob(x) for x in REGISTRATION_GLOB_PATTERNS)
        for file in filter(lambda x: x.suffix not in EXCLUDED_EXTENSIONS, files):
            try:
                get_dataset_type(file, self.dtypes)
                yield file
            except ValueError as ex:
                _logger.error(ex)


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
        # look if there are tasks in the personal projects repo with procedures
        import projects.base
        custom_tasks = Path(projects.base.__file__).parent.joinpath('task_type_procedures.json')
        with open(custom_tasks) as fp:
            lookup.update(json.load(fp))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    if task_type in lookup:
        return lookup[task_type]


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
    Get session start and end time from the Bpod data.

    Parameters
    ----------
    fn : str, pathlib.Path
        Session/task identifier. Only used in warning logs.
    md : dict, list of dict
        A session parameters dictionary or list thereof.
    ses_data : dict, list of dict
        A session data dictionary or list thereof.

    Returns
    -------
    datetime.datetime
        The datetime of the start of the session.
    datetime.datetime
        The datetime of the end of the session, or None is ses_data is None.
    """
    if isinstance(md, dict):
        start_time = _start_time = isostr2date(md['SESSION_DATETIME'])
    else:
        start_time = isostr2date(md[0]['SESSION_DATETIME'])
        _start_time = isostr2date(md[-1]['SESSION_DATETIME'])
        assert isinstance(ses_data, (list, tuple)) and len(ses_data) == len(md)
        assert len(md) == 1 or start_time < _start_time
        ses_data = ses_data[-1]
    if not ses_data:
        return start_time, None
    c = ses_duration_secs = 0
    for sd in reversed(ses_data):
        ses_duration_secs = (sd['behavior_data']['Trial end timestamp'] -
                             sd['behavior_data']['Bpod start timestamp'])
        if ses_duration_secs < (6 * 3600):
            break
        c += 1
    if c:
        _logger.warning(('Trial end timestamps of last %i trials above 6 hours '
                         '(most likely corrupt): %s'), c, str(fn))
    end_time = _start_time + datetime.timedelta(seconds=ses_duration_secs)
    return start_time, end_time


def _get_session_performance(md, ses_data):
    """
    Get performance about the session from Bpod data.
    Note: This does not support custom protocols.

    Parameters
    ----------
    md : dict, list of dict
        A session parameters dictionary or list thereof.
    ses_data : dict, list of dict
        A session data dictionary or list thereof.

    Returns
    -------
    int
        The total number of trials across protocols.
    int
        The total number of correct trials across protocols.
    """
    if not any(filter(None, ses_data or None)):
        return None, None

    if isinstance(md, dict):
        ses_data = [ses_data]
        md = [md]
    else:
        assert isinstance(ses_data, (list, tuple)) and len(ses_data) == len(md)

    n_trials = [x[-1]['trial_num'] for x in ses_data]
    # checks that the number of actual trials and labeled number of trials check out
    assert all(len(x) == n for x, n in zip(ses_data, n_trials))
    # task specific logic
    n_correct_trials = []
    for data, proc in zip(ses_data, map(lambda x: x.get('PYBPOD_PROTOCOL', ''), md)):
        if 'habituationChoiceWorld' in proc:
            n_correct_trials.append(0)
        else:
            n_correct_trials.append(data[-1]['ntrials_correct'])

    return sum(n_trials), sum(n_correct_trials)


def get_local_data_repository(ac):
    """
    Get local data repo name from Globus client.

    Parameters
    ----------
    ac : one.webclient.AlyxClient
        An AlyxClient instance for querying data repositories.

    Returns
    -------
    str
        The (first) data repository associated with the local Globus endpoint ID.
    """
    try:
        assert ac
        globus_id = get_local_endpoint_id()
    except AssertionError:
        return

    data_repo = ac.rest('data-repository', 'list', globus_endpoint_id=globus_id)
    return next((da['name'] for da in data_repo), None)


def get_lab(ac):
    """
    Get list of associated labs from Globus client ID.

    Parameters
    ----------
    ac : one.webclient.AlyxClient
        An AlyxClient instance for querying data repositories.

    Returns
    -------
    list
        The lab names associated with the local Globus endpoint ID.
    """
    globus_id = get_local_endpoint_id()
    lab = ac.rest('labs', 'list', django=f'repositories__globus_endpoint_id,{globus_id}')
    return [la['name'] for la in lab]

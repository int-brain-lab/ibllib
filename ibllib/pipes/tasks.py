from pathlib import Path
import abc
import logging
import io
import importlib
import time
from _collections import OrderedDict
import traceback
import json

from graphviz import Digraph

import ibllib
from ibllib.oneibl import data_handlers
from ibllib.oneibl.data_handlers import get_local_data_repository
from ibllib.oneibl.registration import get_lab
from iblutil.util import Bunch
import one.params
from one.api import ONE

_logger = logging.getLogger(__name__)


class Task(abc.ABC):
    log = ""  # place holder to keep the log of the task for registration
    cpu = 1   # CPU resource
    gpu = 0   # GPU resources: as of now, either 0 or 1
    io_charge = 5  # integer percentage
    priority = 30  # integer percentage, 100 means highest priority
    ram = 4  # RAM needed to run (GB)
    one = None  # one instance (optional)
    level = 0  # level in the pipeline hierarchy: level 0 means there is no parent task
    outputs = None  # place holder for a list of Path containing output files
    time_elapsed_secs = None
    time_out_secs = 3600 * 2  # time-out after which a task is considered dead
    version = ibllib.__version__
    signature = {'input_files': [], 'output_files': []}  # list of tuples (filename, collection, required_flag)
    force = False  # whether or not to re-download missing input files on local server if not present
    job_size = 'small'  # either 'small' or 'large', defines whether task should be run as part of the large or small job services

    def __init__(self, session_path, parents=None, taskid=None, one=None,
                 machine=None, clobber=True, location='server', **kwargs):
        """
        Base task class
        :param session_path: session path
        :param parents: parents
        :param taskid: alyx task id
        :param one: one instance
        :param machine:
        :param clobber: whether or not to overwrite log on rerun
        :param location: location where task is run. Options are 'server' (lab local servers'), 'remote' (remote compute node,
        data required for task downloaded via one), 'AWS' (remote compute node, data required for task downloaded via AWS),
        or 'SDSC' (SDSC flatiron compute node) # TODO 'Globus' (remote compute node, data required for task downloaded via Globus)
        :param args: running arguments
        """
        self.taskid = taskid
        self.one = one
        self.session_path = session_path
        self.register_kwargs = {}
        if parents:
            self.parents = parents
            self.level = max([p.level for p in self.parents]) + 1
        else:
            self.parents = []
        self.machine = machine
        self.clobber = clobber
        self.location = location
        self.plot_tasks = []  # Plotting task/ tasks to create plot outputs during the task
        self.kwargs = kwargs

    @property
    def name(self):
        return self.__class__.__name__

    def run(self, **kwargs):
        """
        --- do not overload, see _run() below---
        wraps the _run() method with
        -   error management
        -   logging to variable
        -   writing a lock file if the GPU is used
        -   labels the status property of the object. The status value is labeled as:
             0: Complete
            -1: Errored
            -2: Didn't run as a lock was encountered
            -3: Incomplete
        """
        # if task id of one properties are not available, local run only without alyx
        use_alyx = self.one is not None and self.taskid is not None
        if use_alyx:
            # check that alyx user is logged in
            if not self.one.alyx.is_logged_in:
                self.one.alyx.authenticate()
            tdict = self.one.alyx.rest('tasks', 'partial_update', id=self.taskid,
                                       data={'status': 'Started'})
            self.log = ('' if not tdict['log'] else tdict['log'] +
                        '\n\n=============================RERUN=============================\n')

        # Setup the console handler with a StringIO object
        logger_level = _logger.level
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        str_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        ch.setFormatter(logging.Formatter(str_format))
        _logger.parent.addHandler(ch)
        _logger.parent.setLevel(logging.INFO)
        _logger.info(f"Starting job {self.__class__}")
        if self.machine:
            _logger.info(f"Running on machine: {self.machine}")
        _logger.info(f"running ibllib version {ibllib.__version__}")
        # setup
        start_time = time.time()
        try:
            setup = self.setUp(**kwargs)
            _logger.info(f"Setup value is: {setup}")
            self.status = 0
            if not setup:
                # case where outputs are present but don't have input files locally to rerun task
                # label task as complete
                _, self.outputs = self.assert_expected_outputs()
            else:
                # run task
                if self.gpu >= 1:
                    if not self._creates_lock():
                        self.status = -2
                        _logger.info(f"Job {self.__class__} exited as a lock was found")
                        new_log = log_capture_string.getvalue()
                        self.log = new_log if self.clobber else self.log + new_log
                        _logger.removeHandler(ch)
                        ch.close()
                        return self.status
                self.outputs = self._run(**kwargs)
                _logger.info(f"Job {self.__class__} complete")
        except Exception:
            _logger.error(traceback.format_exc())
            _logger.info(f"Job {self.__class__} errored")
            self.status = -1

        self.time_elapsed_secs = time.time() - start_time
        # log the outputs
        if isinstance(self.outputs, list):
            nout = len(self.outputs)
        elif self.outputs is None:
            nout = 0
        else:
            nout = 1

        _logger.info(f"N outputs: {nout}")
        _logger.info(f"--- {self.time_elapsed_secs} seconds run-time ---")
        # after the run, capture the log output, amend to any existing logs if not overwrite
        new_log = log_capture_string.getvalue()
        self.log = new_log if self.clobber else self.log + new_log
        _logger.removeHandler(ch)
        ch.close()
        _logger.setLevel(logger_level)
        # tear down
        self.tearDown()
        return self.status

    def register_datasets(self, one=None, **kwargs):
        """
        Register output datasets form the task to Alyx
        :param one:
        :param jobid:
        :param kwargs: directly passed to the register_dataset function
        :return:
        """
        _ = self.register_images()

        return self.data_handler.uploadData(self.outputs, self.version, **kwargs)

    def register_images(self, **kwargs):
        """
        Registers images to alyx database
        :return:
        """
        if self.one and len(self.plot_tasks) > 0:
            for plot_task in self.plot_tasks:
                try:
                    _ = plot_task.register_images(widths=['orig'])
                except Exception:
                    _logger.error(traceback.format_exc())
                    continue

    def rerun(self):
        self.run(overwrite=True)

    def get_signatures(self, **kwargs):
        """
        This is the default but should be overwritten for each task
        :return:
        """
        self.input_files = self.signature['input_files']
        self.output_files = self.signature['output_files']

    @abc.abstractmethod
    def _run(self, overwrite=False):
        """
        This is the method to implement
        :param overwrite: (bool) if the output already exists,
        :return: out_files: files to be registered. Could be a list of files (pathlib.Path),
        a single file (pathlib.Path) an empty list [] or None.
        Within the pipeline, there is a distinction between a job that returns an empty list
         and a job that returns None. If the function returns None, the job will be labeled as
          "empty" status in the database, otherwise, the job has an expected behaviour of not
          returning any dataset.
        """

    def setUp(self, **kwargs):
        """
        Setup method to get the data handler and ensure all data is available locally to run task
        :param kwargs:
        :return:
        """
        if self.location == 'server':
            self.get_signatures(**kwargs)

            input_status, _ = self.assert_expected_inputs(raise_error=False)
            output_status, _ = self.assert_expected(self.output_files, silent=True)

            if input_status:
                self.data_handler = self.get_data_handler()
                _logger.info('All input files found: running task')
                return True

            if not self.force:
                self.data_handler = self.get_data_handler()
                _logger.warning('Not all input files found locally: will still attempt to rerun task')
                # TODO in the future once we are sure that input output task signatures work properly should return False
                # _logger.info('All output files found but input files required not available locally: task not rerun')
                return True
            else:
                # Attempts to download missing data using globus
                _logger.info('Not all input files found locally: attempting to re-download required files')
                self.data_handler = self.get_data_handler(location='serverglobus')
                self.data_handler.setUp()
                # Double check we now have the required files to run the task
                # TODO in future should raise error if even after downloading don't have the correct files
                self.assert_expected_inputs(raise_error=False)
                return True
        else:
            self.data_handler = self.get_data_handler()
            self.data_handler.setUp()
            self.get_signatures(**kwargs)
            self.assert_expected_inputs()
            return True

    def tearDown(self):
        """
        Function after runs()
        Does not run if a lock is encountered by the task (status -2)
        """
        if self.gpu >= 1:
            if self._lock_file_path().exists():
                self._lock_file_path().unlink()

    def cleanUp(self):
        """
        Function to optionally overload to clean up
        :return:
        """
        self.data_handler.cleanUp()

    def assert_expected_outputs(self, raise_error=True):
        """
        After a run, asserts that all signature files are present at least once in the output files
        Mainly useful for integration tests
        :return:
        """
        assert self.status == 0
        _logger.info('Checking output files')
        everything_is_fine, files = self.assert_expected(self.output_files)

        if not everything_is_fine:
            for out in self.outputs:
                _logger.error(f"{out}")
            if raise_error:
                raise FileNotFoundError("Missing outputs after task completion")

        return everything_is_fine, files

    def assert_expected_inputs(self, raise_error=True):
        """
        Before running a task, check that all the files necessary to run the task have been downloaded/ are on the local file
        system already
        :return:
        """
        _logger.info('Checking input files')
        everything_is_fine, files = self.assert_expected(self.input_files)

        if not everything_is_fine and raise_error:
            raise FileNotFoundError("Missing inputs to run task")

        return everything_is_fine, files

    def assert_expected(self, expected_files, silent=False):
        everything_is_fine = True
        files = []
        for expected_file in expected_files:
            actual_files = list(Path(self.session_path).rglob(str(Path(expected_file[1]).joinpath(expected_file[0]))))
            if len(actual_files) == 0 and expected_file[2]:
                everything_is_fine = False
                if not silent:
                    _logger.error(f"Signature file expected {expected_file} not found")
            else:
                if len(actual_files) != 0:
                    files.append(actual_files[0])

        return everything_is_fine, files

    def get_data_handler(self, location=None):
        """
        Gets the relevant data handler based on location argument
        :return:
        """
        location = str.lower(location or self.location)
        if location == 'local':
            return data_handlers.LocalDataHandler(self.session_path, self.signature, one=self.one)
        self.one = self.one or ONE()
        if location == 'server':
            dhandler = data_handlers.ServerDataHandler(self.session_path, self.signature, one=self.one)
        elif location == 'serverglobus':
            dhandler = data_handlers.ServerGlobusDataHandler(self.session_path, self.signature, one=self.one)
        elif location == 'remote':
            dhandler = data_handlers.RemoteHttpDataHandler(self.session_path, self.signature, one=self.one)
        elif location == 'aws':
            dhandler = data_handlers.RemoteAwsDataHandler(self, self.session_path, self.signature, one=self.one)
        elif location == 'sdsc':
            dhandler = data_handlers.SDSCDataHandler(self, self.session_path, self.signature, one=self.one)
        else:
            raise ValueError(f'Unknown location "{location}"')
        return dhandler

    @staticmethod
    def make_lock_file(taskname="", time_out_secs=7200):
        """Creates a GPU lock file with a timeout of"""
        d = {'start': time.time(), 'name': taskname, 'time_out_secs': time_out_secs}
        with open(Task._lock_file_path(), 'w+') as fid:
            json.dump(d, fid)
        return d

    @staticmethod
    def _lock_file_path():
        """the lock file is in ~/.one/gpu.lock"""
        folder = Path.home().joinpath('.one')
        folder.mkdir(exist_ok=True)
        return folder.joinpath('gpu.lock')

    def _make_lock_file(self):
        """creates a lock file with the current time"""
        return Task.make_lock_file(self.name, self.time_out_secs)

    def is_locked(self):
        """Checks if there is a lock file for this given task"""
        lock_file = self._lock_file_path()
        if not lock_file.exists():
            return False

        with open(lock_file) as fid:
            d = json.load(fid)
        now = time.time()
        if (now - d['start']) > d['time_out_secs']:
            lock_file.unlink()
            return False
        else:
            return True

    def _creates_lock(self):
        if self.is_locked():
            return False
        else:
            self._make_lock_file()
            return True


class Pipeline(abc.ABC):
    """
    Pipeline class: collection of related and potentially interdependent tasks
    """
    tasks = OrderedDict()
    one = None

    def __init__(self, session_path=None, one=None, eid=None):
        assert session_path or eid
        self.one = one
        if one and one.alyx.cache_mode and one.alyx.default_expiry.seconds > 1:
            _logger.warning('Alyx client REST cache active; this may cause issues with jobs')
        self.eid = eid
        self.data_repo = get_local_data_repository(self.one)
        if session_path:
            self.session_path = session_path
            if not self.eid:
                # eID for newer sessions may not be in cache so use remote query
                self.eid = one.path2eid(session_path, query_type='remote') if self.one else None
        self.label = self.__module__ + '.' + type(self).__name__

    @staticmethod
    def _get_exec_name(obj):
        """
        For a class, get the executable name as it should be stored in Alyx. When the class
        is created dynamically using the type() built-in function, need to revert to the base
        class to be able to re-instantiate the class from the alyx dictionary on the client side
        :param obj:
        :return: string containing the full module plus class name
        """
        if obj.__module__ == 'abc':
            exec_name = f"{obj.__class__.__base__.__module__}.{obj.__class__.__base__.__name__}"
        else:
            exec_name = f"{obj.__module__}.{obj.name}"
        return exec_name

    def make_graph(self, out_dir=None, show=True):
        if not out_dir:
            out_dir = self.one.alyx.cache_dir if self.one else one.params.get().CACHE_DIR
        m = Digraph('G', filename=str(Path(out_dir).joinpath(self.__module__ + '_graphs.gv')))
        m.attr(rankdir='TD')

        e = Digraph(name='cluster_' + self.label)
        e.attr('node', shape='box')
        e.node('root', label=self.label)

        e.attr('node', shape='ellipse')
        for k in self.tasks:
            j = self.tasks[k]
            if len(j.parents) == 0:
                e.edge('root', j.name)
            else:
                [e.edge(p.name, j.name) for p in j.parents]

        m.subgraph(e)
        m.attr(label=r'\n\Pre-processing\n')
        m.attr(fontsize='20')
        if show:
            m.view()
        return m

    def create_alyx_tasks(self, rerun__status__in=None, tasks_list=None):
        """
        Instantiate the pipeline and create the tasks in Alyx, then create the jobs for the session
        If the jobs already exist, they are left untouched. The re-run parameter will re-init the
        job by emptying the log and set the status to Waiting
        :param rerun__status__in: by default no re-run. To re-run tasks if they already exist,
        specify a list of statuses string that will be re-run, those are the possible choices:
        ['Waiting', 'Started', 'Errored', 'Empty', 'Complete']
        to always patch, the string '__all__' can also be provided
        :return: list of alyx tasks dictionaries (existing and or created)
        """
        rerun__status__in = rerun__status__in or []
        if rerun__status__in == '__all__':
            rerun__status__in = ['Waiting', 'Started', 'Errored', 'Empty', 'Complete']
        assert self.eid
        if self.one is None:
            _logger.warning("No ONE instance found for Alyx connection, set the one property")
            return
        tasks_alyx_pre = self.one.alyx.rest('tasks', 'list', session=self.eid, graph=self.name, no_cache=True)
        tasks_alyx = []
        # creates all the tasks by iterating through the ordered dict

        if tasks_list is not None:
            task_items = tasks_list
            # need to add in the session eid and the parents
        else:
            task_items = [t for _, t in self.tasks.items()]

        for t in task_items:
            # get the parents' alyx ids to reference in the database
            if type(t) == dict:
                t = Bunch(t)
                executable = t.executable
                arguments = t.arguments
                t['time_out_secs'] = t['time_out_sec']
                if len(t.parents):
                    pnames = [p for p in t.parents]
            else:
                executable = self._get_exec_name(t)
                arguments = t.kwargs
                if len(t.parents):
                    pnames = [p.name for p in t.parents]

            if len(t.parents):
                parents_ids = [ta['id'] for ta in tasks_alyx if ta['name'] in pnames]
            else:
                parents_ids = []

            task_dict = {'executable': executable, 'priority': t.priority,
                         'io_charge': t.io_charge, 'gpu': t.gpu, 'cpu': t.cpu,
                         'ram': t.ram, 'module': self.label, 'parents': parents_ids,
                         'level': t.level, 'time_out_sec': t.time_out_secs, 'session': self.eid,
                         'status': 'Waiting', 'log': None, 'name': t.name, 'graph': self.name,
                         'arguments': arguments}
            if self.data_repo:
                task_dict.update({'data_repository': self.data_repo})
            # if the task already exists, patch it otherwise, create it
            talyx = next(filter(lambda x: x["name"] == t.name, tasks_alyx_pre), [])
            if len(talyx) == 0:
                talyx = self.one.alyx.rest('tasks', 'create', data=task_dict)
            elif rerun__status__in == '__all__' or talyx['status'] in rerun__status__in:
                talyx = self.one.alyx.rest(
                    'tasks', 'partial_update', id=talyx['id'], data=task_dict)
            tasks_alyx.append(talyx)
        return tasks_alyx

    def create_tasks_list_from_pipeline(self):
        """
        From a pipeline with tasks, creates a list of dictionaries containing task description that can be used to upload to
        create alyx tasks
        :return:
        """
        tasks_list = []
        for k, t in self.tasks.items():
            # get the parents' alyx ids to reference in the database
            if len(t.parents):
                parent_names = [p.name for p in t.parents]
            else:
                parent_names = []

            task_dict = {'executable': self._get_exec_name(t), 'priority': t.priority,
                         'io_charge': t.io_charge, 'gpu': t.gpu, 'cpu': t.cpu,
                         'ram': t.ram, 'module': self.label, 'parents': parent_names,
                         'level': t.level, 'time_out_sec': t.time_out_secs, 'session': self.eid,
                         'status': 'Waiting', 'log': None, 'name': t.name, 'graph': self.name,
                         'arguments': t.kwargs}
            if self.data_repo:
                task_dict.update({'data_repository': self.data_repo})

            tasks_list.append(task_dict)

        return tasks_list

    def run(self, status__in=['Waiting'], machine=None, clobber=True, **kwargs):
        """
        Get all the session related jobs from alyx and run them
        :param status__in: lists of status strings to run in
        ['Waiting', 'Started', 'Errored', 'Empty', 'Complete']
        :param machine: string identifying the machine the task is run on, optional
        :param clobber: bool, if True any existing logs are overwritten, default is True
        :param kwargs: arguments passed downstream to run_alyx_task
        :return: jalyx: list of REST dictionaries of the job endpoints
        :return: job_deck: list of REST dictionaries of the jobs endpoints
        :return: all_datasets: list of REST dictionaries of the dataset endpoints
        """
        assert self.session_path, "Pipeline object has to be declared with a session path to run"
        if self.one is None:
            _logger.warning("No ONE instance found for Alyx connection, set the one property")
            return
        task_deck = self.one.alyx.rest('tasks', 'list', session=self.eid, no_cache=True)
        # [(t['name'], t['level']) for t in task_deck]
        all_datasets = []
        for i, j in enumerate(task_deck):
            if j['status'] not in status__in:
                continue
            # here we update the status in-place to avoid another hit to the database
            task_deck[i], dsets = run_alyx_task(tdict=j, session_path=self.session_path,
                                                one=self.one, job_deck=task_deck,
                                                machine=machine, clobber=clobber)
            if dsets is not None:
                all_datasets.extend(dsets)
        return task_deck, all_datasets

    def rerun_failed(self, **kwargs):
        return self.run(status__in=['Waiting', 'Held', 'Started', 'Errored', 'Empty'], **kwargs)

    def rerun(self, **kwargs):
        return self.run(status__in=['Waiting', 'Held', 'Started', 'Errored', 'Empty', 'Complete', 'Incomplete'],
                        **kwargs)

    @property
    def name(self):
        return self.__class__.__name__


def run_alyx_task(tdict=None, session_path=None, one=None, job_deck=None,
                  max_md5_size=None, machine=None, clobber=True, location='server', mode='log'):
    """
    Runs a single Alyx job and registers output datasets
    :param tdict:
    :param session_path:
    :param one:
    :param job_deck: optional list of job dictionaries belonging to the session. Needed
    to check dependency status if the jdict has a parent field. If jdict has a parent and
    job_deck is not entered, will query the database
    :param max_md5_size: in bytes, if specified, will not compute the md5 checksum above a given
    filesize to save time
    :param machine: string identifying the machine the task is run on, optional
    :param clobber: bool, if True any existing logs are overwritten, default is True
    :param location: where you are running the task, 'server' - local lab server, 'remote' - any
    compute node/ computer, 'SDSC' - flatiron compute node, 'AWS' - using data from aws s3
    :param mode: str ('log' or 'raise') behaviour to adopt if an error occured. If 'raise', it
    will Raise the error at the very end of this function (ie. after having labeled the tasks)
    :return:
    """
    registered_dsets = []
    # here we need to check parents' status, get the job_deck if not available
    if not job_deck:
        job_deck = one.alyx.rest('tasks', 'list', session=tdict['session'], no_cache=True)
    if len(tdict['parents']):
        # check the dependencies
        parent_tasks = filter(lambda x: x['id'] in tdict['parents'], job_deck)
        parent_statuses = [j['status'] for j in parent_tasks]
        # if any of the parent tasks is not complete, throw a warning
        if any(map(lambda s: s not in ['Complete', 'Incomplete'], parent_statuses)):
            _logger.warning(f"{tdict['name']} has unmet dependencies")
            # if parents are waiting or failed, set the current task status to Held
            # once the parents ran, the descendent tasks will be set from Held to Waiting (see below)
            if any(map(lambda s: s in ['Errored', 'Held', 'Empty', 'Waiting', 'Started', 'Abandoned'],
                       parent_statuses)):
                tdict = one.alyx.rest('tasks', 'partial_update', id=tdict['id'],
                                      data={'status': 'Held'})
            return tdict, registered_dsets
    # creates the job from the module name in the database
    exec_name = tdict['executable']
    strmodule, strclass = exec_name.rsplit('.', 1)
    classe = getattr(importlib.import_module(strmodule), strclass)
    tkwargs = tdict.get('arguments') or {}  # if the db field is null it returns None
    task = classe(session_path, one=one, taskid=tdict['id'], machine=machine, clobber=clobber,
                  location=location, **tkwargs)
    # sets the status flag to started before running
    one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'status': 'Started'})
    status = task.run()
    patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                  'version': task.version}
    # if there is no data to register, set status to Empty
    if task.outputs is None:
        patch_data['status'] = 'Empty'
    # otherwise register data and set (provisional) status to Complete
    else:
        try:
            kwargs = dict(one=one, max_md5_size=max_md5_size)
            if location == 'server':
                # Explicitly pass lab as lab cannot be inferred from path
                kwargs['labs'] = ','.join(get_lab(one.alyx))
            registered_dsets = task.register_datasets(**kwargs)
            patch_data['status'] = 'Complete'
        except Exception:
            _logger.error(traceback.format_exc())
            status = -1

    # overwrite status to errored
    if status == -1:
        patch_data['status'] = 'Errored'
    # Status -2 means a lock was encountered during run, should be rerun
    if status == -2:
        patch_data['status'] = 'Waiting'
    # Status -3 should be returned if a task is Incomplete
    if status == -3:
        patch_data['status'] = 'Incomplete'
    # update task status on Alyx
    t = one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
    # check for dependent held tasks
    # NB: Assumes dependent tasks are all part of the same session!
    next(x for x in job_deck if x['id'] == t['id'])['status'] = t['status']  # Update status in job deck
    dependent_tasks = filter(lambda x: t['id'] in x['parents'] and x['status'] == 'Held', job_deck)
    for d in dependent_tasks:
        assert d['id'] != t['id'], 'task its own parent'
        # if all their parent tasks now complete, set to waiting
        parent_status = [next(x['status'] for x in job_deck if x['id'] == y) for y in d['parents']]
        if all(x in ['Complete', 'Incomplete'] for x in parent_status):
            one.alyx.rest('tasks', 'partial_update', id=d['id'], data={'status': 'Waiting'})
    task.cleanUp()
    if mode == 'raise' and status != 0:
        raise ValueError(task.log)
    return t, registered_dsets

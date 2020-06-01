from pathlib import Path
import abc
import logging
import io
import importlib
import time
from _collections import OrderedDict

from graphviz import Digraph

from ibllib.misc import version
from ibllib.io import params
from oneibl.registration import register_dataset


_logger = logging.getLogger('ibllib')


def alyx_setup_teardown(run):
    """
    Performs the job loading using Alyx
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        # if taskid of one properties are not available, local run only
        if self.one is None or self.taskid is None:
            return run(*args, **kwargs)
        # setup
        self.one.alyx.rest('tasks', 'partial_update', id=self.taskid, data={'status': 'Started'})
        # run
        out = run(*args, **kwargs)
        # teardown
        if self.outputs:
            register_dataset(self.outputs, one=self.one)
        status = 'Complete' if self.status == 0 else 'Errored'
        self.one.alyx.rest('tasks', 'partial_update', id=self.taskid,
                           data={'status': status, 'log': self.log})
        return out
    return wrapper


class Task(abc.ABC):
    log = ""
    cpu = 1
    gpu = 0
    io_charge = 5  # integer percentage
    priority = 30  # integer percentage, 100 means highest priority
    ram = 4  # RAM needed to run (Go)
    one = None  # one instance (optional)
    level = 0
    outputs = None
    time_elapsed_secs = None
    time_out_secs = None
    version = version.ibllib()

    def __init__(self, session_path, parents=None, taskid=None, one=None):
        assert session_path
        self.taskid = taskid
        self.one = one
        self.session_path = session_path
        if parents:
            self.parents = parents
        else:
            self.parents = []

    @property
    def name(self):
        return self.__class__.__name__

    @alyx_setup_teardown
    def run(self, **kwargs):
        """
        --- do not overload, see _run() below---
        wraps the _run() method with
        -   error management
        -   logging to variable
        """
        # setup
        self.setUp()
        # Setup the console handler with a StringIO object
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        str_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        ch.setFormatter(logging.Formatter(str_format))
        _logger.addHandler(ch)
        _logger.info(f"Starting job {self.__class__}")
        # run
        start_time = time.time()
        try:
            self.outputs = self._run(**kwargs)
            self.status = 0
            _logger.info(f"Job {self.__class__} complete")
        except Exception as e:
            _logger.error(f"{e}")
            _logger.info(f"Job {self.__class__} errored")
            self.status = -1
        self.time_elapsed_secs = time.time() - start_time
        _logger.info(f"--- {self.time_elapsed_secs} seconds run-time ---")
        # after the run, capture the log output
        self.log = log_capture_string.getvalue()
        log_capture_string.close()
        _logger.removeHandler(ch)
        # tear down
        self.tearDown()
        return self.status

    def register_datasets(self, one=None, jobid=None):
        assert one
        assert jobid
        if self.outputs:
            if isinstance(self.outputs, list):
                versions = [self.version for _ in self.outputs]
            else:
                versions = [self.version]
            return register_dataset(self.outputs, one=one, versions=versions)

    def rerun(self):
        self.run(overwrite=True)

    @abc.abstractmethod
    def _run(self, overwrite=False):
        """
        This is the method to implement
        :param overwrite: (bool) if the output already exists,
        :return: out_files: files to be registered. Could be a list of files (pathlib.Path),
        a single file (pathlib.Path) an empty list [] or None.
        Whithin the pipeline, there is a distinction between a job that returns an empty list
         and a job that returns None. If the function returns None, the job will be labeled as
          "empty" status in the database, otherwise, the job has an expected behaviour of not
          returning any dataset.
        """

    def setUp(self):
        """
        Function to optionally overload to check inputs.
        :return:
        """

    def tearDown(self):
        """
        Function to optionally overload to check results
        :return:
        """


class Pipeline(abc.ABC):
    tasks = OrderedDict()
    one = None

    def __init__(self, session_path=None, one=None):
        self.one = one
        self.session_path = session_path
        self.eid = one.eid_from_path(session_path) if self.one else None
        self.label = self.__module__ + '.' + type(self).__name__

    def make_graph(self, out_dir=None, show=True):
        if not out_dir:
            par = params.read('one_params')
            out_dir = par.CACHE_DIR
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

    def create_alyx_tasks(self, rerun__status__in=None):
        """
        Instantiate the pipeline and create the tasks in Alyx, then create the jobs for the session
        If the jobs already exist, they are left untouched. The re-run parameter will re-init the
        job by emptying the log and set the status to Waiting
        :param rerun__status__in: by default no re-run. To re-run tasks if they already exist,
        specify a list of statuses string that will be re-run, those are the possible choices:
        ['Waiting', 'Started', 'Errored', 'Empty', 'Complete']
        :return: list of alyx tasks dictionaries (existing and or created)
        """
        rerun__status__in = rerun__status__in or []
        assert self.eid
        if self.one is None:
            _logger.warning("No ONE instance found for Alyx connection, set the one property")
            return
        tasks_alyx = self.one.alyx.rest('tasks', 'list', session=self.eid)
        # creates all the tasks by iterating through the ordered dict
        for k, t in self.tasks.items():
            # get the parents alyx ids to reference in the database
            if len(t.parents):
                pnames = [p.name for p in t.parents]
                parents_ids = [ta['id'] for ta in tasks_alyx if ta['name'] in pnames]
            else:
                parents_ids = []
            task_dict = {'executable': f"{t.__module__}.{t.name}", 'priority': t.priority,
                         'io_charge': t.io_charge, 'gpu': t.gpu, 'cpu': t.cpu,
                         'ram': t.ram, 'module': self.label, 'parents': parents_ids,
                         'level': t.level, 'time_out_sec': t.time_out_secs, 'session': self.eid,
                         'status': 'Waiting', 'log': None, 'name': t.name}
            # if the task already exists, patch it otherwise, create it
            talyx = next(filter(lambda x: x["name"] == t.name, tasks_alyx), [])
            if len(talyx) == 0:
                talyx = self.one.alyx.rest('tasks', 'create', data=task_dict)
            elif talyx['status'] in rerun__status__in:
                talyx = self.one.alyx.rest(
                    'tasks', 'partial_update', id=talyx[0]['id'], data=task_dict)
            tasks_alyx.append(talyx)
        return tasks_alyx

    def run(self, status__in=['Waiting']):
        """
        Get all the session related jobs from alyx and run them
        :return: jalyx: list of REST dictionaries of the job endpoints
        :return: job_deck: list of REST dictionaries of the jobs endpoints
        :return: all_datasets: list of REST dictionaries of the dataset endpoints
        """
        if self.one is None:
            _logger.warning("No ONE instance found for Alyx connection, set the one property")
            return
        task_deck = self.one.alyx.rest('tasks', 'list', session=self.eid)
        all_datasets = []
        for i, j in enumerate(task_deck):
            if j['status'] not in status__in:
                continue
            # here we update the status in-place to avoid another hit to the database
            task_deck[i], dsets = _run_alyx_task(tdict=j, session_path=self.session_path,
                                                 one=self.one, job_deck=task_deck)
            if dsets is not None:
                all_datasets.extend(dsets)
        return task_deck, all_datasets

    def rerun_failed(self):
        return self.run(status__in=['Waiting', 'Started', 'Errored', 'Empty'])

    def rerun(self):
        return self.run(status__in=['Waiting', 'Started', 'Errored', 'Empty', 'Complete'])


def _run_alyx_task(tdict=None, session_path=None, one=None, job_deck=None):
    """
    Runs a single Alyx job and registers output datasets
    :param tdict:
    :param session_path:
    :param one:
    :param job_deck: optional list of job dictionaries belonging to the session. Needed
    to check dependency status if the jdict has a parent field. If jdict has a parent and
    job_deck is not entered, will query the database
    :return:
    """
    registered_dsets = []
    if len(tdict['parents']):
        # here we need to check parents status, get the job_deck if not available
        if not job_deck:
            job_deck = one.alyx.rest('tasks', 'list', session=tdict['session'])
        # check the dependencies
        if not all(list(map(lambda x: x['status'] == 'Complete',
                            filter(lambda x: x['id'] in tdict['parents'], job_deck)))):
            _logger.warning(f"{tdict['name']} has unmet dependencies")
            return tdict, registered_dsets
    # creates the job from the module name in the database
    exec_name = tdict['executable']
    strmodule, strclass = exec_name.rsplit('.', 1)
    classe = getattr(importlib.import_module(strmodule), strclass)
    task = classe(session_path, one=one, taskid=tdict['id'])
    # sets the status flag to started before running
    one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data={'status': 'Started'})
    status = task.run()
    patch_data = {'time_elapsed_secs': task.time_elapsed_secs, 'log': task.log,
                  'version': task.version}
    # only registers successful runs
    if status == 0:
        # on a successful run, if there is no data to register, set status to Empty
        if task.outputs is None:
            patch_data['status'] = 'Empty'
        else:  # otherwise register data and set status to Complete
            registered_dsets = task.register_datasets(one=one, jobid=tdict['id'])
            patch_data['status'] = 'Complete'
    elif status == -1:
        patch_data['status'] = 'Errored'
    t = one.alyx.rest('tasks', 'partial_update', id=tdict['id'], data=patch_data)
    return t, registered_dsets

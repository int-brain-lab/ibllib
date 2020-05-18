from pathlib import Path
import abc
import logging
import io
import importlib

from graphviz import Digraph

from oneibl.one import ONE
from oneibl.registration import register_dataset
from ibllib.io import params

_logger = logging.getLogger('ibllib')


def alyx_setup_teardown(run):
    """
    Performs the job loading using Alyx
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        # if jobid of one properties are not available, local run only
        if self.one is None or self.jobid is None:
            return run(*args, **kwargs)
        # setup
        self.one.alyx.rest('jobs', 'partial_update', id=self.jobid, data={'status': 'Started'})
        # run
        out = run(*args, **kwargs)
        # teardown
        if self.outputs:
            register_dataset(self.outputs, one=self.one)
        status = 'Complete' if self.status == 0 else 'Errored'
        self.one.alyx.rest('jobs', 'partial_update', id=self.jobid,
                           data={'status': status, 'log': self.log})
        return out
    return wrapper


class Job(abc.ABC):
    log = ""
    cpu = 1
    gpu = 0
    io_charge = 5  # integer percentage
    priority = 30  # integer percentage, 100 means highest priority
    ram = 4  # RAM needed to run (Go)
    one = None  # one instance (optional)

    def __init__(self, session_path, parents=None, jobid=None, one=None):
        assert session_path
        self.jobid = jobid
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
    def run(self, status=0, **kwargs):
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
        try:
            self.outputs = self._run(**kwargs)
            self.status = 0
            _logger.info(f"Job {self.__class__} complete")
        except Exception as e:
            _logger.error(f"{e}")
            _logger.info(f"Job {self.__class__} errored")
            self.status = -1
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
            register_dataset(self.outputs, one=one)

    def rerun(self):
        self.run(overwrite=True)

    @abc.abstractmethod
    def _run(self, overwrite=False):
        """
        This is the method to implement
        :param overwrite: (bool) if the output already exists,
        :return: out_files: a list of files (pathlib.Path) to be registered
        """
        pass

    def setUp(self):
        """
        Function to optionally overload to check inputs.
        :return:
        """

    def tearDown(self):
        """
        Function to optionally overload to check results
        :return: 0 if successful, -1 if failed
        """


class Pipeline(abc.ABC):
    jobs = None
    label = ''
    one = None

    def __init__(self, session_path=None, one=None):
        if one is None:
            one = ONE()
        self.one = one
        self.session_path = None
        self.eid = one.eid_from_path(session_path)

    def make_graph(self, out_dir=None, show=True):
        if not out_dir:
            par = params.read('one_params')
            out_dir = par.CACHE_DIR
        m = Digraph('G', filename=str(Path(out_dir).joinpath(self.label + '_graphs.gv')))
        m.attr(rankdir='TD')

        e = Digraph(name='cluster_' + self.label)
        e.attr('node', shape='box')
        e.node('root', label=self.label)

        e.attr('node', shape='ellipse')
        for k in self.jobs:
            j = self.jobs[k]
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

    def init_alyx_tasks(self):
        """Used only when creating a new pipeline"""
        tasks_alyx = []
        for k in self.jobs:
            j = self.jobs[k]
            # self.one.alyx.rest('tasks', 'create')
            task_dict = {'name': j.name, 'priority': j.priority, 'io_charge': j.io_charge,
                         'gpu': j.gpu, 'cpu': j.cpu, 'ram': j.ram, 'pipeline': self.label,
                         'level': j.level}
            if len(j.parents):
                task_dict['parents'] = [p.name for p in j.parents]
            task = self.one.alyx.rest('tasks', 'list', name=j.name)
            if len(task) == 0:
                task = self.one.alyx.rest('tasks', 'create', data=task_dict)
            else:
                task = task[0]
            tasks_alyx.append(task)
        return tasks_alyx

    def register_alyx_jobs(self, rerun=False):
        """
        Instantiate the pipeline and create the tasks in Alyx, then create the jobs for the session
        If the jobs already exist, they are left untouched. The re-run parameter will re-init the
        job by emptying the log and set the status to Waiting
        :param rerun:
        :return:
        """
        jobs_alyx = []
        for k in self.jobs:
            job = self.jobs[k]
            jalyx = self.one.alyx.rest('jobs', 'list', session=self.eid, task=job.name)
            jdict = {'session': self.eid, 'task': job.name, 'status': 'Waiting', 'log': None}
            if len(jalyx) == 0:
                jalyx = self.one.alyx.rest('jobs', 'create', data=jdict)
            elif rerun:
                jalyx = self.one.alyx.rest('jobs', 'partial_update', id=jalyx[0]['id'], data=jdict)
            else:
                jalyx = jalyx[0]
            jobs_alyx.append(jalyx)
        return jobs_alyx


def run_alyx_job(jdict=None, session_path=None, one=None):
    """
    :param jdict:
    :param session_path:
    :param one:
    :return:
    """
    modulename = jdict['pipeline']
    module = importlib.import_module(modulename)
    classe = getattr(module, jdict['task'])
    job = classe(session_path, one=one, jobid=jdict['id'])
    status = job.run(session_path)
    # only registers successful runs
    if status == 0:
        registered_dsets = job.register_datasets(one=one, jobid=jdict['id'])
        one.alyx.rest('jobs', 'partial_update', id=jdict['id'], data={'status': 'Complete'})
    elif status == -1:
        registered_dsets = []
        one.alyx.rest('jobs', 'partial_update', id=jdict['id'], data={'status': 'Errored'})
    return status, registered_dsets

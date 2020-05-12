from pathlib import Path
import abc
import logging
import io

from graphviz import Digraph

from oneibl.one import ONE
from ibllib.io.extractors import (ephys_trials, ephys_fpga)
from ibllib.io import raw_data_loaders as rawio
from ibllib.io import params

_logger = logging.getLogger('ibllib')


class Task(abc.ABC):
    log = ""
    cpu = 1
    gpu = 0
    io_charge = 5  # integer percentage
    priority = 30  # integer percentage, 100 means highest priority
    ram = 4  # RAM needed to run (Go)
    parents = None  # list of Job instances
    inputs = None  # list of pathlib.Path
    outputs = None  # list of pathlib.Path
    session_path = None

    def __init__(self, session_path, parents=None):
        assert session_path  # todo eid swap
        self.session_path = session_path
        if parents:
            self.parents = parents
        else:
            self.parents = []

    @property
    def name(self):
        return self.__class__.__name__

    def run(self, status=0, **kwargs):
        """
        --- do not overload, see _run() below---
        wraps the _run() method with
        -   error management
        -   logging to variable
        """
        if status == -1:
            return
        # Setup the console handler with a StringIO object
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        str_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
        ch.setFormatter(logging.Formatter(str_format))
        _logger.addHandler(ch)
        try:
            self.outputs = self._run(**kwargs)
            self.status = self.tearDown()
            # afte the run, capture the log output
        except Exception as e:
            _logger.error(f"{e}")
            self.status = -1
        self.log = log_capture_string.getvalue()
        log_capture_string.close()
        _logger.removeHandler(ch)
        return self.status

    def register(self):
        # TODO
        # 1) register files
        # 2) update the job status
        pass

    def setUp(self):
        """
        Function to overload to check if it is a re-run ?
        :return:
        """
        pass

    def tearDown(self):
        """
        Function to overload to check results
        :return: 0 if successful, -1 if failed
        """
        return 0

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


class EphysPulses(Task):
    cpu = 2
    io_charge = 20  # this jobs reads raw ap files
    priority = 90  # a lot of jobs depend on this one

    def _run(self, overwrite=False):
        syncs, out_files = ephys_fpga.extract_sync(self.session_path, overwrite=overwrite)
        return out_files


class EphysTrials(Task):
    priority = 90

    def _run(self):
        data = rawio.load_data(self.session_path)
        _logger.info('extract BPOD for ephys session')
        ephys_trials.extract_all(self.session_path, data=data, save=True)
        _logger.info('extract FPGA information for ephys session')
        tmax = data[-1]['behavior_data']['States timestamps']['exit_state'][0][-1] + 60
        ephys_fpga.extract_all(self.session_path, save=True, tmax=tmax)


#
class Pipeline(abc.ABC):
    jobs = None
    label = ''
    one = None
    session_path = None

    def __init__(self, session_path=None, one=None):
        if one is None:
            one = ONE()
        self.one = one
        self.session_path = None
        self.eid = one.eid_from_path(session_path)

    def make_graph(self, out_dir=None):
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
        m.view()

    def init_alyx_tasks(self):
        """Used only when creating a new pipeline"""
        for k in self.jobs:
            j = self.jobs[k]
            task = self.one.alyx.rest('tasks', 'list', name=j.name)
            if len(task) == 0:
                # self.one.alyx.rest('tasks', 'create')
                task_dict = {'name': j.name, 'priority': j.priority, 'io_charge': j.io_charge,
                             'gpu': j.gpu, 'cpu': j.cpu, 'ram': j.ram, 'pipeline': self.label}
                if len(j.parents):
                    task_dict['parents'] = [p.name for p in j.parents]
                self.one.alyx.rest('tasks', 'create', data=task_dict)
                print(task_dict)

    def register_alyx_jobs(self):
        """To be run on session creation"""
        jobs_alyx = []
        for k in self.jobs:
            job = self.jobs[k]
            jalyx = self.one.alyx.rest('jobs', 'list', session=self.eid, task=job.name)
            if len(jalyx) == 0:
                jdict = {'session': self.eid, 'task': job.name}
                jalyx = self.one.alyx.rest('jobs', 'create', data=jdict)
            else:
                jalyx = jalyx[0]
            jobs_alyx.append(jalyx)

        return jobs_alyx


class EphysExtractionPipeline(Pipeline):
    label = 'Ephys'

    def __init__(self, *args, **kwargs):
        super(EphysExtractionPipeline, self).__init__(*args, **kwargs)
        jobs = {}
        jobs['EphysPulses'] = EphysPulses(session_path)
        jobs['EphysTrials'] = EphysTrials(session_path, parents=[jobs['EphysPulses']])
        self.session_path = session_path
        self.jobs = jobs


session_path = "/datadisk/Data/IntegrationTests/ephys/choice_world_init/KS022/2019-12-10/001"
one = ONE(base_url='http://localhost:8000')
ephys_pipe = EphysExtractionPipeline(session_path, one=one)

# ephys_pipe.make_graph()
ephys_pipe.init_alyx_tasks()
# alyx_jobs = ephys_pipe.register_alyx_jobs()

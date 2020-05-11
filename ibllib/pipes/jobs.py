import abc
import logging
import io

import graphviz

from ibllib.pipes.extract_session import get_session_extractor_type
from ibllib.io.extractors import (ephys_trials, ephys_fpga,
                                  biased_wheel, biased_trials,
                                  training_trials, training_wheel)
from ibllib.io import raw_data_loaders as rawio

_logger = logging.getLogger('ibllib')


class Job(abc.ABC):
    log = ""
    n_cpu = 1
    n_gpu = 0
    io_charge = 5  # integer percentage
    priority = 30  # integer percentage, 100 means highest priority
    ram = 4  # RAM needed to run (Go)
    parent = None  # list of Job instances
    inputs = None  # list of pathlib.Path
    outputs = None  # list of pathlib.Path
    session_path = None

    def __init__(self, session_path, parent=None):
        assert session_path  # todo eid swap
        self.session_path = session_path
        self.parent = parent

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
        # ch.setFormatter(formatter)
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


class EphysPulses(Job):
    n_cpu = 2
    io_charge = 20  # this jobs reads raw ap files
    priority = 90  # a lot of jobs depend on this one

    def _run(self, overwrite=False):
        syncs, out_files = ephys_fpga.extract_sync(self.session_path, overwrite=overwrite)
        return out_files


class EphysTrials(Job):
    priority = 90
    parent_class = EphysPulses

    def _run(self):
        data = rawio.load_data(self.session_path)
        _logger.info('extract BPOD for ephys session')
        ephys_trials.extract_all(self.session_path, data=data, save=True)
        _logger.info('extract FPGA information for ephys session')
        tmax = data[-1]['behavior_data']['States timestamps']['exit_state'][0][-1] + 60
        ephys_fpga.extract_all(self.session_path, save=True, tmax=tmax)


class Scheduler():
    pass


##
class Pipeline(abc.ABC):
    jobs = None
    label = ''

    @abc.abstractmethod
    def __init__(self):
        pass

    def make_graph(self):
        from graphviz import Digraph

        m = Digraph('G', filename='job_graphs.gv')
        m.attr(rankdir='TD')

        e = Digraph(name='cluster_ephys')
        e.attr('node', shape='box')
        e.node('root', label=self.label)

        e.attr('node', shape='ellipse')
        for k in self.jobs:
            j = self.jobs[k]
            e.edge(j.parent.name if j.parent else 'root', j.name)

        m.subgraph(e)
        m.attr(label=r'\n\Pre-processing\n')
        m.attr(fontsize='20')

        m.view()


    def run(self):
        pass

    def send2alyx(self):
        pass


class EphysExtractionPipeline(Pipeline):
    label = 'Ephys'

    def __init__(self, session_path):
        jobs = {}
        jobs['EphysPulses'] = EphysPulses(session_path)
        jobs['EphysTrials'] = EphysTrials(session_path, parent=jobs['EphysPulses'])
        self.jobs = jobs


session_path = "/datadisk/Data/IntegrationTests/ephys/choice_world_init/KS022/2019-12-10/001"
self = EphysExtractionPipeline(session_path=session_path)
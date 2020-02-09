"""IBL pipeline.

A task is defined by:

- a name (also name of a Python function)
- a list of relative paths (possibly with a *) to input files/directories (relative to the
  session directory)
- a list of relative paths (possibly with a *) to output files/directories (relative to the
  session directory)
- a list of dependencies (other task names)
- n_cpus
- n_gpus
- io_charge (integer between 0 and 100)
- priority

A scheduler (possibly based on dask) accepts a list of tasks and executes them
on the available resources (CPUs, GPUs).

The list of required tasks is obtained by scanning the existing directory for
tasks that need to be run.

"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from datetime import datetime
from itertools import chain
import logging
from pathlib import Path
import time

from dask import visualize
from dask.distributed import Client, LocalCluster

from alf.io import is_session_path

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

def get_full_paths(session_dir, path_templates):
    """Return a list of paths matching the given templates in the session directory."""
    return list(chain(*(session_dir.glob(tmp) for tmp in path_templates)))


def all_file_globs_exist(session_dir, path_templates):
    """Return whether all path templates (with a wildcard *) exist or not."""
    return (
        len(path_templates) > 0 and
        0 not in [len(list(session_dir.glob(p))) for p in path_templates])


STATUS_LIST = ('pending', 'processing', 'completed', 'error')


def find_session_dirs(roots):
    """Recursively find a list of session dirs in one or several root directories."""
    for r in roots:
        for subdir in Path(r).rglob('*'):
            if subdir.is_dir() and is_session_path(subdir):
                yield subdir


# -------------------------------------------------------------------------------------------------
# Base task class
# -------------------------------------------------------------------------------------------------

class Task:
    """Base class for the tasks."""

    # To set in the child classes.
    name = None
    input_path_templates = ()
    output_path_templates = ()
    depends_on = ()

    # Charge.
    n_cpus = 0
    n_gpus = 0
    io_charge = 0
    priority = None

    # Task status.
    # See also # https://distributed.dask.org/en/latest/scheduling-state.html#task-state
    _status = 'pending'
    started = None
    completed = None

    def __init__(self, session_dir, **kwargs):
        self.__dict__.update(kwargs)
        self.session_dir = Path(session_dir).resolve()
        self.input_paths = get_full_paths(self.session_dir, self.input_path_templates)
        if self.is_complete():
            self.status = 'completed'

    def run(self):
        """To override."""
        print("run", self)
        time.sleep(10)
        #
        # for i in range(10):
        # return np.exp(np.random.normal(int(1e2)))

    @property
    def status(self):
        """Task status: one of %s.""" % ', '.join(STATUS_LIST)
        return self._status

    @status.setter
    def status(self, status):
        assert status in STATUS_LIST
        self._status = status

    def all_input_files_exist(self):
        """Return whether all input files exist so that the task can run."""
        return all_file_globs_exist(self.session_dir, self.input_path_templates)

    def all_output_files_exist(self):
        """Return whether all output files have been created."""
        return all_file_globs_exist(self.session_dir, self.output_path_templates)

    def is_complete(self):
        """Return whether the task is complete, by default, if all output files have been created.
        This method can be overriden for custom checks that the task has successfully completed."""
        return self.all_output_files_exist()

    def __repr__(self):
        return '<Task %s (%s) in %s>' % (self.name, self.status, self.session_dir)


# -------------------------------------------------------------------------------------------------
# IBL tasks
# -------------------------------------------------------------------------------------------------

class ExtractBehaviorTask(Task):
    name = 'extract_behaviour'
    output_path_templates = ('alf/_ibl_trials.*',)
    n_cpus = 1
    io_charge = 50
    priority = 'high'


class RegisterTask(Task):
    name = 'register'
    n_cpus = 1
    io_charge = 10
    priority = 'high'


class CompressVideoTask(Task):
    name = 'compress_avi'
    input_path_templates = ('raw_video_data/*.avi',)
    output_path_templates = ('raw_video_data/*.mp4',)
    n_cpus = 4
    io_charge = 50
    priority = 'normal'


class RawEphysQCTask(Task):
    name = 'raw_ephys_qc'
    input_path_templates = ('raw_ephys_data/probe*/*.ap.bin',)
    output_path_templates = (
        'raw_ephys_data/probe*/_spikeglx_ephysQcTime',
        'raw_ephys_data/probe*/_spikeglx_ephysQcFreq')
    n_cpus = 4
    priority = 'normal'


class ExtractEphysTask(Task):
    name = 'extract_ephys'
    input_path_templates = ('raw_ephys_data/probe*/*.ap.bin',)
    output_path_templates = ('alf/_ibl_trials.*', 'raw_ephys_data/probe*/_spikeglx_sync*')
    n_cpus = 4
    io_charge = 50
    priority = 'high'


class SpikeSortingTask(Task):
    name = 'spike_sorting'
    input_path_templates = ('raw_ephys_data/probe*/*.ap.bin',)
    output_path_templates = ('raw_ephys_data/probe*/*.npy',)
    n_gpus = 1
    priority = 'high'


class CompressEphysTask(Task):
    name = 'compress_ephys'
    input_path_templates = ('raw_ephys_data/probe*/*.ap.bin',)
    output_path_templates = ('raw_ephys_data/probe*/*.ap.cbin',)
    n_cpus = 1
    priority = 'low'


class CompressVideoEphysTask(Task):
    name = 'compress_ephys_avi'
    input_path_templates = ()
    output_path_templates = ('raw_video_data/*.mp4',)
    n_cpus = 4
    io_charge = 50
    priority = 'normal'


class DLCTask(Task):
    name = 'dlc'
    input_path_templates = ('raw_video_data/*.mp4',)
    output_path_templates = ('alf/_dlc_*',)
    n_gpus = 1
    priority = 'normal'
    depends_on = ('compress_avi', 'compress_ephys_avi')


class SpikeSortingQCTask(Task):
    name = 'qc_spike_sorting'
    n_cpus = 1
    priority = 'high'
    depends_on = ('spike_sorting',)


class SpikeSortingMerge(Task):
    name = 'merge_spike_sorting'
    n_cpus = 1
    priority = 'high'


TASK_CLASSES = {
    'compress_avi': CompressVideoTask,
    'compress_ephys': CompressEphysTask,
    'compress_ephys_avi': CompressVideoEphysTask,
    'dlc': DLCTask,
    'extract_behaviour': ExtractBehaviorTask,
    'extract_ephys': ExtractEphysTask,
    'merge_spike_sorting': SpikeSortingMerge,
    'qc_spike_sorting': SpikeSortingQCTask,
    'raw_ephys_qc': RawEphysQCTask,
    'register': RegisterTask,
    'spike_sorting': SpikeSortingTask,
}


# -------------------------------------------------------------------------------------------------
# Pipeline routines
# -------------------------------------------------------------------------------------------------

def missing_tasks(session_dir):
    """Create a list of tasks in a session directory, by scanning the directory and
    finding which tasks have not run yet and thus have not yet created the output files."""
    for task_cls in TASK_CLASSES.values():
        task = task_cls(session_dir)
        if task.status != 'completed':
            yield task


def run_task(name, session_dir, dependencies=()):
    task = TASK_CLASSES[name](session_dir)
    task.started = datetime.now()
    logger.info("Starting task %s on %s.", name, session_dir)
    task.run()
    logger.info("Finished task %s on %s.", name, session_dir)
    task.status = 'completed'
    task.completed = datetime.now()
    return 0


def _count(l):
    return (l)


class Pipeline:
    def __init__(self, root_dir, png_path=None):
        self.session_dirs = list(find_session_dirs((root_dir,)))
        d = {}
        for session_dir in self.session_dirs:
            session_dir = str(session_dir)
            for task in missing_tasks(session_dir):
                d[('task_name', task.name)] = task.name
                dependencies = [(dt, session_dir) for dt in task.depends_on]
                d[(task.name, session_dir)] = (
                    run_task, ('task_name', task.name), session_dir, dependencies)
            d[('end', session_dir)] = (
                _count, [(task_name, session_dir) for task_name in TASK_CLASSES.keys()])
        if png_path:
            visualize(d, filename=png_path)
        self.graph = d
        self.create_cluster()

    def create_cluster(self):
        self.cluster = LocalCluster(
            n_workers=1, processes=False, silence_logs=logging.DEBUG)
        self.client = Client(self.cluster)

    def run(self):
        # TODO: check priority io etc
        # TODO: continuous server of dashboard
        # the get method computes the dask graph
        return self.client.get(
            self.graph, [('end', session_dir) for session_dir in self.session_dirs])


# -------------------------------------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    p = Pipeline('.')
    p.run()

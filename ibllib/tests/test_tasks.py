import tempfile
import unittest
from pathlib import Path
from collections import OrderedDict

from ibllib.misc import version
import ibllib.pipes.tasks
from one.api import ONE
from ibllib.tests import TEST_DB

one = ONE(**TEST_DB)
SUBJECT_NAME = 'algernon'
USER_NAME = 'test_user'

ses_dict = {
    'subject': SUBJECT_NAME,
    'start_time': '2018-04-01T12:48:26.795526',
    'number': 1,
    'users': [USER_NAME]
}

desired_statuses = {
    'Task00': 'Complete',
    'Task01_void': 'Empty',
    'Task02_error': 'Errored',
    'Task10': 'Complete',
    'Task11': 'Held',
    'TaskIncomplete': 'Incomplete',
    'TaskGpuLock': 'Waiting'
}

desired_datasets = ['spikes.times.npy', 'spikes.amps.npy', 'spikes.clusters.npy']
desired_versions = {'spikes.times.npy': 'custom_job00',
                    'spikes.amps.npy': version.ibllib(),
                    'spikes.clusters.npy': version.ibllib()}
desired_logs = 'Running on machine: testmachine'
desired_logs_rerun = {
    'Task00': 1,
    'Task01_void': 2,
    'Task02_error': 1,
    'Task10': 1,
    'Task11': 1,
    'TaskIncomplete': 1,
    'TaskGpuLock': 2
}


#  job to output a single file (pathlib.Path)
class Task00(ibllib.pipes.tasks.Task):
    version = 'custom_job00'

    def _run(self, overwrite=False):
        out_files = self.session_path.joinpath('alf', 'spikes.times.npy')
        out_files.touch()
        return out_files


#  job that outputs nothing
class Task01_void(ibllib.pipes.tasks.Task):

    def _run(self, overwrite=False):
        out_files = None
        return out_files


# job that raises an error on first run
class Task02_error(ibllib.pipes.tasks.Task):
    run_count = 0

    def _run(self, overwrite=False):
        Task02_error.run_count += 1
        if Task02_error.run_count == 1:
            raise Exception('Something dumb happened')
        out_files = self.session_path.joinpath('alf', 'spikes.templates.npy')
        out_files.touch()
        return out_files


# job that outputs a list of files
class Task10(ibllib.pipes.tasks.Task):
    level = 1

    def _run(self, overwrite=False):
        out_files = [
            self.session_path.joinpath('alf', 'spikes.amps.npy'),
            self.session_path.joinpath('alf', 'spikes.clusters.npy')]
        [f.touch() for f in out_files]
        return out_files


#  job to output a single file (pathlib.Path)
class Task11(ibllib.pipes.tasks.Task):
    level = 1

    def _run(self, overwrite=False):
        out_files = self.session_path.joinpath('alf', 'spikes.samples.npy')
        out_files.touch()
        return out_files


#  Job that encounters a GPU lock and is set to Waiting
class TaskGpuLock(ibllib.pipes.tasks.Task):
    gpu = 1

    # Overwrite setUp to create a lock file before running the task and remove it after
    def setUp(self):
        self.make_lock_file()
        self.data_handler = self.get_data_handler()
        return True

    def _run(self, overwrite=False):
        pass


#  Job that encounters a GPU lock and is set to Waiting
class TaskIncomplete(ibllib.pipes.tasks.Task):

    def _run(self, overwrite=False):
        self.status = -3


class SomePipeline(ibllib.pipes.tasks.Pipeline):

    def __init__(self, session_path=None, **kwargs):
        super(SomePipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['Task00'] = Task00(self.session_path)
        tasks['Task01_void'] = Task01_void(self.session_path)
        tasks['Task02_error'] = Task02_error(self.session_path)
        tasks['TaskGpuLock'] = TaskGpuLock(self.session_path)
        tasks['TaskIncomplete'] = TaskIncomplete(self.session_path)
        tasks['Task10'] = Task10(self.session_path, parents=[tasks['Task00']])
        # When both its parents Complete, this task should be set to Waiting and should finally complete
        tasks['Task11'] = Task11(self.session_path, parents=[tasks['Task02_error'],
                                                             tasks['Task00']])
        self.tasks = tasks


class TestPipelineAlyx(unittest.TestCase):

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()

        ses = one.alyx.rest('sessions', 'list', subject=ses_dict['subject'],
                            date_range=[ses_dict['start_time'][:10]] * 2,
                            number=ses_dict['number'],
                            no_cache=True)
        if len(ses):
            one.alyx.rest('sessions', 'delete', ses[0]['url'][-36:])

        ses = one.alyx.rest('sessions', 'create', data=ses_dict)
        session_path = Path(self.td.name).joinpath(
            ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3))
        session_path.joinpath('alf').mkdir(exist_ok=True, parents=True)
        self.session_path = session_path
        self.eid = ses['url'][-36:]

    def tearDown(self) -> None:
        self.td.cleanup()
        one.alyx.rest('sessions', 'delete', id=self.eid)

    def test_pipeline_alyx(self):
        eid = self.eid
        pipeline = SomePipeline(self.session_path, one=one, eid=eid)

        # prepare by deleting all jobs/tasks related
        tasks = one.alyx.rest('tasks', 'list', session=eid, no_cache=True)
        assert(len(tasks) == 0)

        # create tasks and jobs from scratch
        NTASKS = len(pipeline.tasks)
        pipeline.make_graph(show=False)
        alyx_tasks = pipeline.create_alyx_tasks()
        self.assertTrue(len(alyx_tasks) == NTASKS)

        # get the pending jobs from alyx
        tasks = one.alyx.rest('tasks', 'list', session=eid, status='Waiting', no_cache=True)
        self.assertTrue(len(tasks) == NTASKS)

        # run them and make sure their statuses got updated appropriately
        task_deck, datasets = pipeline.run(machine='testmachine')
        check_statuses = [desired_statuses[t['name']] == t['status'] for t in task_deck]
        # [(t['name'], t['status'], desired_statuses[t['name']]) for t in task_deck]
        self.assertTrue(all(check_statuses))
        self.assertTrue(set([d['name'] for d in datasets]) == set(desired_datasets))

        # check logs
        check_logs = [desired_logs in t['log'] if t['log'] else True for t in task_deck]
        self.assertTrue(all(check_logs))

        # also checks that the datasets have been labeled with the proper version
        dsets = one.alyx.rest('datasets', 'list', session=eid, no_cache=True)
        check_versions = [desired_versions[d['name']] == d['version'] for d in dsets]
        self.assertTrue(all(check_versions))

        # make sure that re-running the make job by default doesn't change complete jobs
        pipeline.create_alyx_tasks()
        task_deck = one.alyx.rest('tasks', 'list', session=eid, no_cache=True)
        check_statuses = [desired_statuses[t['name']] == t['status'] for t in task_deck]
        self.assertTrue(all(check_statuses))

        # test the rerun option
        task_deck, dsets = pipeline.rerun_failed(machine='testmachine')
        task_02 = next(t for t in task_deck if t['name'] == 'Task02_error')
        self.assertEqual('Complete', task_02['status'])
        dep_task = next(x for x in task_deck if task_02['id'] in x['parents'])
        assert dep_task['name'] == 'Task11'
        self.assertEqual('Complete', dep_task['status'], 'Failed to set dependent task from "Held" to "Waiting"')

        # check that logs were correctly overwritten
        check_logs = [t['log'].count(desired_logs) == 1 if t['log'] else True for t in task_deck]
        check_rerun = ['===RERUN===' not in t['log'] if t['log'] else True for t in task_deck]
        self.assertTrue(all(check_logs))
        self.assertTrue(all(check_rerun))

        # Rerun without clobber and check that logs are not overwritten
        task_deck, dsets = pipeline.rerun_failed(machine='testmachine', clobber=False)
        check_logs = [t['log'].count(desired_logs) == desired_logs_rerun[t['name']] if t['log']
                      else t['log'] == desired_logs_rerun[t['name']] for t in task_deck]
        check_rerun = ['===RERUN===' in t['log'] if desired_logs_rerun[t['name']] == 2
                       else True for t in task_deck]
        self.assertTrue(all(check_logs))
        self.assertTrue(all(check_rerun))

        # Remove the lock file
        Path.home().joinpath('.one', 'gpu.lock').unlink()


class GpuTask(ibllib.pipes.tasks.Task):
    gpu = 1

    def _run(self, overwrite=False):
        out_files = self.session_path.joinpath('alf', 'gpu.times.npy')
        out_files.touch()
        return out_files


class TestLocks(unittest.TestCase):

    def test_gpu_lock_and_local_data_handler(self) -> None:
        # Remove any existing locks first
        if Path.home().joinpath('.one', 'gpu.lock').exists():
            Path.home().joinpath('.one', 'gpu.lock').unlink()
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td).joinpath('algernon', '2021/02/12', '001')
            session_path.joinpath('alf').mkdir(parents=True)
            task = GpuTask(session_path, one=None, location='local')
            assert task.is_locked() is False
            task.run()
            assert task.status == 0
            assert task.is_locked() is False
            # then make a lock file and make sure it fails and is still locked afterwards
            task._make_lock_file()
            task.run()
            assert task.status == - 2
            assert task.is_locked()
            # test the time out feature
            task.time_out_secs = - 1
            task._make_lock_file()
            assert not task.is_locked()
            task.run()
            assert task.status == 0


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

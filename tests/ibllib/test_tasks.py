import tempfile
import unittest
from pathlib import Path
from collections import OrderedDict

from ibllib.misc import version
import ibllib.pipes.tasks
from oneibl.one import ONE

one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
          username='test_user', password='TapetesBloc18')
SUBJECT_NAME = 'algernon'
USER_NAME = 'test_user'
# one = ONE(base_url='http://localhost:8000')
# SUBJECT_NAME = 'CSP014'
# USER_NAME = 'olivier'

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
    'Task11': 'Held'
}

desired_datasets = ['spikes.times.npy', 'spikes.amps.npy', 'spikes.clusters.npy']
desired_versions = {'spikes.times.npy': 'custom_job00',
                    'spikes.amps.npy': version.ibllib(),
                    'spikes.clusters.npy': version.ibllib()}


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


# job that raises an error
class Task02_error(ibllib.pipes.tasks.Task):
    level = 0

    def _run(self, overwrite=False):
        raise Exception("Something dumb happened")


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


class SomePipeline(ibllib.pipes.tasks.Pipeline):

    def __init__(self, session_path=None, **kwargs):
        super(SomePipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['Task00'] = Task00(self.session_path)
        tasks['Task01_void'] = Task01_void(self.session_path)
        tasks['Task02_error'] = Task02_error(self.session_path)
        tasks['Task10'] = Task10(self.session_path, parents=[tasks['Task00']])
        tasks['Task11'] = Task11(self.session_path, parents=[tasks['Task02_error'],
                                                             tasks['Task00']])
        self.tasks = tasks


class TestPipelineAlyx(unittest.TestCase):

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()

        ses = one.alyx.rest('sessions', 'list', subject=ses_dict['subject'],
                            date_range=[ses_dict['start_time'][:10]] * 2,
                            number=ses_dict['number'])
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
        pipeline = SomePipeline(self.session_path, one=one)

        # prepare by deleting all jobs/tasks related
        tasks = one.alyx.rest('tasks', 'list', session=eid)
        assert(len(tasks) == 0)

        # create tasks and jobs from scratch
        NTASKS = len(pipeline.tasks)
        pipeline.make_graph(show=False)
        alyx_tasks = pipeline.create_alyx_tasks()
        self.assertTrue(len(alyx_tasks) == NTASKS)

        # get the pending jobs from alyx
        tasks = one.alyx.rest('tasks', 'list', session=eid, status='Waiting')
        self.assertTrue(len(tasks) == NTASKS)

        # run them and make sure their statuses got updated appropriately
        task_deck, datasets = pipeline.run()
        check_statuses = [desired_statuses[t['name']] == t['status'] for t in task_deck]
        # [(t['name'], t['status'], desired_statuses[t['name']]) for t in task_deck]
        self.assertTrue(all(check_statuses))
        self.assertTrue(set([d['name'] for d in datasets]) == set(desired_datasets))

        # also checks that the datasets have been labeled with the proper version
        dsets = one.alyx.rest('datasets', 'list', session=eid)
        check_versions = [desired_versions[d['name']] == d['version'] for d in dsets]
        self.assertTrue(all(check_versions))

        # make sure that re-running the make job by default doesn't change complete jobs
        pipeline.create_alyx_tasks()
        task_deck = one.alyx.rest('tasks', 'list', session=eid)
        check_statuses = [desired_statuses[t['name']] == t['status'] for t in task_deck]
        self.assertTrue(all(check_statuses))

        # test the rerun option
        task_deck, dsets = pipeline.rerun_failed()
        check_statuses = [desired_statuses[t['name']] == t['status'] for t in task_deck]
        self.assertTrue(all(check_statuses))

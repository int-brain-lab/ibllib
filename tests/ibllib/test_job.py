import tempfile
import unittest

from pathlib import Path
import ibllib.pipes.jobs
from oneibl.one import ONE

one = ONE(base_url='https://testdev.alyx.internationalbrainlab.org',  # FIXME change the testdev
          username='test_user', password='TapetesBloc18')

ses_dict = {
    'subject': 'algernon',
    'start_time': '2018-04-01T12:48:26.795526',
    'number': 1,
    'users': ['test_user']
}


#  job to output a single file (pathlib.Path)
class Job00(ibllib.pipes.jobs.Job):

    def _run(self, overwrite=False):
        out_files = self.session_path.joinpath('alf', 'spikes.times.npy')
        out_files.touch()
        return out_files


#  job that outputs nothing
class Job01(ibllib.pipes.jobs.Job):

    def _run(self, overwrite=False):
        out_files = None
        return out_files


# job that outputs a list of files
class Job02(ibllib.pipes.jobs.Job):
    level = 1

    def _run(self, overwrite=False):
        out_files = [
            self.session_path.joinpath('alf', 'spikes.amps.npy'),
            self.session_path.joinpath('alf', 'spikes.clusters.npy')]
        [f.touch() for f in out_files]
        return out_files


# job that raises an error
class Job03(ibllib.pipes.jobs.Job):
    level = 2

    def _run(self, overwrite=False):
        raise Exception("Something dumb happened")


class SomePipeline(ibllib.pipes.jobs.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(SomePipeline, self).__init__(session_path, **kwargs)
        jobs = {}
        self.session_path = session_path
        # level 0
        jobs['Job00'] = Job00(self.session_path)
        jobs['Job01'] = Job01(self.session_path)
        jobs['Job02'] = Job02(self.session_path, parents=[jobs['Job01']])
        jobs['Job03'] = Job03(self.session_path, parents=[jobs['Job02'], jobs['Job01']])
        self.jobs = jobs


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
        jobs = one.alyx.rest('jobs', 'list', session=eid)
        tasks = list(set([j['task'] for j in jobs]))
        [one.alyx.rest('tasks', 'delete', id=task) for task in tasks]

        # create tasks and jobs from scratch
        NJOBS = len(pipeline.jobs)
        pipeline.make_graph(show=False)
        alyx_tasks = pipeline.init_alyx_tasks()
        self.assertTrue(len(alyx_tasks) == NJOBS)
        alyx_jobs = pipeline.register_alyx_jobs()
        self.assertTrue(len(alyx_jobs) == NJOBS)

        # get the pending jobs from alyx
        jobs = one.alyx.rest('jobs', 'list', session=eid, status='Waiting')
        self.assertTrue(len(jobs) == NJOBS)

        # run them and make sure their statuses got updated
        for jdict in jobs:
            status, dsets = ibllib.pipes.jobs.run_alyx_job(
                jdict=jdict, session_path=self.session_path, one=one)
        all_jobs = one.alyx.rest('jobs', 'list', session=eid)
        self.assertTrue(len(list(filter(lambda x: x['status'] != "Waiting", all_jobs))) == NJOBS)
        self.assertTrue(len(list(filter(lambda x: x['status'] == "Errored", all_jobs))) == 1)
        self.assertTrue(len(list(filter(lambda x: x['status'] == "Empty", all_jobs))) == 1)
        self.assertTrue(len(list(filter(lambda x: x['status'] == "Complete", all_jobs))) == 2)

        # make sure that re-running the make job by default doesn't change complete jobs
        pipeline.register_alyx_jobs()
        all_jobs = one.alyx.rest('jobs', 'list', session=eid)
        self.assertTrue(len(list(filter(lambda x: x['status'] != "Waiting", all_jobs))) == NJOBS)

        # test the rerun option
        pipeline.register_alyx_jobs(rerun=True)
        all_jobs = one.alyx.rest('jobs', 'list', session=eid)
        self.assertTrue(len(list(filter(lambda x: x['status'] == "Waiting", all_jobs))) == NJOBS)

        # make sure datasets have been registered
        self.assertTrue(len(one.alyx.rest('datasets', 'list', session=eid)) == 3)

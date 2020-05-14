# import unittest
#
# from ibllib.pipes import ephys_preprocessing
# from ibllib.pipes.jobs import run_alyx_job
# from oneibl.one import ONE
#
# session_path = "/datadisk/Data/IntegrationTests/ephys/choice_world_init/KS022/2019-12-10/001"
# one = ONE(base_url='http://localhost:8000')
#
#
# class TestEphysJobs(unittest.TestCase):
#
#     def test_EphysSyncPulses(self):
#         myjob = ephys_preprocessing.EphysPulses(session_path)
#         status = myjob.run(status=0)
#         self.assertTrue(len(myjob.outputs) == 9)
#         self.assertTrue(status == 0)
#
#     def test_EphysTrials(self):
#         myjob = ephys_preprocessing.EphysTrials(session_path)
#         status = myjob.run(status=0)
#         self.assertTrue(status == 0)
#
#
# class TestEphysPipeline(unittest.TestCase):
#
#     def test_pipeline_with_alyx(self):
#         eid = one.eid_from_path(session_path)
#
#         # prepare by deleting all jobs/tasks related
#         jobs = one.alyx.rest('jobs', 'list', session=eid)
#         tasks = list(set([j['task'] for j in jobs]))
#         [one.alyx.rest('tasks', 'delete', id=task) for task in tasks]
#
#         # create jobs from scratch
#         NJOBS = 2
#         ephys_pipe = ephys_preprocessing.EphysExtractionPipeline(session_path, one=one)
#         ephys_pipe.make_graph(show=False)
#         alyx_tasks = ephys_pipe.init_alyx_tasks()
#         self.assertTrue(len(alyx_tasks) == NJOBS)
#
#         alyx_jobs = ephys_pipe.register_alyx_jobs()
#         self.assertTrue(len(alyx_jobs) == NJOBS)
#
#         # get the pending jobs from alyx
#         jobs = one.alyx.rest('jobs', 'list', session=eid, status='Waiting')
#         self.assertTrue(len(jobs) == NJOBS)
#
#         # run them and make sure their statuses got updated
#         for jdict in jobs:
#             run_alyx_job(jdict=jdict, session_path=session_path, one=one)
#         jobs = one.alyx.rest('jobs', 'list', session=eid, status='Waiting')
#         jobs_done = one.alyx.rest('jobs', 'list', session=eid, status='Complete')
#         self.assertTrue(len(jobs_done) == NJOBS)
#         self.assertTrue(len(jobs) == 0)
#
#         # make sure that re-running the make job by default doesn't change complete jobs
#         alyx_jobs = ephys_pipe.register_alyx_jobs()
#         self.assertTrue(len(one.alyx.rest('jobs', 'list',
#                                           session=eid, status='Waiting')) == 0)
#         self.assertTrue(len(one.alyx.rest('jobs', 'list',
#                                           session=eid, status='Complete')) == NJOBS)
#
#         # test the rerun option
#         alyx_jobs = ephys_pipe.register_alyx_jobs(rerun=True)
#         self.assertTrue(len(one.alyx.rest('jobs', 'list',
#                                           session=eid, status='Waiting')) == NJOBS)
#         self.assertTrue(len(one.alyx.rest('jobs', 'list',
#                                           session=eid, status='Complete')) == 0)

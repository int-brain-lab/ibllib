import unittest
import ibllib.pipes.jobs as jobs

session_path = "/datadisk/Data/IntegrationTests/ephys/choice_world_init/KS022/2019-12-10/001"


class TestEphysPipeline(unittest.TestCase):

    def test_EphysSyncPulses(self):
        myjob = jobs.EphysPulses(session_path)
        status = myjob.run(status=0)
        self.assertTrue(len(myjob.outputs) == 9)
        self.assertTrue(status == 0)

    def test_EphysTrials(self):
        myjob = jobs.EphysTrials(session_path)
        status = myjob.run(status=0)
        self.assertTrue(status == 0)

import unittest

from ibllib.qc.oneqc_metrics import ONEQC
from oneibl.one import ONE


one = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
          password='TapetesBloc18')


class TestONEQCMetrics(unittest.TestCase):
    """
    Testing of one dstypes will require test sessions with one data in the test db
    """
    def setUp(self):
        # An ephys session on the test DB
        # Subj from testDB = clns0730
        self.one = one
        self.test_eid = 'cf264653-2deb-44cb-aa84-89b82507028a'

    def test_ONEQC_lazy(self):
        oneqc = ONEQC(self.test_eid, one=self.one, bpod_ntrials=None, lazy=True)
        self.assertTrue(oneqc.bpod_ntrials == 616)
        self.assertTrue(oneqc.metrics is None)
        self.assertTrue(oneqc.passed is None)
        oneqc.compute()
        self.assertTrue(oneqc.metrics is not None)
        self.assertTrue(oneqc.passed is not None)

    def test_ONEQC(self):
        oneqc = ONEQC(self.test_eid, one=self.one, bpod_ntrials=None, lazy=False)
        self.assertTrue(oneqc.metrics is not None)
        self.assertTrue(oneqc.passed is not None)


if __name__ == "__main__":
    unittest.main(exit=False)

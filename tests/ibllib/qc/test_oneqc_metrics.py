# Mock dataset
import unittest

from ibllib.qc import oneqc_metrics as oneqc
from oneibl.one import ONE

one = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
          password='TapetesBloc18')


class TestONEQCMetrics(unittest.TestCase):
    """
    Testing of one dstypes will require test sessions with one data in the test db
    XXX: NEED METHOD TO FORCE MODULES TO USE TEST ONE!!!!
    XXX: NEED METHOD TO FORCE MODULES TO USE TEST ONE!!!!
    XXX: NEED METHOD TO FORCE MODULES TO USE TEST ONE!!!!
    XXX: NEED METHOD TO FORCE MODULES TO USE TEST ONE!!!!
    XXX: NEED METHOD TO FORCE MODULES TO USE TEST ONE!!!!
    XXX: NEED METHOD TO FORCE MODULES TO USE TEST ONE!!!!
    """
    def setUp(self):
        # An ephys session on the test DB
        self.eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"
        self.eid_testdb = "0e3f67af-a0f5-404b-8617-aa08ab7f0dc3"

    def test_build_extended_qc_frame(self):
        metrics = oneqc.get_oneqc_metrics_frame(self.eid, apply_criteria=False)
        criteria = oneqc.get_oneqc_metrics_frame(self.eid, apply_criteria=True)
        self.assertTrue(metrics == criteria)


one.alyx.rest('sessions', 'list', project='ibl_neuropixel_brainwide_01', extended_qc='_bpod_stimOff_itiIn_delays__lt,0.99')

if __name__ == "__main__":
    unittest.main(exit=False)

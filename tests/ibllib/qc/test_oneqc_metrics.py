# Mock dataset
import unittest

from ibllib.qc import oneqc_metrics as oneqc
from oneibl.one import ONE


one = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
          password='TapetesBloc18')

oneqc.one = one


class TestONEQCMetrics(unittest.TestCase):
    """
    Testing of one dstypes will require test sessions with one data in the test db
    """
    def setUp(self):
        # An ephys session on the test DB
        # Subj from testDB = clns0730
        self.test_eid = 'cf264653-2deb-44cb-aa84-89b82507028a'

    def test_build_extended_qc_frame(self):
        # Metrics and criteria ar equal as test will hit first if statement
        # and jsut return the frame with None as all values
        # No datasetTypes are found so metrics == criteria
        metrics = oneqc.get_oneqc_metrics_frame(self.test_eid, apply_criteria=False)
        criteria = oneqc.get_oneqc_metrics_frame(self.test_eid, apply_criteria=True)
        self.assertTrue(metrics == criteria)


if __name__ == "__main__":
    unittest.main(exit=False)

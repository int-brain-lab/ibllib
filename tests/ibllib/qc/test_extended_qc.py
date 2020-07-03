# Mock dataset
import unittest

from ibllib.qc import ExtendedQC
from oneibl.one import ONE

one = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
          password='TapetesBloc18')


class TestExtendedQC(unittest.TestCase):
    """
    """
    def setUp(self):
        self.one = one
        # An ephys session on the test DB
        self.eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"
        self.eid2 = "cf264653-2deb-44cb-aa84-89b82507028a"
        self.eqc_dict = {'some': 0, 'dict': 1}

    def test_ExtendedQC_lazy(self):
        eqc = ExtendedQC(one=self.one, eid=self.eid, lazy=True)
        self.assertTrue(eqc.read_extended_qc() is None)
        self.assertTrue(eqc.update_extended_qc() is None)
        self.assertTrue(eqc.frame is None)
        # FIXME: no session that passes both ONE QC and Bpod QC exist in test DB
        # eqc.compute_all_qc()
        # self.assertTrue(eqc.bpodqc is not None)
        # self.assertTrue(eqc.oneqc is not None)
        # eqc.build_extended_qc_frame()
        # self.assertTrue(isinstance(eqc.frame, dict))

    def test_compute_session_status(self):
        frame = {
            "test1": 1,
            "test2": 0.5,
            "test3": 0.99
        }
        # Should return CRITICAL

        # TODO instanciate class necessary for testing small frame?
        # eqc = ExtendedQC(one=self.one, eid=self.eid, lazy=True)
        # compute_session_status(frame)


if __name__ == "__main__":
    unittest.main(exit=False)

# Mock dataset
import unittest

from ibllib.qc import extended_qc as eqc

one = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
          password='TapetesBloc18')
class TestExtendedQC(unittest.TestCase):
    """
    """
    def setUp(self):
        self.eqc_dict = {'some': 0, 'dict': 1}
        # An ephys session on the test DB
        self.eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"

    def test_build_extended_qc_frame(self):
        pass


one.alyx.rest('sessions', 'list', project='ibl_neuropixel_brainwide_01', extended_qc='_bpod_stimOff_itiIn_delays__lt,0.99')

if __name__ == "__main__":
    unittest.main(exit=False)

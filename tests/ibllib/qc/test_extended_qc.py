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

    def test_read(self):
        one.alyx.json_field_write(endpoint, uuid, filed_name, data)
        one.alyx.json_field_read(endpoint, uuid, filed_name)
        one.alyx.json_field_update(endpoint, uuid, filed_name, data)
        one.alyx.json_field_remove_key(endpoint, uuid, filed_name, key)
        one.alyx.json_field_delete(endpoint, uuid, filed_name, data)

    test_json_query
        create one session
        add same dict to both sessions eqc field -> tests write
        test query returns both sessions -> tests generic query
        update one value of one sess -> tests update method
        test query returns 1 -> tests query __ge pattern
        query on different key that returns 2
        delete this key from one session -> tests remove key method
        same query should return one
        delete one of the dicts in the field -> tests delete


one.alyx.rest('sessions', 'list', project='ibl_neuropixel_brainwide_01', extended_qc='_bpod_stimOff_itiIn_delays__lt,0.99')

if __name__ == "__main__":
    unittest.main(exit=False)

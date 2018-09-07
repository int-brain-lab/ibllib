import unittest
import numpy as np
from oneibl.one import ONE, SessionDataInfo


class TestLoad(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        myone = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
                    password='TapetesBloc18')
        self.One = myone

    def test_list(self):
        myone = self.One
        eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
        a = myone.list(eid)
        self.assertTrue(len(a) == 29)
        self.assertTrue(isinstance(a, list))

    def test_load(self):
        # Test with 3 actual datasets predefined
        myone = self.One
        dataset_types = ['clusters.peakChannel', 'clusters._phy_annotation', 'clusters.probes']
        eid = ('https://test.alyx.internationalbrainlab.org/'
               'sessions/86e27228-8708-48d8-96ed-9aa61ab951db')
        t, cr, cl = myone.load(eid, dataset_types=dataset_types)
        d = myone.load(eid, dataset_types=dataset_types, dclass_output=True)
        self.assertTrue(np.all(d.data[0] == t))

    def test_load_empty(self):
        # Test with a session that doesn't have any dataset on the Flat Iron
        myone = self.One
        eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
        dataset_types = ['wheel.velocity', 'wheel.timestamps']
        a, b = myone.load(eid, dataset_types=dataset_types)
        self.assertTrue(a is None and b is None)

    def test_load_from_uuid(self):
        # Test the query with only the UUID string and not the full URL (no data here)
        myone = self.One
        eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
        dataset_types = ['wheel.velocity', 'wheel.timestamps']
        aa = myone.load(eid, dataset_types=dataset_types)
        self.assertTrue(len(aa) == 2)

    def test_load_str(self):
        myone = self.One
        eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
        a = myone.load(eid, 'eye.raw')
        self.assertTrue(len(a) == 1)

    def test_load_all_data_available(self):
        # Test without a dataset list should download everything and output a dictionary
        myone = self.One
        eid = '3bca3bef-4173-49a3-85d7-596d62f0ae16'
        a = myone.load(eid)
        self.assertTrue(len(a.data) == 5)

    def test_ls(self):
        # Test when the dataset type requested is not unique
        myone = self.One
        # test users
        [l1, f] = myone.ls(table=['users'])
        l2, f2 = myone.ls_users()
        self.assertTrue(isinstance(l1[0], str) and isinstance(l1[0], str))
        self.assertEqual(l1,l2)
        # test subjects
        [l1, f] = myone.ls(table='subjects')
        l2, f2 = myone.ls_subjects()
        self.assertTrue(isinstance(l1[0], str) and isinstance(l1[0], str))
        self.assertEqual(l1,l2)
        # test datasets
        [l1, f] = myone.ls(table='dataset-types')
        l2, f2 = myone.ls_dataset_types()
        self.assertTrue(isinstance(l1[0], str) and isinstance(l1[0], str))
        self.assertEqual(l1,l2)
        # test with 2 tables, string format
        [l, f] = myone.ls(table=['users', 'dataset-types'])
        self.assertTrue(isinstance(l[0], list) and len(l) == 2)

    def test_search_simple(self):
        myone = self.One
        # Test users
        usr = ['olivier', 'nbonacchi']
        sl, sd = myone.search(users=usr)
        self.assertTrue(isinstance(sl, list) and isinstance(sd, list))
        self.assertTrue(all([set(usr).issubset(set(u)) for u in [s['users'] for s in sd]]))
        # when the user is a string instead of a list
        sl1, sd1 = myone.search(users=['olivier'])
        sl2, sd2 = myone.search(users='olivier')
        self.assertTrue(sl1 == sl2 and sd1 == sd2)
        # test for the dataset type
        dtyp = ['spikes.times', 'titi.tata']
        sl, sd = myone.search(dataset_types=dtyp)
        self.assertTrue(len(sl) == 0)
        dtyp = ['channels.site']
        sl, sd = myone.search(dataset_types=dtyp)
        self.assertTrue(len(sl) == 2)
        dtyp = ['spikes.times', 'channels.site']
        sl, sd = myone.search(dataset_types=dtyp)
        self.assertTrue(len(sl) == 1)

    def test_session_data_info(self):
        myone = self.One
        eid = '3bca3bef-4173-49a3-85d7-596d62f0ae16'
        dinfo = myone.session_data_info(eid)
        self.assertTrue(isinstance(dinfo, SessionDataInfo))
        print(dinfo)  # tests the __str__method of the dataclass

    def test_session_does_not_exist(self):
        eid = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
        self.assertRaises(FileNotFoundError, self.One.load, eid)


if __name__ == '__main__':
    unittest.main()

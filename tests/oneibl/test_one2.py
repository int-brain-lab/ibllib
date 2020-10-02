import unittest
from pathlib import Path
from uuid import UUID

import numpy as np

from oneibl.one2 import ONE2
from alf.io import AlfBunch
from ibllib.exceptions import ALFObjectNotFound, ALFMultipleObjectsFound
from ibllib.misc.exp_ref import eid2ref


class TestONE2(unittest.TestCase):
    def setUp(self):
        # self.one = ONE2(
        #     base_url="https://test.alyx.internationalbrainlab.org",
        #     username="test_user",
        #     password="TapetesBloc18",
        # )
        self.one = ONE2()

    def test_load_object(self):
        eid = 'f3ce3197-d534-4618-bf81-b687555d1883'  # Spikes
        eid2 = '2d768cde-65d4-4374-af2e-6ff3bf606eb4'  # Trials

        # Test loading simple object
        obj = self.one.load_object(eid2, 'wheel')
        self.assertIsInstance(obj, AlfBunch)
        self.assertCountEqual(['position', 'timestamps'], obj.keys())

        # Test loading object with wildcard
        obj = self.one.load_object(eid2, '*ials')
        expected = 15
        self.assertEqual(expected, len(obj))
        self.assertTrue('intervals' in obj)

        # Test download only flag
        files = self.one.load_object(eid2, 'trials', download_only=True)
        self.assertEqual(expected, len(files))
        self.assertTrue(all(isinstance(f, Path) for f in files))

        # Test loading with collection
        obj = self.one.load_object(eid, 'spikes', collection='alf/probe01')
        self.assertEqual(6, len(obj))
        self.assertTrue('depths' in obj)

        # Test exceptions: multiple objects
        with self.assertRaises(ALFMultipleObjectsFound):
            self.one.load_object(eid2, 'wh*')

        # Test exception: multiple collections
        with self.assertRaises(AssertionError):
            self.one.load_object(eid, 'spikes', 'all')

        # Test exception: object not found
        with self.assertRaises(ALFObjectNotFound):
            self.one.load_object(eid2, 'spikes', collection='alf/probe01')

        # Test with decorator
        obj = self.one.load_object(UUID(eid2), 'wheel')
        self.assertIsInstance(obj, AlfBunch)

    def test_load_dataset(self):
        eid = 'f3ce3197-d534-4618-bf81-b687555d1883'  # Spikes
        eid2 = '2d768cde-65d4-4374-af2e-6ff3bf606eb4'  # Trials

        # Test loading simple object
        ds = self.one.load_dataset(eid2, '_ibl_wheel.position.npy')
        expected = np.array([0.00153398, -0., 0.00153398])
        np.testing.assert_array_almost_equal(expected, ds[:3])

        # Test loading dataset with wildcard
        ds = self.one.load_dataset(eid2, '*wheel.position.npy')
        np.testing.assert_array_almost_equal(expected, ds[:3])

        # Test download only flag
        filename = self.one.load_dataset(eid2, '*wheel.position.npy', download_only=True)
        self.assertIsInstance(filename, Path)
        self.assertTrue(str(filename).endswith('wheel.position.npy'))

        # Test loading with collection
        ds = self.one.load_dataset(eid, 'spikes.times.npy', collection='alf/probe01')
        expected = np.array([0.00275089, 0.00331756, 0.00341756])
        np.testing.assert_array_almost_equal(expected, ds[:3])

        # Test exceptions: multiple datasets
        with self.assertRaises(ALFMultipleObjectsFound):
            self.one.load_dataset(eid2, '_ibl_*')

        # Test exception: multiple collections
        with self.assertRaises(AssertionError):
            self.one.load_dataset(eid, 'spikes.times.npy', 'all')

        # Test exception: dataset not found
        with self.assertRaises(ALFObjectNotFound):
            self.one.load_dataset(eid2, 'spikes.times.npy')

    def test_to_eid(self):
        eid = 'f3ce3197-d534-4618-bf81-b687555d1883'
        url = 'https://alyx.internationalbrainlab.org/sessions/' + eid
        uuid = UUID(eid)
        ref = eid2ref(eid, one=self.one)
        ref_str = eid2ref(eid, one=self.one, as_dict=False)
        path = self.one.path_from_eid(eid)

        for id in (eid, url, uuid, ref, path, ref_str):
            self.assertEqual(self.one.to_eid(id), eid)

        with self.assertRaises(ValueError):
            self.one.to_eid('e73hj')
            self.one.to_eid({'subject': 'flowers'})

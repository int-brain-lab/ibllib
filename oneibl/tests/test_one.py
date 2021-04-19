import unittest
from unittest import mock
from uuid import UUID
import requests
from pathlib import Path
import tempfile
import shutil

import numpy as np

import ibllib.io.hashfile as hashfile
from ibllib.exceptions import ALFObjectNotFound
from alf.io import remove_uuid_file
import oneibl.params as params
from oneibl.one import ONE


one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
          username='test_user',
          password='TapetesBloc18')


class TestOneSetup(unittest.TestCase):

    def setUp(self) -> None:
        self.pars_file = Path.home().joinpath('.fake_pars', '.oneibl')

    def tearDown(self) -> None:
        self.pars_file.unlink(missing_ok=True)
        self.pars_file.parent.rmdir()

    def test_setup_silent(self):
        # Mock getfile function to return a path to non-existent file instead of usual one pars
        with mock.patch('oneibl.params.iopar.getfile') as mock_getfile:
            mock_getfile.return_value = str(self.pars_file)
            one = ONE(offline=True, silent=True)
        self.assertCountEqual(one._par.as_dict(), params.default().as_dict())
        self.assertTrue(self.pars_file.exists())

    def test_setup(self):
        params.input = lambda prompt: 'mock_input'
        params.getpass = lambda prompt: 'mock_pwd'
        params.print = lambda text: 'mock_print'
        # Mock getfile function to return a path to non-existent file instead of usual one pars
        with mock.patch('oneibl.params.iopar.getfile') as mock_getfile:
            mock_getfile.return_value = str(self.pars_file)
            one = ONE(offline=True, silent=False)
        self.assertEqual(one._par.ALYX_PWD, 'mock_pwd')
        self.assertTrue(self.pars_file.exists())


class TestSearch(unittest.TestCase):

    def test_search_simple(self):
        # Test users
        usr = ['olivier', 'nbonacchi']
        sl, sd = one.search(users=usr, details=True)
        self.assertTrue(isinstance(sl, list) and isinstance(sd, list))
        self.assertTrue(all([set(usr).issubset(set(u)) for u in [s['users'] for s in sd]]))
        # when the user is a string instead of a list
        sl1, sd1 = one.search(users=['olivier'], details=True)
        sl2, sd2 = one.search(users='olivier', details=True)
        self.assertTrue(sl1 == sl2 and sd1 == sd2)
        ses = one.search(users='olivier', number=6)
        self.assertTrue(not ses)
        # test for the dataset type
        dtyp = ['spikes.times', 'titi.tata']
        sl, sd = one.search(dataset_types=dtyp, details=True)
        self.assertTrue(len(sl) == 0)
        dtyp = ['channels.site']
        sl, sd = one.search(dataset_types=dtyp, details=True)
        self.assertTrue(len(sl) == 2)
        dtyp = ['spikes.times', 'channels.site']
        sl, sd = one.search(dataset_types=dtyp, details=True)
        self.assertTrue(len(sl) >= 1)
        # test empty return for non-existent user
        self.assertTrue(len(one.search(users='asdfa')) == 0)
        # test search with the lab keyword
        self.assertTrue(len(one.search(lab='zadorlab')) == 1)


class TestList(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        eids = ['cf264653-2deb-44cb-aa84-89b82507028a', '4e0b3320-47b7-416e-b842-c34dc9004cf8']
        self.eid = eids[0]
        self.eid2 = eids[1]

    def test_list(self):
        # tests with a single input and a list input.
        # One of the datasets has its exists flag set to False; it should be excluded from the list
        eid = self.eid
        dt = one.list(eid)  # returns dataset-type
        self.assertTrue(isinstance(dt, list))
        self.assertFalse(any(str(x) == 'channels.rawRow.npy' for x in dt))
        self.assertEqual(28, len(dt))

        dt = one.list(eid, details=True)  # returns dict of dataset-types
        self.assertTrue(isinstance(dt[0], dict))


class TestLoad(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        eids = ['cf264653-2deb-44cb-aa84-89b82507028a', '4e0b3320-47b7-416e-b842-c34dc9004cf8']
        self.eid = eids[0]
        self.eid2 = eids[1]

    def test_load_multiple_sessions(self):
        # init stuff to run from cli
        eids = [self.eid, self.eid2]

        # 2 sessions, data exists on both, dclass output
        out = one.load(eids, dataset_types='channels.site', dclass_output=True)
        self.assertTrue(len(out.data) == 2)
        self.assertTrue(all([len(o) == 748 for o in out.data]))
        # same with a list output
        out = one.load(eids, dataset_types='channels.site')
        # we have 2 sessions, thus 2 elements in the list and they're all 748 elements arrays
        self.assertTrue(len(out) == 2)
        self.assertTrue(all([len(o) == 748 for o in out]))

        # here the dataset type only exists for the first session
        out = one.load(eids, dataset_types='licks.times')
        self.assertTrue(len(out) == 2)
        self.assertTrue(len(out[0]) == 5126 and out[1] is None)
        # test the order of output
        out = one.load([eids[-1], eids[0]], dataset_types='licks.times')
        self.assertTrue(len(out) == 2)
        self.assertTrue(len(out[1]) == 5126 and out[0] is None)
        # same with dataclass output
        out = one.load(eids, dataset_types='licks.times', dclass_output=True)
        self.assertTrue(len(out.data) == 2)
        self.assertTrue(len(out.data[0]) == 5126 and out.data[1] is None)
        # now with reversed order and dataclass output
        out = one.load([eids[-1], eids[0]], dataset_types='licks.times', dclass_output=True)
        self.assertTrue(len(out.data) == 2)
        self.assertTrue(len(out.data[1]) == 5126 and out.data[0] is None)

    def test_load_uuid(self):
        dataset_types = ['eye.blink']
        eid = ('https://test.alyx.internationalbrainlab.org/'
               'sessions/' + self.eid)
        filename = one.load(eid, dataset_types=dataset_types, download_only=True, keep_uuid=True)
        uuid_fn = filename[0]
        filename = one.load(eid, dataset_types=dataset_types, download_only=True)
        self.assertTrue(filename[0] == remove_uuid_file(uuid_fn))
        self.assertFalse(Path(uuid_fn).exists())
        one.load(eid, dataset_types=dataset_types, download_only=True, keep_uuid=True)
        self.assertTrue(Path(uuid_fn).exists())

    def test_load(self):
        # Test with 3 actual datasets predefined
        dataset_types = ['clusters.channels', 'clusters._phy_annotation', 'clusters.probes']
        eid = ('https://test.alyx.internationalbrainlab.org/'
               'sessions/' + self.eid)
        t, cr, cl = one.load(eid, dataset_types=dataset_types)
        d = one.load(eid, dataset_types=dataset_types, dclass_output=True)
        ind = int(np.where(np.array(d.dataset_type) == 'clusters.channels')[0])
        self.assertTrue(np.all(d.data[ind] == t))
        # Now load with another dset in between that doesn't exist
        t_, cr_, cl_ = one.load(eid, dataset_types=['clusters.channels', 'turlu',
                                                    'clusters.probes'])
        self.assertTrue(np.all(t == t_))
        self.assertTrue(np.all(cl == cl_))
        self.assertTrue(cr_ is None)
        # Now try in offline mode where the file already exists
        t_ = one.load(eid, dataset_types=['clusters.channels'])
        self.assertTrue(np.all(t == t_))

    def test_load_empty(self):
        # Test with a session that doesn't have any dataset on the Flat Iron
        eid = self.eid
        dataset_types = ['wheel.velocity', 'passiveTrials.included']
        a, b = one.load(eid, dataset_types=dataset_types)
        self.assertTrue(a is None and b is None)

    def test_load_from_uuid(self):
        # Test the query with only the UUID string and not the full URL (no data here)
        eid = self.eid
        dataset_types = ['wheel.velocity', 'wheel.timestamps']
        aa = one.load(eid, dataset_types=dataset_types)
        self.assertTrue(len(aa) == 2)

    def test_load_all_data_available(self):
        # Test without a dataset list should download everything and output a dictionary
        eid = self.eid2
        a = one.load(eid, dataset_types='__all__')
        self.assertTrue(len(a.data) >= 5)

    def test_load_fileformats(self):
        # npy already works for other tests around, tsv and csv implemented so far
        eid = self.eid
        one.load(eid, dataset_types=['probes.description'])

    def test_session_does_not_exist(self):
        eid = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
        self.assertRaises(requests.HTTPError, one.load, eid)

    def test_download_hash(self):
        eid = self.eid
        # get the original file from the server
        file = one.load(eid, dataset_types=['channels.localCoordinates'], download_only=True,
                        clobber=True)[0]
        fsize = file.stat().st_size
        hash = hashfile.md5(file)
        data_server = np.load(file)
        # overwrite the local file
        np.save(file, np.zeros([25, 0]))
        # here we patch the dataset with the server filesize and hash
        dset = one.alyx.rest('datasets', 'list',
                             dataset_type='channels.localCoordinates', session=eid)
        one.alyx.rest('datasets', 'partial_update', id=dset[0]['url'][-36:],
                      data={'file_size': fsize, 'hash': hash})
        data = one.load(eid, dataset_types=['channels.localCoordinates'])[0]
        self.assertTrue(data.shape == data_server.shape)
        # Verify new hash / filesizes added to cache table
        rec, = one._make_dataclass_offline(eid, dataset_types='channels.localCoordinates')
        self.assertEqual(rec.file_size, fsize)
        self.assertEqual(rec.hash, hash)
        # here we patch a dataset and make sure it overwrites if the checksum is different
        np.save(file, data_server * 2)
        data = one.load(eid, dataset_types=['channels.localCoordinates'])[0]
        self.assertTrue(data.shape == data_server.shape)
        self.assertTrue(np.all(np.equal(data, data_server)))
        # here we corrupt the md5 hash on the database, the file will get downloaded again,
        # but on checking the file one.load should have labeled the json field for database
        # maintenance
        one.alyx.rest('datasets', 'partial_update', id=dset[0]['url'][-36:],
                      data={'file_size': fsize, 'hash': "5d1d13589934440a9947c2477b2e61ea"})
        one.load(eid, dataset_types=['channels.localCoordinates'])[0]
        fr = one.alyx.rest('files', 'list', django=f"dataset,{dset[0]['url'][-36:]},"
                                                   f"data_repository__globus_is_personal,False")
        self.assertTrue(fr[0]['json'] == {'mismatch_hash': True})

    def test_load_object(self):
        # Test download_only flag
        files = one.load_object(self.eid, 'channels', collection=None, download_only=True)
        self.assertTrue(len(files) == 4)
        self.assertIsInstance(files[0], Path)

        # Test loading
        obj = one.load_object(self.eid, 'channels', collection=None)
        # One of the datasets has its exists flag set to False; it should be excluded from the list
        expected = ['brainLocation', 'probe', 'site', 'sitePositions']  # 'rawRow' missing
        self.assertCountEqual(obj.keys(), expected)

        with self.assertRaises(ALFObjectNotFound):
            one.load_object(self.eid, 'channels', collection='alf')

        with self.assertRaises(ValueError):
            one.load_object('fake', 'channels', collection='alf')


class TestMisc(unittest.TestCase):

    def test_validate_date_range(self):
        from oneibl.one import _validate_date_range
        # test with list of strings
        expval = ['2018-11-04', '2018-11-04']
        val = ['2018-11-04',
               ['2018-11-04'],
               ('2018-11-04'),
               ['2018-11-04', '2018-11-04'],
               ]
        for v in val:
            self.assertEqual(_validate_date_range(v), expval)
        val = ('2018-11-04', '2018-11-04')
        self.assertEqual(_validate_date_range(val), val)

    def test_to_eid(self):
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        url = 'https://test.alyx.internationalbrainlab.org/sessions/' + eid
        uuid = UUID(eid)
        # ref = eid2ref(eid, one=one)
        # ref_str = eid2ref(eid, one=one, as_dict=False)
        path = one.path_from_eid(eid)

        for id in (eid, url, uuid, path):
            self.assertEqual(one.to_eid(id), eid)

        # Test list
        self.assertEqual(one.to_eid([id, url]), [eid, eid])

        with self.assertRaises(ValueError):
            one.to_eid('e73hj')
            one.to_eid({'subject': 'flowers'})

    def test_path_to_url(self):
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        session_path = one.path_from_eid(eid)
        # Test URL is returned
        filepath = session_path.joinpath('alf', '_ibl_wheel.position.npy')
        url = one.url_from_path(filepath)
        expected = ('mainenlab/Subjects/clns0730/2018-08-24/1/'
                    '_ibl_wheel.position.a0155492-ee9d-4584-ba4e-7c86f9b12d3a.npy')
        self.assertIn(expected, url)
        # Test errors raised
        with self.assertRaises(ALFObjectNotFound):
            one.url_from_path(
                session_path.joinpath('raw_video_data', '_iblrig_leftCamera.raw.mp4'))

    def test_datasets_from_type(self):
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        dsets = one.datasets_from_type(eid, 'eye.blink')
        self.assertCountEqual(dsets, ['eye.blink.npy'])
        dsets, = one.datasets_from_type(eid, 'eye.blink', full=True)
        self.assertIsInstance(dsets, dict)


class TestOneOffline(unittest.TestCase):

    def setUp(self) -> None:
        # init: create a temp directory and copy the fixtures
        init_cache_file = Path(__file__).parent.joinpath('fixtures', '.one_cache.parquet')

        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()

        cache_dir = Path(self.test_dir.name)
        shutil.copyfile(init_cache_file, cache_dir.joinpath(init_cache_file.name))

        # test the constructor
        self.one = ONE(offline=True)
        self.assertTrue(self.one._cache.shape[1] == 14)

        self.eid = 'cf264653-2deb-44cb-aa84-89b82507028a'

    def test_one_offline(self) -> None:
        # test the load with download false so it returns only file paths
        one.list(self.eid)
        dtypes = ['_spikeglx_sync.channels',
                  '_spikeglx_sync.polarities',
                  '_spikeglx_sync.times',
                  '_iblrig_taskData.raw',
                  '_iblrig_taskSettings.raw',
                  'ephysData.raw.meta',
                  'camera.times',
                  'ephysData.raw.wiring']
        one.load(self.eid, dataset_types=dtypes, dclass_output=False,
                 download_only=True, offline=False)

    def test_path_eid(self):
        """Test `path_from_eid` and `eid_from_path` methods"""
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        # path from eid
        session_path = self.one.path_from_eid(eid)
        self.assertEqual(session_path.parts[-3:], ('clns0730', '2018-08-24', '002'))
        # eid from path
        self.assertEqual(eid, one.eid_from_path(session_path))

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

import unittest
from unittest.mock import patch
import os
import uuid
import tempfile
from pathlib import Path
import sys

import numpy as np
from one.api import ONE
from iblutil.io import params

from ibllib.tests import TEST_DB
from ibllib.io import flags, misc, globus, video
import ibllib.io.raw_data_loaders as raw


class TestsParams(unittest.TestCase):

    def setUp(self):
        self.par_dict = {'A': 'tata',
                         'O': 'toto',
                         'I': 'titi',
                         'num': 15,
                         'liste': [1, 'turlu'],
                         'apath': Path('/gna/gna/gna')}
        params.write('toto', self.par_dict)
        params.write('toto', params.from_dict(self.par_dict))

    def test_params(self):
        #  first go to and from dictionary
        par_dict = self.par_dict
        par = params.from_dict(par_dict)
        self.assertEqual(params.as_dict(par), par_dict)
        # next go to and from dictionary via json
        par2 = params.read('toto')
        self.assertEqual(par, par2)

    def test_param_get_file(self):
        home_dir = Path(params.getfile("toto")).parent
        # straight case the file is .{str} in the home directory
        assert home_dir.joinpath(".toto") == Path(params.getfile("toto"))
        # straight case the file is .{str} in the home directory
        assert home_dir.joinpath(".toto") == Path(params.getfile(".toto"))
        # subfolder case
        assert home_dir.joinpath(".toto", ".titi") == Path(params.getfile("toto/titi"))

    def test_new_default_param(self):
        # in this case an updated version of the codes brings in a new parameter
        default = {'A': 'tata2',
                   'O': 'toto2',
                   'I': 'titi2',
                   'E': 'tete2',
                   'num': 15,
                   'liste': [1, 'turlu']}
        expected_result = {'A': 'tata',
                           'O': 'toto',
                           'I': 'titi',
                           'num': 15,
                           'liste': [1, 'turlu'],
                           'apath': str(Path('/gna/gna/gna')),
                           'E': 'tete2',
                           }
        par2 = params.read('toto', default=default)
        self.assertCountEqual(par2.as_dict(), expected_result)
        # on the next path the parameter has been added to the param file
        par2 = params.read('toto', default=default)
        self.assertCountEqual(par2.as_dict(), expected_result)
        # check that it doesn't break if a named tuple is given instead of a dict
        par3 = params.read('toto', default=par2)
        self.assertEqual(par2, par3)
        # check that a non-existing parfile raises error
        pstring = str(uuid.uuid4())
        with self.assertRaises(FileNotFoundError):
            params.read(pstring)
        # check that a non-existing parfile with default returns default
        par = params.read(pstring, default=default)
        self.assertCountEqual(par, params.from_dict(default))
        # even if this default is a Params named tuple
        par = params.read(pstring, default=par)
        self.assertEqual(par, params.from_dict(default))
        # check default empty dict
        pstring = 'foobar'
        filename = Path(params.getfile(pstring))
        self.assertFalse(filename.exists())
        par = params.read(pstring, default={})
        self.assertIsNone(par)
        self.assertTrue(filename.exists())

    def tearDown(self):
        # at last delete the param file
        Path(params.getfile('toto')).unlink(missing_ok=True)
        Path(params.getfile('foobar')).unlink(missing_ok=True)


class TestsRawDataLoaders(unittest.TestCase):

    def setUp(self):
        self.tempfile = tempfile.NamedTemporaryFile(delete=False)
        self.bin_session_path = Path(__file__).parent.joinpath(
            'fixtures', 'io', 'data_loaders', "_iblrig_test_mouse_2020-01-01_001")

    def testFlagFileRead(self):
        # empty file should return True
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)
        # test with 2 lines and a trailing
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('file1\nfile2\n')
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)), set(['file1', 'file2']))
        # test with 2 lines and a trailing, Windows convention
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('file1\r\nfile2\r\n')
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)), set(['file1', 'file2']))

    def testAppendFlagFile(self):
        #  DO NOT CHANGE THE ORDER OF TESTS BELOW
        # prepare a file with 3 dataset types
        file_list = ['_ibl_extraRewards.times', '_ibl_lickPiezo.raw', '_ibl_lickPiezo.timestamps']
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('\n'.join(file_list))
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)), set(file_list))

        # with an existing file containing files, writing more files append to it
        file_list_2 = ['turltu']
        # also makes sure that if a string is provided it works
        flags.write_flag_file(self.tempfile.name, file_list_2[0])
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)),
                         set(file_list + file_list_2))

        # writing again keeps unique file values
        flags.write_flag_file(self.tempfile.name, file_list_2[0])
        n = sum([1 for f in flags.read_flag_file(self.tempfile.name) if f == file_list_2[0]])
        self.assertEqual(n, 1)

        # with an existing file containing files, writing empty filelist returns True for all files
        flags.write_flag_file(self.tempfile.name, None)
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

        # with an existing empty file, writing filelist returns True for all files
        flags.write_flag_file(self.tempfile.name, ['file1', 'file2'])
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

        # makes sure that read after write empty list also returns True
        flags.write_flag_file(self.tempfile.name, [])
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

        # with an existing empty file, writing filelist returns the list if clobber
        flags.write_flag_file(self.tempfile.name, ['file1', 'file2', 'file3'], clobber=True)
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)),
                         set(['file1', 'file2', 'file3']))

        # test the removal of a file within the list
        flags.excise_flag_file(self.tempfile.name, removed_files='file1')
        self.assertEqual(sorted(flags.read_flag_file(self.tempfile.name)), ['file2', 'file3'])

        # if file-list is True it means all files and file_list should be empty after read
        flags.write_flag_file(self.tempfile.name, file_list=True)
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

    def test_load_encoder_trial_info(self):
        self.session = Path(__file__).parent.joinpath('extractors', 'data', 'session_biased_ge5')
        data = raw.load_encoder_trial_info(self.session)
        self.assertTrue(data is not None)

    def test_load_camera_ssv_times(self):
        session = Path(__file__).parent.joinpath('extractors', 'data', 'session_ephys')
        with self.assertRaises(ValueError):
            raw.load_camera_ssv_times(session, 'tail')
        bonsai, camera = raw.load_camera_ssv_times(session, 'body')
        self.assertTrue(bonsai.size == camera.size == 6001)
        self.assertEqual(bonsai.dtype.str, '<M8[ns]')
        self.assertEqual(str(bonsai[0]), '2020-08-19T16:42:57.790361600')
        expected = np.array([69.466875, 69.5, 69.533, 69.566125, 69.59925])
        np.testing.assert_array_equal(expected, camera[:5])
        # Many sessions have the columns in the wrong order.  Here we write 5 lines from the
        # fixture file to another file in a temporary folder, with the columns swapped.
        from_file = session.joinpath('raw_video_data', '_iblrig_bodyCamera.timestamps.ssv')
        with tempfile.TemporaryDirectory() as tempdir:
            # New file with columns swapped (also checks loading files with UUID in name)
            filename = f'_iblrig_leftCamera.timestamps.{uuid.uuid4()}.ssv'
            to_file = Path(tempdir).joinpath('raw_video_data', filename)
            to_file.parent.mkdir(exist_ok=True)
            with open(from_file, 'r') as a, open(to_file, 'w') as b:
                for i in range(5):
                    # Read line from fixture file and write into file in swapped order
                    b.write('{1} {0} {2}'.format(*a.readline().split(' ')))
            assert to_file.exists(), 'failed to write test file'
            bonsai, camera = raw.load_camera_ssv_times(to_file.parents[1], 'left')
            # Verify that values returned in the same order as before
            self.assertEqual(bonsai.dtype.str, '<M8[ns]')
            self.assertEqual(camera.dtype.str, '<f8')
            self.assertAlmostEqual(69.466875, camera[0])

    def test_load_camera_gpio(self):
        """
        Embedded frame data comes from 057e25ef-3f80-42e8-aa9f-e259df8bc9ad, left camera
        :return:
        """
        session = Path(__file__).parent.joinpath('extractors', 'data', 'session_ephys')
        session2 = Path(__file__).parent.joinpath(
            'fixtures', 'io', 'data_loaders', '_iblrig_test_mouse_2020-01-01_001'
        )
        gpio = raw.load_camera_gpio(session, 'body', as_dicts=True)
        gpio2 = raw.load_camera_gpio(session2, 'left', as_dicts=True)
        self.assertEqual(len(gpio), 4)  # One dict per pin
        self.assertEqual(len(gpio2), 4)  # One dict per pin
        *gpio_, gpio_4 = gpio  # Check last dict; pin 4 should have one pulse
        self.assertTrue(all(k in ('indices', 'polarities') for k in gpio_4.keys()))
        np.testing.assert_array_equal(gpio_4['indices'], np.array([166, 172], dtype=np.int64))
        np.testing.assert_array_equal(gpio_4['polarities'], np.array([1, -1]))

        # Test raw flag
        gpio = raw.load_camera_gpio(session, 'body', as_dicts=False)
        self.assertEqual(gpio.dtype, bool)
        self.assertEqual(gpio.shape, (510, 4))

        # Test empty / None
        self.assertIsNone(raw.load_camera_gpio(None, 'body'))
        self.assertIsNone(raw.load_camera_gpio(session, 'right'))
        [self.assertIsNone(x) for x in raw.load_camera_gpio(session, 'right', as_dicts=True)]

        # Test noisy GPIO data
        side = 'right'
        with tempfile.TemporaryDirectory() as tdir:
            session_path = Path(tdir).joinpath('mouse', '2020-06-01', '001')
            session_path.joinpath('raw_video_data').mkdir(parents=True)
            # Test loads file with UUID
            did = uuid.uuid4()  # Random uuid
            filename = session_path / 'raw_video_data' / f'_iblrig_{side}Camera.GPIO.{did}.bin'
            np.full(1000, 1.87904819e+09, dtype=np.float64).tofile(filename)
            with self.assertLogs('ibllib', level='WARNING'):
                raw.load_camera_gpio(session_path, side, as_dicts=True)

            # Test dead pin array
            np.zeros(3000, dtype=np.float64).tofile(filename)
            with self.assertLogs('ibllib', level='ERROR'):
                gpio = raw.load_camera_gpio(session_path, side, as_dicts=True)
                [self.assertIsNone(x) for x in gpio]

    def test_load_camera_frame_count(self):
        """
        Embedded frame data comes from 057e25ef-3f80-42e8-aa9f-e259df8bc9ad, left camera
        :return:
        """
        session = Path(__file__).parent.joinpath('extractors', 'data', 'session_ephys')
        count = raw.load_camera_frame_count(session, 'body', raw=False)
        np.testing.assert_array_equal(count, np.arange(510, dtype=np.int32))
        self.assertEqual(count.dtype, int)

        # Test raw flag
        count = raw.load_camera_frame_count(session, 'body', raw=True)
        self.assertEqual(count[0], int(16696704))

        # Test empty / None
        self.assertIsNone(raw.load_camera_frame_count(None, 'body'))
        self.assertIsNone(raw.load_camera_frame_count(session, 'right'))

    def test_load_embedded_frame_data(self):
        session = Path(__file__).parent.joinpath('extractors', 'data', 'session_ephys')
        count, gpio = raw.load_embedded_frame_data(session, 'body')
        self.assertEqual(count[0], 0)
        self.assertIsInstance(gpio[-1], dict)
        count, gpio = raw.load_embedded_frame_data(session, 'body', raw=True)
        self.assertNotEqual(count[0], 0)
        self.assertIsInstance(gpio, np.ndarray)

    def test_load_camera_frameData(self):
        import pandas as pd
        fd_raw = raw.load_camera_frameData(self.bin_session_path, raw=True)
        fd = raw.load_camera_frameData(self.bin_session_path)
        # Wrong camera input file not found
        with self.assertRaises(AssertionError):
            raw.load_camera_frameData(self.bin_session_path, camera='right')
        # Shape
        self.assertTrue(fd.shape[1] == 4)
        self.assertTrue(fd_raw.shape[1] == 4)
        # Type
        self.assertTrue(isinstance(fd, pd.DataFrame))
        self.assertTrue(isinstance(fd_raw, pd.DataFrame))
        # Column names
        df_cols = ["Timestamp", "embeddedTimeStamp",
                   "embeddedFrameCounter", "embeddedGPIOPinState"]
        self.assertTrue(np.all([x in fd.columns for x in df_cols]))
        self.assertTrue(np.all([x in fd_raw.columns for x in df_cols]))
        # Column types
        parsed_dtypes = {
            "Timestamp": np.float64,
            "embeddedTimeStamp": np.float64,
            "embeddedFrameCounter": np.int64,
            "embeddedGPIOPinState": object
        }
        self.assertTrue(fd.dtypes.to_dict() == parsed_dtypes)
        self.assertTrue(all([x == np.int64 for x in fd_raw.dtypes]))

    def tearDown(self):
        self.tempfile.close()
        os.unlink(self.tempfile.name)


class TestsMisc(unittest.TestCase):

    def setUp(self):
        self._tdir = tempfile.TemporaryDirectory()
        # self.addClassCleanup(tmpdir.cleanup)  # py3.8
        self.tempdir = Path(self._tdir.name)
        self.subdirs = [
            self.tempdir / 'test_empty_parent',
            self.tempdir / 'test_empty_parent' / 'test_empty',
            self.tempdir / 'test_empty',
            self.tempdir / 'test_full',
        ]
        self.file = self.tempdir / 'test_full' / 'file.txt'

        _ = [x.mkdir() for x in self.subdirs]
        self.file.touch()

    def tearDown(self) -> None:
        self._tdir.cleanup()

    def _resetup_folders(self):
        self.file.unlink()
        (self.tempdir / 'test_full').rmdir()
        _ = [x.rmdir() for x in self.subdirs if x.exists()]
        _ = [x.mkdir() for x in self.subdirs]
        self.file.touch()

    def test_delete_empty_folders(self):
        pre = [x.exists() for x in self.subdirs]
        pre_expected = [True, True, True, True]
        self.assertTrue(all([x == y for x, y in zip(pre, pre_expected)]))

        # Test dry run
        pos_expected = None
        pos = misc.delete_empty_folders(self.tempdir)
        self.assertTrue(pos == pos_expected)
        # Test dry=False, non recursive
        pos_expected = [True, False, False, True]
        misc.delete_empty_folders(self.tempdir, dry=False)
        pos = [x.exists() for x in self.subdirs]
        self.assertTrue(all([x == y for x, y in zip(pos, pos_expected)]))

        self._resetup_folders()

        # Test recursive
        pos_expected = [False, False, False, True]
        misc.delete_empty_folders(self.tempdir, dry=False, recursive=True)
        pos = [x.exists() for x in self.subdirs]
        self.assertTrue(all([x == y for x, y in zip(pos, pos_expected)]))


class TestsGlobus(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.multiple('globus_sdk',
                                      NativeAppAuthClient=unittest.mock.DEFAULT,
                                      RefreshTokenAuthorizer=unittest.mock.DEFAULT,
                                      TransferClient=unittest.mock.DEFAULT)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_as_globus_path(self):
        # A Windows path
        if sys.platform == 'win32':
            # "/E/FlatIron/integration"
            actual = globus.as_globus_path('E:\\FlatIron\\integration')
            self.assertTrue(actual.startswith('/E/'))
            # A relative POSIX path
            actual = globus.as_globus_path('/mnt/foo/../data/integration')
            expected = '/mnt/data/integration'  # "/C/mnt/data/integration
            self.assertTrue(actual.endswith(expected))

        # A globus path
        actual = globus.as_globus_path('/E/FlatIron/integration')
        expected = '/E/FlatIron/integration'
        self.assertEqual(expected, actual)

    @unittest.mock.patch('iblutil.io.params.read')
    def test_login_auto(self, mock_params):
        client_id = 'h3u2ier'
        # Test ValueError thrown with incorrect parameters
        mock_params.return_value = None  # No parameters saved
        with self.assertRaises(ValueError):
            globus.login_auto(client_id)
        # mock_params.assert_called_with('globus/default')

        pars = params.from_dict({'access_token': '7r3hj89', 'expires_at_seconds': '2020-09-10'})
        mock_params.return_value = pars  # Incomplete parameter object
        with self.assertRaises(ValueError):
            globus.login_auto(client_id)

        # Complete parameter object
        mock_params.return_value = pars.set('refresh_token', '37yh4')
        gtc = globus.login_auto(client_id)
        self.assertIsInstance(gtc, unittest.mock.Mock)
        mock, _ = self.patcher.get_original()
        mock.assert_called_once_with(client_id)


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.one = ONE(**TEST_DB)

    def setUp(self) -> None:
        self.eid = '8dd0fcb0-1151-4c97-ae35-2e2421695ad7'
        self.url = ('https://ibl.flatironinstitute.org/mainenlab/'
                    'Subjects/ZM_1743/2019-06-14/001/raw_video_data/'
                    '_iblrig_leftCamera.raw.71cfeef2-2aa5-46b5-b88f-ca07e3d92474.mp4')

    def test_label_from_path(self):
        # Test file path
        session_path = self.one.eid2path(self.eid)
        video_path = session_path / 'raw_video_data' / '_iblrig_bodyCamera.raw.mp4'
        label = video.label_from_path(video_path)
        self.assertEqual('body', label)
        # Test URL
        label = video.label_from_path(self.url)
        self.assertEqual('left', label)
        # Test file name
        label = video.label_from_path('_iblrig_rightCamera.raw.mp4')
        self.assertEqual('right', label)
        # Test wrong file
        label = video.label_from_path('_iblrig_taskSettings.raw.json')
        self.assertIsNone(label)

    def test_url_from_eid(self):
        assert self.one.mode != 'remote'
        actual = video.url_from_eid(self.eid, 'left', self.one)
        self.assertEqual(self.url, actual)
        actual = video.url_from_eid(self.eid, one=self.one)
        expected = {'left': self.url}
        self.assertEqual(expected, actual)
        actual = video.url_from_eid(self.eid, label=('left', 'right'), one=self.one)
        expected = {'left': self.url, 'right': None}
        self.assertEqual(expected, actual)

        # Test remote mode
        old_mode = self.one.mode
        self.one.mode = 'remote'
        actual = video.url_from_eid(self.eid, label='left', one=self.one)
        self.assertEqual(self.url, actual)
        self.one.mode = old_mode

        # Test arg checks
        with self.assertRaises(ValueError):
            video.url_from_eid(self.eid, 'back')

    def test_assert_valid_label(self):
        with self.assertRaises(ValueError):
            video.assert_valid_label('tail')
        label = video.assert_valid_label('LEFT')
        self.assertEqual(label, 'left')
        # Verify works with lists
        labels = video.assert_valid_label(['Right', 'body'])
        self.assertEqual(labels, ('right', 'body'))
        with self.assertRaises(TypeError):
            video.assert_valid_label(None)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

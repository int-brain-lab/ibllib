import logging
import tempfile
import unittest
from pathlib import Path

import ciso8601
import numpy as np

import alf.extractors as ex
import ibllib.io.flags as flags
from alf import transfer_rig_data
from ibllib.io import raw_data_loaders as loaders


class TestExtractTrialData(unittest.TestCase):

    def setUp(self):
        self.session_path = Path(__file__).parent.joinpath('data')
        self.data = loaders.load_data(self.session_path)
        # turn off logging for unit testing as we will purposely go into warning/error cases
        self.logger = logging.getLogger('ibllib').setLevel(50)

    def test_stimOn_times(self):
        st = ex.training_trials.get_stimOn_times('', save=False, data=self.data)
        self.assertTrue(isinstance(st, np.ndarray))

    def test_encoder_positions_duds(self):
        dy = loaders.load_encoder_positions(self.session_path)
        self.assertEqual(dy.bns_ts.dtype.name, 'datetime64[ns]')
        self.assertTrue(dy.shape[0] == 2)

    def test_encoder_events_duds(self):
        dy = loaders.load_encoder_events(self.session_path)
        self.assertEqual(dy.bns_ts.dtype.name, 'datetime64[ns]')
        self.assertTrue(dy.shape[0] == 7)

    def test_interpolation(self):
        # straight test that it returns an usable function
        ta = np.array([0., 1., 2., 3., 4., 5.])
        tb = np.array([0., 1.1, 2.0, 2.9, 4., 5.])
        finterp = ex.time_interpolation(ta, tb)
        self.assertTrue(np.all(finterp(ta) == tb))
        # next test if sizes are not similar
        tc = np.array([0., 1.1, 2.0, 2.9, 4., 5., 6.])
        finterp = ex.time_interpolation(ta, tc)
        self.assertTrue(np.all(finterp(ta) == tb))

    def test_ciso8601(self):
        dt = ciso8601.parse_datetime('2018-01-16T14:21:32')
        self.assertFalse(not dt)


class TestTransferRigData(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_data_folder = Path(self.tmp_dir.name)
        self.session_path = self.root_data_folder / "src" / 'algernon' / '2019-01-21' / '001'
        self.session_path.mkdir(parents=True, exist_ok=True)
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data").mkdir()
        (self.session_path / "raw_video_data").mkdir()
        (self.session_path / "raw_behavior_data" / "random.data1.ext").touch()

    def test_transfer(self):
        src_subjects_path = self.root_data_folder / "src"
        dst_subjects_path = self.root_data_folder / "dst"
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        gsrc = [x.name for x in list(src_subjects_path.rglob('*.*'))]
        gdst = [x.name for x in list(dst_subjects_path.rglob('*.*'))]
        self.assertTrue('extract_me.flag' in gdst)
        gdst = [x for x in gdst if x != 'extract_me.flag']
        self.assertEqual(gsrc, gdst)
        # Test if folder exists not copy because no flag
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        # Test if flag exists and folder exists in dst
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)

    def tearDown(self):
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main(exit=False)

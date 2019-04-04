import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

import alf.extractors as ex
import ibllib.io.flags as flags
from alf import transfer_rig_data
from ibllib.io import raw_data_loaders as loaders


class TestExtractTrialData(unittest.TestCase):

    def setUp(self):
        self.session_path = Path(__file__).parent / 'data' / 'session'
        self.data = loaders.load_data(self.session_path)
        self.session_path_biased = Path(__file__).parent / 'data' / 'session_biased'
        self.data_biased = loaders.load_data(self.session_path_biased)
        self.wheel_path = Path(__file__).parent / 'data' / 'wheel'
        # turn off logging for unit testing as we will purposely go into warning/error cases
        self.logger = logging.getLogger('ibllib').setLevel(50)

    def test_stimOn_times(self):
        st = ex.training_trials.get_stimOn_times('', save=False, data=self.data)
        self.assertTrue(isinstance(st, np.ndarray))

    def test_encoder_positions_duds(self):
        dy = loaders.load_encoder_positions(self.session_path)
        self.assertEqual(dy.bns_ts.dtype.name, 'object')
        self.assertTrue(dy.shape[0] == 14)

    def test_encoder_events_duds(self):
        dy = loaders.load_encoder_events(self.session_path)
        self.assertEqual(dy.bns_ts.dtype.name, 'object')
        self.assertTrue(dy.shape[0] == 7)

    def test_encoder_positions_clock_reset(self):
        dy = loaders.load_encoder_positions(self.session_path)
        dat = np.array([849736, 1532230, 1822449, 1833514, 1841566, 1848206, 1853979, 1859144])
        self.assertTrue(np.all(np.diff(dy['re_ts']) > 0))
        self.assertTrue(all(dy['re_ts'][6:] - 2**32 - dat == 0))

    def test_encoder_positions_clock_errors(self):
        # here we test for 2 kinds of file corruption that happen
        # 1/2 the first sample time is corrupt and absurdly high and should be discarded
        # 2/2 2 samples are swapped and need to be swapped back
        dy = loaders.load_encoder_positions(self.session_path_biased)
        self.assertTrue(np.all(np.diff(np.array(dy.re_ts)) > 0))

    def test_wheel_folder(self):
        # the wheel folder contains other errors in bpod output that had to be addressed
        # 2 first samples timestamp AWOL instead of only one
        wf = self.wheel_path / '_iblrig_encoderPositions.raw.2firstsamples.ssv'
        df = loaders._load_encoder_positions_file(wf)
        self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))
        # corruption in the middle of file
        wf = self.wheel_path / '_iblrig_encoderEvents.raw.CorruptMiddle.ssv'
        df = loaders._load_encoder_events_file(wf)
        self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))

    def test_interpolation(self):
        # straight test that it returns an usable function
        ta = np.array([0., 1., 2., 3., 4., 5.])
        tb = np.array([0., 1.1, 2.0, 2.9, 4., 5.])
        finterp = ex.training_wheel.time_interpolation(ta, tb)
        self.assertTrue(np.all(finterp(ta) == tb))
        # next test if sizes are not similar
        tc = np.array([0., 1.1, 2.0, 2.9, 4., 5., 6.])
        finterp = ex.training_wheel.time_interpolation(ta, tc)
        self.assertTrue(np.all(finterp(ta) == tb))

    def test_choice(self):
        choice = ex.training_trials.get_choice(self.session_path)
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in self.data])
        self.assertTrue(all(choice[trial_nogo]) == 0)
        signed_contrast = np.array([t['signed_contrast'] for t in self.data])
        if not all(signed_contrast == 0):
            return
        else:
            # This will only fail is the mouse always answers with no go on a
            # 0% contrast OR if the choice has been extracted wrong
            self.assertTrue(any(choice[signed_contrast == 0] != 0))

    def test_goCue_times(self):
        gc_times = ex.training_trials.get_goCueOnset_times(self.session_path)
        self.assertTrue(not gc_times or gc_times)

    def test_contrastLR(self):
        cl, cr = ex.training_trials.get_contrastLR(self.session_path)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))
        cl, cr = ex.biased_trials.get_contrastLR(self.session_path_biased)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))


class TestTransferRigData(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_data_folder = Path(self.tmp_dir.name)
        self.session_path = self.root_data_folder / "src" / 'algernon' / '2019-01-21' / '001'
        self.session_path.mkdir(parents=True, exist_ok=True)
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data").mkdir()
        (self.session_path / "raw_video_data").mkdir()
        (self.session_path / "raw_behavior_data" / '_iblrig_micData.raw.wav').touch()
        (self.session_path / "raw_video_data" / '_iblrig_leftCamera.raw.avi').touch()

    def test_transfer(self):
        src_subjects_path = self.root_data_folder / "src"
        dst_subjects_path = self.root_data_folder / "dst"
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        gsrc = [x.name for x in list(src_subjects_path.rglob('*.*'))]
        gdst = [x.name for x in list(dst_subjects_path.rglob('*.*'))]
        self.assertTrue('extract_me.flag' in gdst)
        gdst = [x for x in gdst if x != 'extract_me.flag']
        self.assertTrue('compress_video.flag' in gdst)
        gdst = [x for x in gdst if x != 'compress_video.flag']
        self.assertTrue('_iblrig_micData.raw.wav' in gdst)
        gdst = [x for x in gdst if x != '_iblrig_micData.raw.wav']
        self.assertTrue('_iblrig_leftCamera.raw.avi' in gdst)
        gdst = [x for x in gdst if x != '_iblrig_leftCamera.raw.avi']

        self.assertEqual(gsrc, gdst)
        # Test if folder exists not copy because no flag
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        # Test if flag exists and folder exists in dst
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / '_iblrig_micData.raw.wav').touch()
        (self.session_path / "raw_video_data" / '_iblrig_leftCamera.raw.avi').touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        # Test transfer w/o video and audio
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / "random.data1.ext").touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)

    def tearDown(self):
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main(exit=False)

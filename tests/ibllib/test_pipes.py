import unittest
import tempfile
from pathlib import Path

import ibllib.pipes.extract_session
from ibllib.pipes import experimental_data
from ibllib.pipes import misc


class TestCompression(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_empty_flag_file(self):
        with tempfile.TemporaryDirectory() as tdir:
            flag = Path(tdir).joinpath('compress_video.flag')
            flag.touch()
            experimental_data.compress_video(tdir, dry=True)
            self.assertFalse(flag.exists())


class TestExtractors(unittest.TestCase):

    def test_task_names_extractors(self):
        """
        This is to test against regressions
        """
        task_out = [
            ('_iblrig_tasks_biasedChoiceWorld3.7.0', 'biased'),
            ('_iblrig_tasks_biasedScanningChoiceWorld5.2.3', 'biased'),
            ('_iblrig_tasks_trainingChoiceWorld3.6.0', 'training'),
            ('_iblrig_tasks_ephysChoiceWorld5.1.3', 'ephys'),
            ('_iblrig_calibration_frame2TTL4.1.3', None),
            ('_iblrig_tasks_habituationChoiceWorld3.6.0', 'habituation'),
            ('_iblrig_tasks_scanningOptoChoiceWorld5.0.0', None),
            ('_iblrig_tasks_RewardChoiceWorld4.1.3', None),
            ('_iblrig_calibration_screen4.1.3', None),
            ('_iblrig_tasks_ephys_certification4.1.3', 'sync_ephys'),
        ]
        for to in task_out:
            out = ibllib.pipes.extract_session.get_task_extractor_type(to[0])
            self.assertEqual(out, to[1])

    def test_get_session_folder(self):
        inp = (Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001/raw_behavior_data/'
                    '_iblrig_micData.raw.wav'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),)
        out = (Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),)
        for i, o in zip(inp, out):
            self.assertEqual(o, ibllib.pipes.extract_session.get_session_path(i))


class TestPipesMisc(unittest.TestCase):
    """
    """

    def setUp(self):
        # Define folders
        self.local_data_path = Path(tempfile.gettempdir()) / 'iblrig_data' / 'Subjects'
        self.remote_data_path = Path(tempfile.gettempdir()) / 'iblrig_data' / 'Subjects'
        self.local_session_path = self.local_data_path / 'Subjects' / \
            'test_subject' / '1900-01-01' / '001'
        self.rbdata = self.local_session_path / 'raw_behavior_data'
        self.rvdata = self.local_session_path / 'raw_video_data'
        self.redata = self.local_session_path / 'raw_ephys_data'
        # Make folders
        self.local_data_path.mkdir(exist_ok=True, parents=True)
        self.remote_data_path.mkdir(exist_ok=True, parents=True)
        self.local_session_path.mkdir(exist_ok=True, parents=True)
        self.rbdata.mkdir(exist_ok=True, parents=True)
        self.rvdata.mkdir(exist_ok=True, parents=True)
        self.redata.mkdir(exist_ok=True, parents=True)

    def test_behavior_exists(self):
        assert(misc.behavior_exists('.') is False)
        assert(misc.behavior_exists(self.test_local_session) is True)

    def test_get_new_filename(self):
        binFname3A = 'ignoreThisPart_g0_t0.imec.ap.bin'
        binFname3B = 'ignoreThisPart_g0_t0.imec0.ap.bin'
        metaFname3A = 'ignoreThisPart_g0_t0.imec.ap.meta'
        metaFname3B = 'ignoreThisPart_g0_t0.imec0.ap.meta'
        probe1_3B_Meta = 'ignoreThisPart_g0_t0.imec1.ap.meta'
        different_gt = 'ignoreThisPart_g1_t2.imec0.ap.meta'

        newfname = misc.get_new_filename(binFname3A)
        self.assertTrue(newfname == '_spikeglx_ephysData_g0_t0.imec.ap.bin')
        newfname = misc.get_new_filename(binFname3B)
        self.assertTrue(newfname == '_spikeglx_ephysData_g0_t0.imec0.ap.bin')
        newfname = misc.get_new_filename(metaFname3A)
        self.assertTrue(newfname == '_spikeglx_ephysData_g0_t0.imec.ap.meta')
        newfname = misc.get_new_filename(metaFname3B)
        self.assertTrue(newfname == '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        newfname = misc.get_new_filename(probe1_3B_Meta)
        self.assertTrue(newfname == '_spikeglx_ephysData_g0_t0.imec1.ap.meta')
        newfname = misc.get_new_filename(different_gt)
        self.assertTrue(newfname == '_spikeglx_ephysData_g1_t2.imec0.ap.meta')

    def test_rename_ephys_files(self):
        # Make a bunch of fake files 3A
        misc.rename_ephys_files(self.test_local_session)
        # Make a bunch of fake files 3B

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main(exit=False)

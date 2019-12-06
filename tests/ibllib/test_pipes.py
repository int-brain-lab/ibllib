import shutil
import tempfile
import unittest
from pathlib import Path

import ibllib.pipes.extract_session
from ibllib.pipes import experimental_data, misc


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


class TestPipesMisc(unittest.TestCase):
    """
    """

    def setUp(self):
        # Define folders
        self.root_test_folder = Path(tempfile.gettempdir()) / 'ibllib_test'
        self.local_data_path = self.root_test_folder / 'local' / 'iblrig_data' / 'Subjects'
        self.remote_data_path = self.root_test_folder / 'remote' / 'iblrig_data' / 'Subjects'
        self.local_session_path_3A = self.local_data_path / \
            'test_subject' / '1900-01-01' / '001'
        self.local_session_path_3B = self.local_data_path / \
            'test_subject' / '1900-01-02' / '001'
        self.raw_ephys_data_path_3A = self.local_session_path_3A / 'raw_ephys_data'
        self.raw_ephys_data_path_3B = self.local_session_path_3B / 'raw_ephys_data'
        self.probe00_path_3A = self.raw_ephys_data_path_3A / 'probe00'
        self.probe01_path_3A = self.raw_ephys_data_path_3A / 'probe01'
        self.probe00_path_3B = self.raw_ephys_data_path_3B / 'probe00'
        self.probe01_path_3B = self.raw_ephys_data_path_3B / 'probe01'

        self.bad_ephys_folder_3A = self.raw_ephys_data_path_3A / 'bad_ephys_folder'
        self.bad_probe00_folder_3A = self.bad_ephys_folder_3A / 'some_probe00_folder'
        self.bad_probe01_folder_3A = self.bad_ephys_folder_3A / 'some_probe01_folder'

        self.bad_ephys_folder_3B = self.raw_ephys_data_path_3B / 'bad_folder_g0_t0'
        self.bad_probe00_folder_3B = self.bad_ephys_folder_3B / 'bad_folder_g0_t0.imec0'
        self.bad_probe01_folder_3B = self.bad_ephys_folder_3B / 'bad_folder_g0_t0.imec1'
        # Make folders
        (self.local_session_path_3A / 'raw_behavior_data').mkdir(exist_ok=True, parents=True)
        (self.local_session_path_3B / 'raw_behavior_data').mkdir(exist_ok=True, parents=True)
        self.probe00_path_3A.mkdir(exist_ok=True, parents=True)
        self.probe00_path_3B.mkdir(exist_ok=True, parents=True)
        self.probe01_path_3A.mkdir(exist_ok=True, parents=True)
        self.probe01_path_3B.mkdir(exist_ok=True, parents=True)

        self.bad_probe00_folder_3A.mkdir(exist_ok=True, parents=True)
        self.bad_probe01_folder_3A.mkdir(exist_ok=True, parents=True)
        self.bad_probe00_folder_3B.mkdir(exist_ok=True, parents=True)
        self.bad_probe01_folder_3B.mkdir(exist_ok=True, parents=True)
        # Make some files
        self.populate_bad_folders()

    def populate_bad_folders(self):
        # 3A
        (self.bad_probe00_folder_3A / 'blabla_g0_t0.imec.ap.bin').touch()
        (self.bad_probe00_folder_3A / 'blabla_g0_t0.imec.lf.bin').touch()
        (self.bad_probe00_folder_3A / 'blabla_g0_t0.imec.ap.meta').touch()
        (self.bad_probe00_folder_3A / 'blabla_g0_t0.imec.lf.meta').touch()

        (self.bad_probe01_folder_3A / 'blabla_g0_t0.imec.ap.bin').touch()
        (self.bad_probe01_folder_3A / 'blabla_g0_t0.imec.lf.bin').touch()
        (self.bad_probe01_folder_3A / 'blabla_g0_t0.imec.ap.meta').touch()
        (self.bad_probe01_folder_3A / 'blabla_g0_t0.imec.lf.meta').touch()
        # 3B
        (self.bad_probe00_folder_3B / 'blabla_g0_t0.imec0.ap.bin').touch()
        (self.bad_probe00_folder_3B / 'blabla_g0_t0.imec0.lf.bin').touch()
        (self.bad_probe00_folder_3B / 'blabla_g0_t0.imec0.ap.meta').touch()
        (self.bad_probe00_folder_3B / 'blabla_g0_t0.imec0.lf.meta').touch()

        (self.bad_probe01_folder_3B / 'blabla_g0_t0.imec1.ap.bin').touch()
        (self.bad_probe01_folder_3B / 'blabla_g0_t0.imec1.lf.bin').touch()
        (self.bad_probe01_folder_3B / 'blabla_g0_t0.imec1.ap.meta').touch()
        (self.bad_probe01_folder_3B / 'blabla_g0_t0.imec1.lf.meta').touch()

        (self.bad_ephys_folder_3B / 'blabla_g0_t0.nidq.bin').touch()
        (self.bad_ephys_folder_3B / 'blabla_g0_t0.nidq.meta').touch()

    def test_behavior_exists(self):
        assert(misc.behavior_exists('.') is False)
        assert(misc.behavior_exists(self.local_session_path_3A) is True)

    def test_check_transfer(self):
        misc.check_transfer(self.bad_probe00_folder_3A, self.bad_probe00_folder_3A)
        misc.check_transfer(str(self.bad_probe00_folder_3A), self.bad_probe00_folder_3A)
        with self.assertRaises(AssertionError) as cm:
            misc.check_transfer(self.bad_probe00_folder_3A, self.bad_probe00_folder_3B)
        err = cm.exception
        self.assertEqual(str(err), 'src_files != dst_files')

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

    def _test_rename_ephys_files(self):
        # Test 3A
        misc.rename_ephys_files(self.local_session_path_3A)
        ap_files = list(self.local_session_path_3A.rglob('*.ap.*'))
        lf_files = list(self.local_session_path_3A.rglob('*.lf.*'))
        self.assertTrue(len(ap_files) >= 1)
        self.assertTrue(len(lf_files) >= 1)
        for f in ap_files:
            print(f.parent.name, f.name)
            self.assertTrue('_spikeglx_ephysData_g' in str(f))
            self.assertTrue('_t' in str(f))
            self.assertTrue('.imec.ap' in str(f))
            self.assertTrue(('.bin' in str(f) or '.meta' in str(f)))
            self.assertTrue(('probe00' in str(f) or 'probe01' in str(f)))
        for f in lf_files:
            print(f.parent.name, f.name)
            self.assertTrue('_spikeglx_ephysData_g' in str(f))
            self.assertTrue('_t' in str(f))
            self.assertTrue('.imec.lf' in str(f))
            self.assertTrue(('.bin' in str(f) or '.meta' in str(f)))  # Test 3B
            self.assertTrue(('probe00' in str(f) or 'probe01' in str(f)))
        # Test 3B
        misc.rename_ephys_files(self.local_session_path_3B)
        ap_files = list(self.local_session_path_3B.rglob('*.ap.*'))
        lf_files = list(self.local_session_path_3B.rglob('*.lf.*'))
        nidq_files = list(self.local_session_path_3B.rglob('*.nidq.*'))
        for f in ap_files:
            print(f.name)
            self.assertTrue('_spikeglx_ephysData_g' in str(f))
            self.assertTrue('_t' in str(f))
            self.assertTrue(('.imec0.ap' in str(f) or 'imec1.ap' in str(f)))
            self.assertTrue(('.bin' in str(f) or '.meta' in str(f)))
        for f in lf_files:
            print(f.name)
            self.assertTrue('_spikeglx_ephysData_g' in str(f))
            self.assertTrue('_t' in str(f))
            self.assertTrue(('.imec0.lf' in str(f) or 'imec1.lf' in str(f)))
            self.assertTrue(('.bin' in str(f) or '.meta' in str(f)))  # Test 3B
        for f in nidq_files:
            print(f.name)
            self.assertTrue('_spikeglx_ephysData_g' in str(f))
            self.assertTrue('_t' in str(f))
            self.assertTrue('.nidq' in str(f))
            self.assertTrue(('.bin' in str(f) or '.meta' in str(f)))

    def test_rename_and_move(self):
        self._test_rename_ephys_files()
        # TODO: Move files from the bad folders to the local_probe folders?
        # Test for 3A
        misc.move_ephys_files(self.local_session_path_3A)
        probe00_files = list(self.probe00_path_3A.rglob('*'))
        probe00_file_names = [x.name for x in probe00_files]
        probe01_files = list(self.probe01_path_3A.rglob('*'))
        probe01_file_names = [x.name for x in probe01_files]

        self.assertTrue(
            '_spikeglx_ephysData_g0_t0.imec.ap.bin' in probe00_file_names and
            '_spikeglx_ephysData_g0_t0.imec.ap.meta' in probe00_file_names and
            '_spikeglx_ephysData_g0_t0.imec.lf.bin' in probe00_file_names and
            '_spikeglx_ephysData_g0_t0.imec.lf.meta' in probe00_file_names
        )
        self.assertTrue(
            '_spikeglx_ephysData_g0_t0.imec.ap.bin' in probe01_file_names and
            '_spikeglx_ephysData_g0_t0.imec.ap.meta' in probe01_file_names and
            '_spikeglx_ephysData_g0_t0.imec.lf.bin' in probe01_file_names and
            '_spikeglx_ephysData_g0_t0.imec.lf.meta' in probe01_file_names
        )
        # Test for 3B
        misc.move_ephys_files(self.local_session_path_3B)
        probe00_files = list(self.probe00_path_3B.rglob('*'))
        probe00_file_names = [x.name for x in probe00_files]
        probe01_files = list(self.probe01_path_3B.rglob('*'))
        probe01_file_names = [x.name for x in probe01_files]
        nidq_files = list(self.raw_ephys_data_path_3B.glob('*.nidq.*'))
        nidq_file_names = [x.name for x in nidq_files]
        self.assertTrue(
            '_spikeglx_ephysData_g0_t0.imec0.ap.bin' in probe00_file_names and
            '_spikeglx_ephysData_g0_t0.imec0.ap.meta' in probe00_file_names and
            '_spikeglx_ephysData_g0_t0.imec0.lf.bin' in probe00_file_names and
            '_spikeglx_ephysData_g0_t0.imec0.lf.meta' in probe00_file_names
        )
        self.assertTrue(
            '_spikeglx_ephysData_g0_t0.imec1.ap.bin' in probe01_file_names and
            '_spikeglx_ephysData_g0_t0.imec1.ap.meta' in probe01_file_names and
            '_spikeglx_ephysData_g0_t0.imec1.lf.bin' in probe01_file_names and
            '_spikeglx_ephysData_g0_t0.imec1.lf.meta' in probe01_file_names
        )
        self.assertTrue(
            '_spikeglx_ephysData_g0_t0.nidq.bin' in nidq_file_names and
            '_spikeglx_ephysData_g0_t0.nidq.meta' in nidq_file_names
        )

    def test_create_ephys_flags(self):
        extract = self.local_session_path_3A.joinpath('extract_ephys.flag')
        qc = self.local_session_path_3A.joinpath('raw_ephys_qc.flag')
        spike_sorting0 = self.local_session_path_3A / 'raw_ephys_data' / 'probe00'
        spike_sorting1 = self.local_session_path_3A / 'raw_ephys_data' / 'probe01'
        spike_sorting0 = spike_sorting0.joinpath('spike_sorting.flag')
        spike_sorting1 = spike_sorting1.joinpath('spike_sorting.flag')
        misc.create_ephys_flags(self.local_session_path_3A)
        self.assertTrue(extract.exists())
        self.assertTrue(qc.exists())
        self.assertTrue(spike_sorting0.exists())
        self.assertTrue(spike_sorting1.exists())
        # Test recreate
        misc.create_ephys_flags(self.local_session_path_3A)
        self.assertTrue(extract.exists())
        self.assertTrue(qc.exists())
        self.assertTrue(spike_sorting0.exists())
        self.assertTrue(spike_sorting1.exists())
        # Remove flags after test
        extract.unlink()
        qc.unlink()
        spike_sorting0.unlink()
        spike_sorting1.unlink()
        # test removal
        self.assertFalse(extract.exists())
        self.assertFalse(qc.exists())
        self.assertFalse(spike_sorting0.exists())
        self.assertFalse(spike_sorting1.exists())

    def tearDown(self):
        shutil.rmtree(self.root_test_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(exit=False)

import shutil
import tempfile
import unittest
from pathlib import Path

import ibllib.io.extractors.base
import ibllib.tests.fixtures.utils as fu
from ibllib.pipes import misc
from oneibl.one import ONE
import ibllib.pipes.scan_fix_passive_files as fix


class TestExtractors2Tasks(unittest.TestCase):

    def test_task_to_pipeline(self):
        dd = ibllib.io.extractors.base._get_task_types_json_config()
        types = list(set([dd[k] for k in dd]))
        # makes sure that for every defined task type there is an acutal pipeline
        for type in types:
            assert ibllib.io.extractors.base._get_pipeline_from_task_type(type)
            print(type, ibllib.io.extractors.base._get_pipeline_from_task_type(type))
        pipe_out = [
            ("ephys_biased_opto", "ephys"),
            ("ephys_training", "ephys"),
            ("training", "training"),
            ("biased_opto", "training"),
            ("habituation", "training"),
            ("biased", "training"),
            ("mock_ephys", "ephys"),
            ("sync_ephys", "ephys"),
            ("ephys", "ephys"),
        ]
        for typ, exp in pipe_out:
            assert ibllib.io.extractors.base._get_pipeline_from_task_type(typ) == exp

    def test_task_names_extractors(self):
        """
        This is to test against regressions
        """
        # input a tuple task /
        task_out = [
            ("_iblrig_tasks_biasedChoiceWorld3.7.0", "biased"),
            ("_iblrig_tasks_biasedScanningChoiceWorld5.2.3", "biased"),
            ("_iblrig_tasks_trainingChoiceWorld3.6.0", "training"),
            ("_iblrig_tasks_trainingChoiceWorldWidefield", "ephys_training"),
            ("_iblrig_tasks_widefieldChoiceWorld", "ephys"),
            ("_iblrig_tasks_ephysChoiceWorld5.1.3", "ephys"),
            ("_iblrig_calibration_frame2TTL4.1.3", None),
            ("_iblrig_tasks_habituationChoiceWorld3.6.0", "habituation"),
            ("_iblrig_tasks_scanningOptoChoiceWorld5.0.0", None),
            ("_iblrig_tasks_RewardChoiceWorld4.1.3", None),
            ("_iblrig_calibration_screen4.1.3", None),
            ("_iblrig_tasks_ephys_certification4.1.3", "sync_ephys"),
            ("optokarolinaChoiceWorld5.34", "biased"),
            ("karolinaChoiceWorld5.34", "biased"),
            ("ephyskarolinaChoiceWorld4.34", "ephys"),
            ("passive_opto", "ephys"),
            ("_iblrig_tasks_opto_ephysChoiceWorld", "ephys_biased_opto"),
            ("_iblrig_tasks_opto_biasedChoiceWorld", "biased_opto"),
        ]
        for to in task_out:
            out = ibllib.io.extractors.base.get_task_extractor_type(to[0])
            assert out == to[1]


class TestPipesMisc(unittest.TestCase):
    """"""

    def setUp(self):
        # Define folders
        self.root_test_folder = Path(tempfile.gettempdir()) / "ibllib_test"
        self.local_data_path = self.root_test_folder / "local" / "iblrig_data" / "Subjects"
        self.remote_data_path = self.root_test_folder / "remote" / "iblrig_data" / "Subjects"
        self.local_session_path_3A = self.local_data_path / "test_subject" / "1900-01-01" / "001"
        self.local_session_path_3B = self.local_data_path / "test_subject" / "1900-01-02" / "001"
        self.raw_ephys_data_path_3A = self.local_session_path_3A / "raw_ephys_data"
        self.raw_ephys_data_path_3B = self.local_session_path_3B / "raw_ephys_data"
        self.probe00_path_3A = self.raw_ephys_data_path_3A / "probe00"
        self.probe01_path_3A = self.raw_ephys_data_path_3A / "probe01"
        self.probe00_path_3B = self.raw_ephys_data_path_3B / "probe00"
        self.probe01_path_3B = self.raw_ephys_data_path_3B / "probe01"

        self.bad_ephys_folder_3A = self.raw_ephys_data_path_3A / "bad_ephys_folder"
        self.bad_probe00_folder_3A = self.bad_ephys_folder_3A / "some_probe00_folder"
        self.bad_probe01_folder_3A = self.bad_ephys_folder_3A / "some_probe01_folder"

        self.bad_ephys_folder_3B = self.raw_ephys_data_path_3B / "bad_folder_g0_t0"
        self.bad_probe00_folder_3B = self.bad_ephys_folder_3B / "bad_folder_g0_t0.imec0"
        self.bad_probe01_folder_3B = self.bad_ephys_folder_3B / "bad_folder_g0_t0.imec1"
        # Make folders
        (self.local_session_path_3A / "raw_behavior_data").mkdir(exist_ok=True, parents=True)
        (self.local_session_path_3B / "raw_behavior_data").mkdir(exist_ok=True, parents=True)
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
        (self.bad_probe00_folder_3A / "blabla_g0_t0.imec.ap.bin").touch()
        (self.bad_probe00_folder_3A / "blabla_g0_t0.imec.lf.bin").touch()
        (self.bad_probe00_folder_3A / "blabla_g0_t0.imec.ap.meta").touch()
        (self.bad_probe00_folder_3A / "blabla_g0_t0.imec.lf.meta").touch()

        (self.bad_probe01_folder_3A / "blabla_g0_t0.imec.ap.bin").touch()
        (self.bad_probe01_folder_3A / "blabla_g0_t0.imec.lf.bin").touch()
        (self.bad_probe01_folder_3A / "blabla_g0_t0.imec.ap.meta").touch()
        (self.bad_probe01_folder_3A / "blabla_g0_t0.imec.lf.meta").touch()
        # 3B
        (self.bad_probe00_folder_3B / "blabla_g0_t0.imec0.ap.bin").touch()
        (self.bad_probe00_folder_3B / "blabla_g0_t0.imec0.lf.bin").touch()
        (self.bad_probe00_folder_3B / "blabla_g0_t0.imec0.ap.meta").touch()
        (self.bad_probe00_folder_3B / "blabla_g0_t0.imec0.lf.meta").touch()

        (self.bad_probe01_folder_3B / "blabla_g0_t0.imec1.ap.bin").touch()
        (self.bad_probe01_folder_3B / "blabla_g0_t0.imec1.lf.bin").touch()
        (self.bad_probe01_folder_3B / "blabla_g0_t0.imec1.ap.meta").touch()
        (self.bad_probe01_folder_3B / "blabla_g0_t0.imec1.lf.meta").touch()

        (self.bad_ephys_folder_3B / "blabla_g0_t0.nidq.bin").touch()
        (self.bad_ephys_folder_3B / "blabla_g0_t0.nidq.meta").touch()

    def test_behavior_exists(self):
        assert misc.behavior_exists(".") is False
        assert misc.behavior_exists(self.local_session_path_3A) is True

    def test_check_transfer(self):
        misc.check_transfer(self.bad_probe00_folder_3A, self.bad_probe00_folder_3A)
        misc.check_transfer(str(self.bad_probe00_folder_3A), self.bad_probe00_folder_3A)
        with self.assertRaises(AssertionError) as cm:
            misc.check_transfer(self.bad_probe00_folder_3A, self.bad_probe00_folder_3B)
        err = cm.exception
        self.assertEqual(str(err), "src_files != dst_files")

    def test_get_new_filename(self):
        binFname3A = "ignoreThisPart_g0_t0.imec.ap.bin"
        binFname3B = "ignoreThisPart_g0_t0.imec0.ap.bin"
        metaFname3A = "ignoreThisPart_g0_t0.imec.ap.meta"
        metaFname3B = "ignoreThisPart_g0_t0.imec0.ap.meta"
        probe1_3B_Meta = "ignoreThisPart_g0_t0.imec1.ap.meta"
        different_gt = "ignoreThisPart_g1_t2.imec0.ap.meta"

        newfname = misc.get_new_filename(binFname3A)
        self.assertTrue(newfname == "_spikeglx_ephysData_g0_t0.imec.ap.bin")
        newfname = misc.get_new_filename(binFname3B)
        self.assertTrue(newfname == "_spikeglx_ephysData_g0_t0.imec0.ap.bin")
        newfname = misc.get_new_filename(metaFname3A)
        self.assertTrue(newfname == "_spikeglx_ephysData_g0_t0.imec.ap.meta")
        newfname = misc.get_new_filename(metaFname3B)
        self.assertTrue(newfname == "_spikeglx_ephysData_g0_t0.imec0.ap.meta")
        newfname = misc.get_new_filename(probe1_3B_Meta)
        self.assertTrue(newfname == "_spikeglx_ephysData_g0_t0.imec1.ap.meta")
        newfname = misc.get_new_filename(different_gt)
        self.assertTrue(newfname == "_spikeglx_ephysData_g1_t2.imec0.ap.meta")

    def _test_rename_ephys_files(self):
        # Test 3A
        misc.rename_ephys_files(self.local_session_path_3A)
        ap_files = list(self.local_session_path_3A.rglob("*.ap.*"))
        lf_files = list(self.local_session_path_3A.rglob("*.lf.*"))
        self.assertTrue(len(ap_files) >= 1)
        self.assertTrue(len(lf_files) >= 1)
        for f in ap_files:
            print(f.parent.name, f.name)
            self.assertTrue("_spikeglx_ephysData_g" in str(f))
            self.assertTrue("_t" in str(f))
            self.assertTrue(".imec.ap" in str(f))
            self.assertTrue((".bin" in str(f) or ".meta" in str(f)))
            self.assertTrue(("probe00" in str(f) or "probe01" in str(f)))
        for f in lf_files:
            print(f.parent.name, f.name)
            self.assertTrue("_spikeglx_ephysData_g" in str(f))
            self.assertTrue("_t" in str(f))
            self.assertTrue(".imec.lf" in str(f))
            self.assertTrue((".bin" in str(f) or ".meta" in str(f)))  # Test 3B
            self.assertTrue(("probe00" in str(f) or "probe01" in str(f)))
        # Test 3B
        misc.rename_ephys_files(self.local_session_path_3B)
        ap_files = list(self.local_session_path_3B.rglob("*.ap.*"))
        lf_files = list(self.local_session_path_3B.rglob("*.lf.*"))
        nidq_files = list(self.local_session_path_3B.rglob("*.nidq.*"))
        for f in ap_files:
            print(f.name)
            self.assertTrue("_spikeglx_ephysData_g" in str(f))
            self.assertTrue("_t" in str(f))
            self.assertTrue((".imec0.ap" in str(f) or "imec1.ap" in str(f)))
            self.assertTrue((".bin" in str(f) or ".meta" in str(f)))
        for f in lf_files:
            print(f.name)
            self.assertTrue("_spikeglx_ephysData_g" in str(f))
            self.assertTrue("_t" in str(f))
            self.assertTrue((".imec0.lf" in str(f) or "imec1.lf" in str(f)))
            self.assertTrue((".bin" in str(f) or ".meta" in str(f)))  # Test 3B
        for f in nidq_files:
            print(f.name)
            self.assertTrue("_spikeglx_ephysData_g" in str(f))
            self.assertTrue("_t" in str(f))
            self.assertTrue(".nidq" in str(f))
            self.assertTrue((".bin" in str(f) or ".meta" in str(f)))

    def test_rename_and_move(self):
        self._test_rename_ephys_files()
        # Test for 3A
        misc.move_ephys_files(self.local_session_path_3A)
        probe00_files = list(self.probe00_path_3A.rglob("*"))
        probe00_file_names = [x.name for x in probe00_files]
        probe01_files = list(self.probe01_path_3A.rglob("*"))
        probe01_file_names = [x.name for x in probe01_files]

        self.assertTrue(
            "_spikeglx_ephysData_g0_t0.imec.ap.bin" in probe00_file_names
            and "_spikeglx_ephysData_g0_t0.imec.ap.meta" in probe00_file_names
            and "_spikeglx_ephysData_g0_t0.imec.lf.bin" in probe00_file_names
            and "_spikeglx_ephysData_g0_t0.imec.lf.meta" in probe00_file_names
        )
        self.assertTrue(
            "_spikeglx_ephysData_g0_t0.imec.ap.bin" in probe01_file_names
            and "_spikeglx_ephysData_g0_t0.imec.ap.meta" in probe01_file_names
            and "_spikeglx_ephysData_g0_t0.imec.lf.bin" in probe01_file_names
            and "_spikeglx_ephysData_g0_t0.imec.lf.meta" in probe01_file_names
        )
        # Test for 3B
        misc.move_ephys_files(self.local_session_path_3B)
        probe00_files = list(self.probe00_path_3B.rglob("*"))
        probe00_file_names = [x.name for x in probe00_files]
        probe01_files = list(self.probe01_path_3B.rglob("*"))
        probe01_file_names = [x.name for x in probe01_files]
        nidq_files = list(self.raw_ephys_data_path_3B.glob("*.nidq.*"))
        nidq_file_names = [x.name for x in nidq_files]
        self.assertTrue(
            "_spikeglx_ephysData_g0_t0.imec0.ap.bin" in probe00_file_names
            and "_spikeglx_ephysData_g0_t0.imec0.ap.meta" in probe00_file_names
            and "_spikeglx_ephysData_g0_t0.imec0.lf.bin" in probe00_file_names
            and "_spikeglx_ephysData_g0_t0.imec0.lf.meta" in probe00_file_names
        )
        self.assertTrue(
            "_spikeglx_ephysData_g0_t0.imec1.ap.bin" in probe01_file_names
            and "_spikeglx_ephysData_g0_t0.imec1.ap.meta" in probe01_file_names
            and "_spikeglx_ephysData_g0_t0.imec1.lf.bin" in probe01_file_names
            and "_spikeglx_ephysData_g0_t0.imec1.lf.meta" in probe01_file_names
        )
        self.assertTrue(
            "_spikeglx_ephysData_g0_t0.nidq.bin" in nidq_file_names
            and "_spikeglx_ephysData_g0_t0.nidq.meta" in nidq_file_names
        )

    def test_create_ephys_transfer_done_flag(self):
        # Create ephys flag file for completed transfer
        misc.create_ephys_transfer_done_flag(self.local_session_path_3A)
        # Check it was created
        ephys = Path(self.local_session_path_3A).joinpath("ephys_data_transferred.flag")
        self.assertTrue(ephys.exists())
        # Remove it
        ephys.unlink()

    def test_create_video_transfer_done_flag(self):
        # Create video flag file for completed transfer
        misc.create_video_transfer_done_flag(self.local_session_path_3A)
        # Check it was created
        video = Path(self.local_session_path_3A).joinpath("video_data_transferred.flag")
        self.assertTrue(video.exists())
        # Remove it
        video.unlink()

    def test_check_create_raw_session_flag(self):
        from ibllib.tests.fixtures import utils as futils

        raw_session = Path(self.local_session_path_3A).joinpath("raw_session.flag")
        ephys = Path(self.local_session_path_3A).joinpath("ephys_data_transferred.flag")
        video = Path(self.local_session_path_3A).joinpath("video_data_transferred.flag")
        # Add settings file
        fpath = self.local_session_path_3A / "raw_behavior_data" / "_iblrig_taskSettings.raw.json"
        fpath.touch()
        futils.populate_task_settings(
            fpath, patch={"PYBPOD_PROTOCOL": "some_ephysChoiceWorld_task"}
        )
        ""
        # Check not created
        misc.check_create_raw_session_flag(self.local_session_path_3A)
        self.assertFalse(raw_session.exists())
        # Create only ephys flag
        misc.create_ephys_transfer_done_flag(self.local_session_path_3A)
        # Check not created
        misc.check_create_raw_session_flag(self.local_session_path_3A)
        self.assertFalse(raw_session.exists())
        ephys.unlink()
        # Create only video flag
        misc.create_video_transfer_done_flag(self.local_session_path_3A)
        # Check not created
        misc.check_create_raw_session_flag(self.local_session_path_3A)
        self.assertFalse(raw_session.exists())
        video.unlink()
        # Create ephys and video flag file for completed transfer
        misc.create_ephys_transfer_done_flag(self.local_session_path_3A)
        misc.create_video_transfer_done_flag(self.local_session_path_3A)
        # Check it was created
        misc.check_create_raw_session_flag(self.local_session_path_3A)
        self.assertTrue(raw_session.exists())
        # Check other flags deleted
        self.assertFalse(ephys.exists())
        self.assertFalse(video.exists())
        raw_session.unlink()
        # Check if biased session
        futils.populate_task_settings(
            fpath, patch={"PYBPOD_PROTOCOL": "some_biasedChoiceWorld_task"}
        )
        misc.create_video_transfer_done_flag(self.local_session_path_3A)
        misc.check_create_raw_session_flag(self.local_session_path_3A)
        self.assertTrue(raw_session.exists())
        self.assertFalse(video.exists())
        raw_session.unlink()

    def test_create_ephys_flags(self):
        extract = self.local_session_path_3A.joinpath("extract_ephys.flag")
        qc = self.local_session_path_3A.joinpath("raw_ephys_qc.flag")
        spike_sorting0 = self.local_session_path_3A / "raw_ephys_data" / "probe00"
        spike_sorting1 = self.local_session_path_3A / "raw_ephys_data" / "probe01"
        spike_sorting0 = spike_sorting0.joinpath("spike_sorting.flag")
        spike_sorting1 = spike_sorting1.joinpath("spike_sorting.flag")
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

    def test_create_alyx_probe_insertions(self):
        # Connect to test DB
        one = ONE(
            username="test_user",
            password="TapetesBloc18",
            base_url="https://test.alyx.internationalbrainlab.org",
        )
        # Use existing session on test database
        eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"
        # Force probe insertion 3A
        misc.create_alyx_probe_insertions(
            eid, one=one, model="3A", labels=["probe00", "probe01"], force=True
        )
        # Verify it's been inserted
        alyx_insertion = one.alyx.rest("insertions", "list", session=eid)
        alyx_insertion = [x for x in alyx_insertion if x["model"] == "3A"]
        self.assertTrue(alyx_insertion[0]["model"] == "3A")
        self.assertTrue(alyx_insertion[0]["name"] in ["probe00", "probe01"])
        self.assertTrue(alyx_insertion[1]["model"] == "3A")
        self.assertTrue(alyx_insertion[1]["name"] in ["probe00", "probe01"])
        # Cleanup DB
        one.alyx.rest("insertions", "delete", id=alyx_insertion[0]["id"])
        one.alyx.rest("insertions", "delete", id=alyx_insertion[1]["id"])
        # Force probe insertion 3B
        misc.create_alyx_probe_insertions(eid, one=one, model="3B2", labels=["probe00", "probe01"])
        # Verify it's been inserted
        alyx_insertion = one.alyx.rest("insertions", "list", session=eid)
        self.assertTrue(alyx_insertion[0]["model"] == "3B2")
        self.assertTrue(alyx_insertion[0]["name"] in ["probe00", "probe01"])
        self.assertTrue(alyx_insertion[1]["model"] == "3B2")
        self.assertTrue(alyx_insertion[1]["name"] in ["probe00", "probe01"])
        # Cleanup DB
        one.alyx.rest("insertions", "delete", id=alyx_insertion[0]["id"])
        one.alyx.rest("insertions", "delete", id=alyx_insertion[1]["id"])

    def tearDown(self):
        shutil.rmtree(self.root_test_folder, ignore_errors=True)


class TestScanFixPassiveFiles(unittest.TestCase):
    """"""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Session 001 and 002 are 2 sessions with ephys and passive badly transferred
        (
            self.session_path,
            self.passive_session_path,
        ) = fu.create_fake_ephys_recording_bad_passive_transfer_sessions(
            self.tmp_dir.name, lab="fakelab", mouse="fakemouse", date="1900-01-01", num="001"
        )
        # Create another complete ephys session same mouse and date
        self.other_good_session = fu.create_fake_complete_ephys_session(
            self.tmp_dir.name, lab="fakelab", mouse="fakemouse", date="1900-01-01", increment=True
        )

    def test_scan_fix(self):
        from_to_pairs, moved_ok = fix.execute(self.tmp_dir.name, dry=True)
        self.assertTrue(len(from_to_pairs) == 1)
        self.assertTrue(sum(moved_ok) == 0)
        from_to_pairs, moved_ok = fix.execute(self.tmp_dir.name, dry=False)
        self.assertTrue(len(from_to_pairs) == 1)
        self.assertTrue(sum(moved_ok) == 1)
        # Second run Nothing to do
        from_to_pairs, moved_ok = fix.execute(self.tmp_dir.name, dry=True)
        self.assertTrue(len(from_to_pairs) == 0)
        self.assertTrue(sum(moved_ok) == 0)
        from_to_pairs, moved_ok = fix.execute(self.tmp_dir.name, dry=False)
        self.assertTrue(len(from_to_pairs) == 0)
        self.assertTrue(sum(moved_ok) == 0)

    def test_find_pairs(self):
        from_to_pairs = fix.find_pairs(self.tmp_dir.name)
        from_path_parts = ['fakelab', 'Subjects', 'fakemouse', '1900-01-01', '002']
        self.assertTrue(all(x in Path(from_to_pairs[0][0]).parts for x in from_path_parts))
        to_path_parts = ['fakelab', 'Subjects', 'fakemouse', '1900-01-01', '001']
        self.assertTrue(all(x in Path(from_to_pairs[0][1]).parts for x in to_path_parts))

    def test_move_rename_pairs(self):
        # Test all outputs of find function
        from_to_pairs = []
        moved_ok = fix.move_rename_pairs(from_to_pairs)
        self.assertTrue(not moved_ok)
        # Bad paths
        from_to_pairs = [("bla", "ble")]
        moved_ok = fix.move_rename_pairs(from_to_pairs)
        self.assertTrue(sum(moved_ok) == 0)
        # Same as execute
        from_to_pairs = fix.find_pairs(self.tmp_dir.name)
        moved_ok = fix.move_rename_pairs(from_to_pairs)
        self.assertTrue(sum(moved_ok) == 1)

    def tearDown(self):
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

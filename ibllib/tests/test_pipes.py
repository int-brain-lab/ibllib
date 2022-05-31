import json
import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from one.api import ONE

import ibllib.io.extractors.base
import ibllib.pipes.scan_fix_passive_files as fix
import ibllib.tests.fixtures.utils as fu
from ibllib.pipes import misc
from ibllib.tests import TEST_DB


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
            ("ephys_passive_opto", "ephys_passive_opto")
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
            ("_iblrig_tasks_widefieldChoiceWorld", "ephys_biased_opto"),
            ("_iblrig_tasks_ephysChoiceWorld5.1.3", "ephys"),
            ("_iblrig_calibration_frame2TTL4.1.3", None),
            ("_iblrig_tasks_habituationChoiceWorld3.6.0", "habituation"),
            ("_iblrig_tasks_scanningOptoChoiceWorld5.0.0", None),
            ("_iblrig_tasks_RewardChoiceWorld4.1.3", None),
            ("_iblrig_calibration_screen4.1.3", None),
            ("_iblrig_tasks_ephys_certification4.1.3", "sync_ephys"),
            ("passive_opto", "ephys"),
            ("_iblrig_tasks_opto_ephysChoiceWorld", "ephys_biased_opto"),
            ("_iblrig_tasks_opto_biasedChoiceWorld", "biased_opto"),
            # personal projects: Karolina
            ("_iblrig_tasks_optoChoiceWorld", 'biased_opto'),  # legacy not used anymore
            ("optokarolinaChoiceWorld5.34", "biased_opto"),
            ("karolinaChoiceWorld5.34", "biased_opto"),
            ("ephyskarolinaChoiceWorld4.34", "ephys_biased_opto"),
            ("_iblrig_tasks_ksocha_ephysOptoStimulation", "ephys_passive_opto"),
            ("_iblrig_tasks_ksocha_ephysOptoChoiceWorld", "ephys_biased_opto"),
            ("_iblrig_tasks_passiveChoiceWorld", "ephys_replay"),
        ]
        # first test that the function returns expected output
        for to in task_out:
            out = ibllib.io.extractors.base.get_task_extractor_type(to[0])
            assert out == to[1]
        # then check that all task types are represented in the modality choice
        for to in task_out:
            if to[1] is None:
                continue
            assert ibllib.io.extractors.base._get_pipeline_from_task_type(to[1]) is not None


class TestPipesMisc(unittest.TestCase):
    """"""

    def setUp(self):
        # Data emulating local rig data
        self.root_test_folder = tempfile.TemporaryDirectory()

        # Create two rig sessions, one with 3A probe data and one with 3B probe data
        self.session_path_3A = fu.create_fake_session_folder(self.root_test_folder.name)
        # fu.create_fake_raw_behavior_data_folder(self.session_path_3A)
        self.session_path_3B = fu.create_fake_session_folder(self.root_test_folder.name)
        fu.create_fake_raw_behavior_data_folder(self.session_path_3B)

        # Make some files
        fu.populate_raw_spikeglx(self.session_path_3B / 'raw_ephys_data', '3B', n_probes=3)

        ephys_folder = self.session_path_3A / 'raw_ephys_data'
        fu.populate_raw_spikeglx(ephys_folder, '3A', legacy=True, n_probes=1)
        # IBL protocol is for users to copy data to the right probe folder
        shutil.move(ephys_folder.joinpath('raw_ephys_folder'),
                    ephys_folder.joinpath('my_run_probe00'))

    def test_behavior_exists(self):
        self.assertFalse(misc.behavior_exists(self.session_path_3A))
        self.assertTrue(misc.behavior_exists(self.session_path_3B))

    def test_check_transfer(self):
        misc.check_transfer(self.session_path_3A, self.session_path_3A)
        misc.check_transfer(str(self.session_path_3A), self.session_path_3A)
        with self.assertRaises(AssertionError):
            misc.check_transfer(self.session_path_3A, self.session_path_3B)

    def test_get_new_filename(self):
        different_gt = "ignoreThisPart_g1_t2.imec.ap.meta"
        nidaq = 'foobar_g0_t0.nidq.cbin'

        for suf in ('.ap.bin', '0.ap.bin', '1.ap.meta'):
            newfname = misc.get_new_filename(f'ignoreThisPart_g0_t0.imec{suf}')
            self.assertEqual(f'_spikeglx_ephysData_g0_t0.imec{suf}', newfname)
        newfname = misc.get_new_filename(different_gt)
        self.assertEqual('_spikeglx_ephysData_g1_t2.imec.ap.meta', newfname)
        self.assertEqual('_spikeglx_ephysData_g0_t0.nidq.cbin', misc.get_new_filename(nidaq))

        # Test errors
        with self.assertRaises(ValueError):
            misc.get_new_filename('_spikeglx_ephysData_g0_t0.wiring')
        with self.assertRaises(ValueError):
            misc.get_new_filename('_spikeglx_ephysData.meta.cbin')

    def _test_rename_ephys_files(self, path, expected_n):
        """Test SpikeGLX output files are correctly renamed"""
        misc.rename_ephys_files(path)
        n = 0
        for f in path.rglob("*.*.*"):
            if any(x in f.name for x in ('.ap.', '.lf.', '.nidq.')):
                self.assertTrue(f.name.startswith('_spikeglx_ephysData_g'))
                n += 1
        self.assertEqual(expected_n, n)

    def test_rename_and_move(self):
        # Test for 3A
        self._test_rename_ephys_files(self.session_path_3A, 4)
        misc.move_ephys_files(self.session_path_3A)
        probe_folders = list(self.session_path_3A.rglob("*probe*"))
        self.assertTrue(len(probe_folders) == 1 and probe_folders[0].parts[-1] == 'probe00')
        expected = [
            '_spikeglx_ephysData_g0_t0.imec.ap.bin',
            '_spikeglx_ephysData_g0_t0.imec.ap.meta',
            '_spikeglx_ephysData_g0_t0.imec.lf.bin',
            '_spikeglx_ephysData_g0_t0.imec.lf.meta'
        ]
        self.assertCountEqual(expected, [x.name for x in probe_folders[0].glob('*')])

        # Test for 3B
        self._test_rename_ephys_files(self.session_path_3B, 14)
        misc.move_ephys_files(self.session_path_3B)
        probe_folders = sorted(self.session_path_3B.rglob("*probe*"))
        # Check moved into 'probexx' folders
        self.assertTrue(len(probe_folders) == 3)
        self.assertCountEqual((f'probe0{x}' for x in range(3)),
                              [x.parts[-1] for x in probe_folders])
        for i in range(3):
            expected = [
                f'_spikeglx_ephysData_g0_t0.imec{i}.ap.bin',
                f'_spikeglx_ephysData_g0_t0.imec{i}.ap.meta',
                f'_spikeglx_ephysData_g0_t0.imec{i}.lf.bin',
                f'_spikeglx_ephysData_g0_t0.imec{i}.lf.meta'
            ]
            self.assertCountEqual(expected, [x.name for x in probe_folders[i].glob('*')])

        nidq_files = self.session_path_3B.joinpath('raw_ephys_data').glob("*.nidq.*")
        expected = ['_spikeglx_ephysData_g0_t0.nidq.bin', '_spikeglx_ephysData_g0_t0.nidq.meta']
        self.assertCountEqual(expected, [x.name for x in nidq_files])

    def test_create_ephys_transfer_done_flag(self):
        # Create ephys flag file for completed transfer
        misc.create_ephys_transfer_done_flag(self.session_path_3A)
        # Check it was created
        ephys = Path(self.session_path_3A).joinpath("ephys_data_transferred.flag")
        self.assertTrue(ephys.exists())
        # Remove it
        ephys.unlink()

    def test_create_video_transfer_done_flag(self):
        # Create video flag file for completed transfer
        misc.create_video_transfer_done_flag(self.session_path_3A)
        # Check it was created
        video = Path(self.session_path_3A).joinpath("video_data_transferred.flag")
        self.assertTrue(video.exists())
        # Remove it
        video.unlink()

    def test_check_create_raw_session_flag(self):

        raw_session = Path(self.session_path_3B).joinpath("raw_session.flag")
        ephys = Path(self.session_path_3B).joinpath("ephys_data_transferred.flag")
        video = Path(self.session_path_3B).joinpath("video_data_transferred.flag")
        # Add settings file
        fpath = self.session_path_3B / "raw_behavior_data" / "_iblrig_taskSettings.raw.json"
        fu.populate_task_settings(
            fpath, patch={"PYBPOD_PROTOCOL": "some_ephysChoiceWorld_task"}
        )
        # Check not created
        misc.check_create_raw_session_flag(self.session_path_3B)
        self.assertFalse(raw_session.exists())
        # Create only ephys flag
        misc.create_ephys_transfer_done_flag(self.session_path_3B)
        # Check not created
        misc.check_create_raw_session_flag(self.session_path_3B)
        self.assertFalse(raw_session.exists())
        ephys.unlink()
        # Create only video flag
        misc.create_video_transfer_done_flag(self.session_path_3B)
        # Check not created
        misc.check_create_raw_session_flag(self.session_path_3B)
        self.assertFalse(raw_session.exists())
        video.unlink()
        # Create ephys and video flag file for completed transfer
        misc.create_ephys_transfer_done_flag(self.session_path_3B)
        misc.create_video_transfer_done_flag(self.session_path_3B)
        # Check it was created
        misc.check_create_raw_session_flag(self.session_path_3B)
        self.assertTrue(raw_session.exists())
        # Check other flags deleted
        self.assertFalse(ephys.exists())
        self.assertFalse(video.exists())
        raw_session.unlink()
        # Check if biased session
        fu.populate_task_settings(
            fpath, patch={"PYBPOD_PROTOCOL": "some_biasedChoiceWorld_task"}
        )
        misc.create_video_transfer_done_flag(self.session_path_3B)
        misc.check_create_raw_session_flag(self.session_path_3B)
        self.assertTrue(raw_session.exists())
        self.assertFalse(video.exists())
        raw_session.unlink()
        # Check if training session
        fu.populate_task_settings(
            fpath, patch={"PYBPOD_PROTOCOL": "some_trainingChoiceWorld_task"}
        )
        misc.create_video_transfer_done_flag(self.session_path_3B)
        misc.check_create_raw_session_flag(self.session_path_3B)
        self.assertTrue(raw_session.exists())
        self.assertFalse(video.exists())
        raw_session.unlink()
        # Check if habituation session
        fu.populate_task_settings(
            fpath, patch={"PYBPOD_PROTOCOL": "some_habituationChoiceWorld_task"}
        )
        misc.create_video_transfer_done_flag(self.session_path_3B)
        misc.check_create_raw_session_flag(self.session_path_3B)
        self.assertTrue(raw_session.exists())
        self.assertFalse(video.exists())
        raw_session.unlink()

    def test_create_ephys_flags(self):
        extract = self.session_path_3B.joinpath('extract_ephys.flag')
        qc = self.session_path_3B.joinpath('raw_ephys_qc.flag')
        # Create some probe folders for test
        raw_ephys = self.session_path_3B.joinpath('raw_ephys_data')
        probe_dirs = [raw_ephys.joinpath(f'probe{i:02}') for i in range(3)]
        [x.mkdir(exist_ok=True) for x in probe_dirs]
        misc.create_ephys_flags(self.session_path_3B)
        self.assertTrue(extract.exists())
        self.assertTrue(qc.exists())
        self.assertTrue(all(x.joinpath('spike_sorting.flag').exists() for x in probe_dirs))
        # Test recreate
        misc.create_ephys_flags(self.session_path_3B)
        self.assertTrue(extract.exists())
        self.assertTrue(qc.exists())
        self.assertTrue(all(x.joinpath('spike_sorting.flag').exists() for x in probe_dirs))

    def test_create_alyx_probe_insertions(self):
        # Connect to test DB
        one = ONE(**TEST_DB)
        # Use existing session on test database
        eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"
        # Force probe insertion 3A
        misc.create_alyx_probe_insertions(
            eid, one=one, model="3A", labels=["probe00", "probe01"], force=True
        )
        # Verify it's been inserted
        alyx_insertion = one.alyx.rest("insertions", "list", session=eid, no_cache=True)
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
        alyx_insertion = one.alyx.rest("insertions", "list", session=eid, no_cache=True)
        self.assertTrue(alyx_insertion[0]["model"] == "3B2")
        self.assertTrue(alyx_insertion[0]["name"] in ["probe00", "probe01"])
        self.assertTrue(alyx_insertion[1]["model"] == "3B2")
        self.assertTrue(alyx_insertion[1]["name"] in ["probe00", "probe01"])
        # Cleanup DB
        one.alyx.rest("insertions", "delete", id=alyx_insertion[0]["id"])
        one.alyx.rest("insertions", "delete", id=alyx_insertion[1]["id"])

    def test_probe_names_from_session_path(self):
        pnames = ['probe01', 'probe03', 'just_a_probe']

        with tempfile.TemporaryDirectory() as tdir:
            session_path = Path(tdir).joinpath('Algernon', '2021-02-12', '001')
            raw_ephys_path = session_path.joinpath('raw_ephys_data')
            raw_ephys_path.mkdir(parents=True, exist_ok=True)
            raw_ephys_path.joinpath("_spikeglx_ephysData_g0_t0.nidq.meta").touch()
            for pname in pnames:
                probe_path = raw_ephys_path.joinpath(pname)
                probe_path.mkdir()
                probe_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.meta').touch()
                probe_path.joinpath('nested_folder').mkdir()
                probe_path.joinpath('nested_folder', 'toto.ap.meta').touch()
            assert set(misc.probe_labels_from_session_path(session_path)) == set(pnames)

    def test_rename_session(self):
        self._inputs = ('foo', '2020-02-02', '002')
        with mock.patch('builtins.input', self._input_side_effect):
            new_path = misc.rename_session(self.session_path_3B, ask=True)
        self.assertEqual(self.session_path_3B.parents[2].joinpath(*self._inputs), new_path)
        self.assertTrue(new_path.exists())
        # Test assertions
        self._inputs = ('foo', 'May-21', '000')
        with mock.patch('builtins.input', self._input_side_effect):
            with self.assertRaises(AssertionError):
                misc.rename_session(self.session_path_3B, ask=True)
        with self.assertRaises(ValueError):
            misc.rename_session(self.root_test_folder.name)  # Not a valid session path

    def _input_side_effect(self, prompt):
        """input mock function to verify prompts"""
        if 'NAME' in prompt:
            self.assertTrue('fakemouse' in prompt)
            return self._inputs[0]
        elif 'DATE' in prompt:
            self.assertTrue('1900-01-01' in prompt)
            return self._inputs[1]
        else:
            self.assertTrue('002' in prompt)
            return self._inputs[2]

    def tearDown(self):
        self.root_test_folder.cleanup()


class TestSyncData(unittest.TestCase):
    """Tests for the ibllib.pipes.misc.confirm_widefield_remote_folder"""
    raw_widefield = [
        'dorsal_cortex_landmarks.json',
        'fakemouse_SpatialSparrow_19000101_182010.camlog',
        'fakemouse_SpatialSparrow_19000101_182010_2_540_640_uint16-002.dat',
        'snapshots/19000101_190154_1photon.tif'
    ]

    def setUp(self):
        # Data emulating local rig data
        self.root_test_folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.root_test_folder.cleanup)

        # Change location of transfer list
        par_file = Path(self.root_test_folder.name).joinpath('.ibl_local_transfers').as_posix()
        self.patch = unittest.mock.patch('iblutil.io.params.getfile', return_value=par_file)
        self.patch.start()
        self.addCleanup(self.patch.stop)

        self.remote_repo = Path(self.root_test_folder.name).joinpath('remote_repo')
        self.remote_repo.joinpath('fakelab/Subjects').mkdir(parents=True)

        self.local_repo = Path(self.root_test_folder.name).joinpath('local_repo')
        self.local_repo.mkdir()

        self.session_path = fu.create_fake_session_folder(self.local_repo)
        widefield_path = self.session_path.joinpath('raw_widefield_data')
        for file in self.raw_widefield:
            p = widefield_path.joinpath(file)
            p.parent.mkdir(exist_ok=True)
            p.touch()
        # Create video data too
        fu.create_fake_raw_video_data_folder(self.session_path)

    @mock.patch('ibllib.pipes.misc.check_create_raw_session_flag')
    def test_confirm_widefield_remote_folder(self, chk_fcn):
        # NB Mock check_create_raw_session_flag which requires a valid task settings file
        # With no data in the remote repo, no data should be transferred
        with mock.patch('builtins.input', new=self.assertFalse):
            misc.confirm_widefield_remote_folder(self.local_repo, self.remote_repo)
        self.assertFalse(list(filter(lambda x: x.is_file(), self.remote_repo.rglob('*'))))

        # Create a remote session with behaviour data
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch('builtins.input', new=self.assertFalse):
            misc.confirm_widefield_remote_folder(self.local_repo, self.remote_repo)
            chk_fcn.assert_called()

        # Check files were copied
        n_copied = sum(map(lambda x: x.is_file(), remote_session.joinpath('raw_widefield_data').rglob('*')))
        self.assertEqual(n_copied, 4)
        local_transfers_file = Path(self.root_test_folder.name).joinpath('.ibl_local_transfers')
        self.assertTrue(local_transfers_file.exists())
        # Transfers file should be empty as all files transferred successfully
        with open(local_transfers_file) as fp:
            self.assertCountEqual(json.load(fp), [])

        # Delete transferred widefield folder and test user prompt
        shutil.rmtree(remote_session.joinpath('raw_widefield_data'))
        # New session for same date and subject
        new_remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(new_remote_session)
        with mock.patch('builtins.input', return_value='002'):
            misc.confirm_widefield_remote_folder(self.local_repo, self.remote_repo)
        # Data should have been copied into session #2
        n_copied = sum(map(lambda x: x.is_file(), new_remote_session.joinpath('raw_widefield_data').rglob('*')))
        self.assertEqual(n_copied, 4)
        # Local data should have been renamed
        expected = 'fakemouse/1900-01-01/002/raw_widefield_data'
        self.assertTrue(expected in next(self.local_repo.rglob('raw_widefield_data')).as_posix())

        # Test behaviour when a transfer fails
        shutil.rmtree(remote_session)
        with unittest.mock.patch('ibllib.pipes.misc.check_transfer', side_effect=AssertionError), \
                self.assertLogs(logging.getLogger('ibllib'), logging.ERROR):
            misc.confirm_widefield_remote_folder(self.local_repo, self.remote_repo)
        with open(local_transfers_file) as fp:
            transfer_data = json.load(fp)
            self.assertEqual(len(transfer_data), 1)

    def test_rdiff_install(self):
        # remove rdiff-backup for testing
        if os.name == "nt" and Path("C:\\tools\\rdiff-backup.exe").exists():
            # remove executable if on windows
            Path("C:\\tools\\rdiff-backup.exe").unlink()
        else:  # anything not Windows, remove package with pip
            try:
                subprocess.run(["pip", "uninstall", "rdiff-backup", "--yes"], check=True)
            except subprocess.CalledProcessError as e:
                print(e)

        # verify rdiff-backup command is intentionally not functioning anymore
        try:
            subprocess.run(["rdiff-backup", "--version"], check=True)  # should throw FileNotFound
        except FileNotFoundError:
            # call function to have rdiff-backup reinstalled
            self.assertTrue(misc.rdiff_install())
            # assert rdiff-backup command is functioning
            self.assertTrue(subprocess.run(
                ["rdiff-backup", "--version"], capture_output=True).returncode == 0)
        except subprocess.CalledProcessError as e:
            print(e)  # something went wrong with the test

    @mock.patch("ibllib.pipes.misc.check_create_raw_session_flag")
    def test_transfer_video_folders(self, chk_fcn):
        # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        self.session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local session w/o transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/o behavior folder
        self.session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        shutil.rmtree(self.remote_repo / "fakelab" / "Subjects" / "fakemouse" / "1900-01-01" / "001" / "raw_behavior_data")
        # with self.assertLogs(logging.getLogger("ibllib"), logging.WARNING):
        misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        self.session_path.joinpath("transfer_me.flag").unlink()
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/o date folder
        self.session_path.joinpath("transfer_me.flag").touch()
        fu.create_fake_raw_behavior_data_folder(remote_session)
        shutil.rmtree(self.remote_repo / "fakelab" / "Subjects" / "fakemouse")
        misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        self.session_path.joinpath("transfer_me.flag").unlink()
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local sessions w/ transfer_me.flag 1900-01-01, 2 remote sessions w/ raw_behavior_data 1900-01-01
        self.session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        remote_session002 = fu.create_fake_session_folder(self.remote_repo, date="1900-01-01")
        fu.create_fake_raw_behavior_data_folder(remote_session002)
        with mock.patch("builtins.input", side_effect=["002"]):
            misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(self.remote_repo)

        # Test - 2 local sessions w/ transfer_me.flag 1900-01-01, 2 remote sessions w/ raw_behavior_data 1900-01-01
        self.session_path.joinpath("transfer_me.flag").touch()
        local_session002 = fu.create_fake_session_folder(self.local_repo, date="1900-01-01")
        fu.create_fake_raw_video_data_folder(local_session002)
        local_session002.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        remote_session002 = fu.create_fake_session_folder(self.remote_repo, date="1900-01-01")
        fu.create_fake_raw_behavior_data_folder(remote_session002)
        with mock.patch("builtins.input", side_effect=["001", "002"]):
            misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(local_session002)
        shutil.rmtree(self.remote_repo)

        # Test - 2 local sessions w/ transfer_me.flag 1900-01-01, 1 remote sessions w/ raw_behavior_data 1900-01-01
        self.session_path.joinpath("transfer_me.flag").touch()
        local_session002 = fu.create_fake_session_folder(self.local_repo, date="1900-01-01")
        fu.create_fake_raw_video_data_folder(local_session002)
        local_session002.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch("builtins.input", side_effect=["002"]):
            misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(local_session002)
        shutil.rmtree(self.remote_repo)

    @mock.patch("ibllib.pipes.misc.check_create_raw_session_flag")
    def test_rsync_paths(self, chk_fcn):
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        src = self.local_repo / "fakelab/Subjects/fakemouse/1900-01-01/001/raw_video_data"
        dst = self.remote_repo / "fakelab/Subjects/fakemouse/1900-01-01/001/raw_video_data"
        misc.rsync_paths(src, dst)
        # Check files were copied
        n_copied = sum(1 for _ in self.remote_repo.rglob("raw_video_data/*"))
        self.assertEqual(n_copied, 13)

    @mock.patch('ibllib.pipes.misc.check_create_raw_session_flag')
    def test_confirm_video_remote_folder(self, chk_fcn):
        # NB Mock check_create_raw_session_flag which requires a valid task settings file
        # With no data in the remote repo, no data should be transferred
        with mock.patch('builtins.input', new=self.assertFalse):
            misc.confirm_video_remote_folder(self.local_repo, self.remote_repo)
        self.assertFalse(list(filter(lambda x: x.is_file(), self.remote_repo.rglob('*'))))

        # Create a remote session with behaviour data
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        # Create transfer_me flag
        self.session_path.joinpath('transfer_me.flag').touch()
        with mock.patch('builtins.input', new=self.assertFalse):
            misc.confirm_video_remote_folder(self.local_repo, self.remote_repo)
            chk_fcn.assert_called()

        # Check files were copied
        n_copied = sum(1 for _ in self.remote_repo.rglob('raw_video_data/*'))
        self.assertEqual(n_copied, 13)
        local_transfers_file = Path(self.root_test_folder.name).joinpath('.ibl_local_transfers')
        self.assertTrue(local_transfers_file.exists())
        # Transfers file should be empty as all files transferred successfully
        with open(local_transfers_file) as fp:
            self.assertCountEqual(json.load(fp), [])

        # Delete transferred video folder and test user prompt
        shutil.rmtree(remote_session.joinpath('raw_video_data'))
        # New session for same date and subject
        new_remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(new_remote_session)
        self.session_path.joinpath('transfer_me.flag').touch()
        with mock.patch('builtins.input', side_effect=['h', '\n', '002']):
            misc.confirm_video_remote_folder(self.local_repo, self.remote_repo)
        # Data should have been copied into session #2
        n_copied = sum(1 for _ in self.remote_repo.rglob('raw_video_data/*'))
        self.assertEqual(n_copied, 13)
        # Local data should have been renamed
        expected = 'fakemouse/1900-01-01/002/raw_video_data'
        self.assertTrue(expected in next(self.local_repo.rglob('raw_video_data')).as_posix())

        # Test behaviour when a transfer fails
        self.session_path.parent.joinpath('002', 'transfer_me.flag').touch()
        shutil.rmtree(remote_session)
        with unittest.mock.patch('ibllib.pipes.misc.check_transfer', side_effect=AssertionError), \
                self.assertLogs(logging.getLogger('ibllib'), logging.ERROR):
            misc.confirm_video_remote_folder(self.local_repo, self.remote_repo)
        with open(local_transfers_file) as fp:
            transfer_data = json.load(fp)
            self.assertEqual(len(transfer_data), 1)

    def test_backup_session(self):
        # Test when backup path does NOT already exist
        self.assertTrue(misc.backup_session(self.session_path))

        # Test when backup path does exist
        bk_session_path = Path(*self.session_path.parts[:-4]).joinpath(
            "Subjects_backup_renamed_sessions", Path(*self.session_path.parts[-3:]))
        Path(bk_session_path.parent).mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileExistsError):
            misc.backup_session(self.session_path)
        print(">>> Error messages regarding a 'backup session already exists' or a 'given session "
              "path does not exist' is expected in this test. <<< ")

        # Test when a bad session path is given
        self.assertFalse(misc.backup_session("a session path that does NOT exist"))


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


class TestMultiPartsRecordings(unittest.TestCase):

    def test_create_multiparts_flags(self):
        meta_files = [
            "001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.meta",
            "001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.meta",
            "003/raw_ephys_data/probe00/_spikeglx_ephysData_g2_t0.imec0.ap.meta",
            "003/raw_ephys_data/probe01/_spikeglx_ephysData_g2_t0.imec1.ap.meta",
            "002/raw_ephys_data/probe00/_spikeglx_ephysData_g1_t0.imec0.ap.meta",
            "002/raw_ephys_data/probe01/_spikeglx_ephysData_g1_t0.imec1.ap.meta",
            "004/raw_ephys_data/probe00/_spikeglx_ephysData_g3_t0.imec0.ap.meta",
            "004/raw_ephys_data/probe01/_spikeglx_ephysData_g3_t0.imec1.ap.meta"]
        with tempfile.TemporaryDirectory() as tdir:
            root_path = Path(tdir).joinpath('Algernon', '2021-02-12')
            for meta_file in meta_files:
                root_path.joinpath(meta_file).parent.mkdir(parents=True)
                root_path.joinpath(meta_file).touch()
            recordings = misc.multi_parts_flags_creation(root_path)
            for sf in root_path.rglob('*.sequence.json'):
                with open(sf) as fid:
                    d = json.load(fid)
                    assert len(d['files']) == 4
        assert len(recordings['probe00']) == 4
        assert len(recordings['probe01']) == 4


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

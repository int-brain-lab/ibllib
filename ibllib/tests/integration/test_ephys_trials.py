"""Test NI DAQ trials extraction."""
import numpy as np
import numpy.testing

import one.alf.io as alfio
from one.alf.path import get_session_path
from one.api import ONE

from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq, HabituationTrialsNidq
from ibllib.io.extractors.base import run_extractor_classes
from ibllib.io.extractors.biased_trials import ProbaContrasts

from ci.tests import base


class TestEphysTaskExtraction(base.IntegrationTest):
    """Test the FpgaTrials extractor."""

    alf_files = ['_ibl_trials.table.pqt', '_ibl_trials.goCueTrigger_times.npy', '_ibl_trials.quiescencePeriod.npy']
    """A subset of expected output datasets."""

    trials_task = ChoiceWorldTrialsNidq
    """The task to use for trials extraction (may depend on task protocol)."""

    # Expect 'alf', 'raw_ephys_data', 'raw_behavior_data' collections in each.
    required_files = [
        'ephys/ephys_choice_world_task/CSP004/2019-11-27/001/*',  # normal session
        'ephys/ephys_choice_world_task/ibl_witten_13/2019-11-25/001/*',  # FPGA stops before bpod, custom sync
        # 'ephys/ephys_choice_world_task/ibl_witten_27/2021-01-21/001/*'  # frame2ttl flicker
    ]

    # def test_task_extraction_output(self):
    # # TODO This was removed because it appears to be a redundant test;
    #     should check if alf folder still required or can be deleted.
    #     init_folder = self.data_path.joinpath('ephys', 'choice_world_init')
    #     self.sessions = [f.parent for f in init_folder.rglob('raw_ephys_data')]
    #     for session_path in self.sessions:
    #         self._task_extraction_assertions(session_path)

    def test_task_extraction(self):
        """Test ephysChoiceWorld task extraction with NI DAQ.

        Test each session in `required_files` list.
        """
        folders = (next(f, None) for f in map(self.data_path.glob, self.required_files))
        self.sessions = set(filter(None, map(get_session_path, folders)))
        for session_path in self.sessions:
            with self.subTest(msg=session_path.relative_to(self.data_path)):
                self._task_extraction_assertions(session_path)

    def _task_extraction_assertions(self, session_path, trials_task=None):
        """Compare task extraction with expected ALF trials output."""
        self.backup_alf(session_path)
        alf_path = session_path.joinpath('alf')
        bk_path = alf_path.parent / 'alf.bk'

        # Run extractor
        TrialsTask = trials_task or self.trials_task
        task = TrialsTask(session_path,
                          one=ONE(mode='local'), collection='raw_behavior_data',
                          sync_collection='raw_ephys_data')
        fpga_trials, _ = task.extract_behaviour(save=True)
        tqc_ephys = task.run_qc(fpga_trials.copy(), update=False, plot_qc=False)

        # check that the output is complete
        for f in self.alf_files:
            with self.subTest(file=f):
                self.assertTrue(alf_path.joinpath(f).exists())
        # check dimensions after alf load
        alf_trials = alfio.load_object(alf_path, 'trials')
        self.assertTrue(alfio.check_dimensions(alf_trials) == 0)
        # check new trials table the same as old individual datasets
        # in the future if extraction changes this test can be removed
        if any(bk_path.glob('*trials*')):
            alf_trials_old = alfio.load_object(alf_path.parent / 'alf.bk', 'trials')
            for k, v in alf_trials.items():
                if k in alf_trials_old:  # added quiescencePeriod from the old dataset
                    with self.subTest(attribute=k, session='/'.join(session_path.parts[-3:])):
                        numpy.testing.assert_array_almost_equal(v, alf_trials_old[k])
        # go deeper and check the internal fpga trials structure consistency
        fpga_trials = {k: v for k, v in fpga_trials.items() if 'wheel' not in k}
        # check dimensions
        self.assertEqual(alfio.check_dimensions(fpga_trials), 0)
        # check some task-specific trial events
        self._check_task_trial_events(fpga_trials)

        # compute the task qc
        _, res_ephys = tqc_ephys.run(bpod_only=False, download_data=False)

        TaskQC = type(tqc_ephys)  # Use the same TaskQC class as the task for the Bpod only QC
        tqc_bpod = TaskQC(session_path, one=ONE(mode='local'))
        tqc_bpod.extractor = TaskQCExtractor(session_path)
        tqc_bpod.extractor.settings = task.extractor.settings
        tqc_bpod.extractor.data = tqc_bpod.extractor.rename_data(task.extractor.bpod_trials.copy())
        tqc_bpod.extractor.frame_ttls = task.extractor.bpod_extractor.frame2ttl  # used in iblapps QC viewer
        tqc_bpod.extractor.audio_ttls = task.extractor.bpod_extractor.audio  # used in iblapps QC viewer

        _, res_bpod = tqc_bpod.run(bpod_only=True, download_data=False)

        # for a swift comparison using variable explorer
        # import pandas as pd
        # df = pd.DataFrame([[res_bpod[k], res_ephys[k]] for k in res_ephys], index=res_ephys.keys())

        for k in res_ephys:
            if k == '_task_response_feedback_delays':
                continue  # FIXME explain why this check is skipped
            with self.subTest(check=k):
                check_diff = (res_bpod[k] or 0) - (res_ephys[k] or 0)
                self.assertFalse(
                    np.abs(check_diff) > .2, f'{k} bpod: {res_bpod[k]}, ephys: {res_ephys[k]}'
                )

    def _check_task_trial_events(self, trials):
        """Check task-specific trial events."""
        # check that the stimOn < stimFreeze < stimOff
        self.assertTrue(np.less(trials['table']['stimOn_times'], trials['stimOff_times']).all())
        nogo = trials['table']['choice'] == 0
        self.assertTrue(np.less(trials['stimFreeze_times'][~nogo], trials['stimOff_times'][~nogo]).all())
        # a trial is either an error-nogo or a reward
        self.assertTrue(np.all(np.isnan(trials['valveOpen_times'] * trials['errorCue_times'])))
        self.assertTrue(np.all(np.logical_xor(np.isnan(trials['valveOpen_times']),
                                              np.isnan(trials['errorCue_times']))))


class TestEphysHabituationTaskExtraction(TestEphysTaskExtraction):
    """Test the FpgaTrialsHabituation extractor."""

    # Expect 'alf', 'raw_ephys_data', 'raw_behavior_data' collections in each.
    required_files = [
        'ephys/habituation_choice_world_task/MM015/2023-10-05/002/*',  # normal session
    ]

    trials_task = HabituationTrialsNidq

    alf_files = ['_ibl_trials.intervals.npy',
                 '_ibl_trials.stimCenter_times.npy',
                 '_ibl_trials.goCue_times.npy',
                 '_ibl_trials.feedback_times.npy',
                 '_ibl_trials.rewardVolume.npy']
    """A subset of expected output datasets."""

    def _check_task_trial_events(self, trials):
        """Check habituationChoiceWorld specific trial events.

        Check that the stimOn < stimCenter < stimOff.
        Note that we don't test the first trial as the stimulus on the first trial is messed up!
        """
        self.assertTrue(
            np.less(trials['stimOn_times'][1:], trials['stimOff_times'][1:]).all())
        self.assertTrue(
            np.less(trials['stimCenter_times'][1:], trials['stimOff_times'][1:]).all())


class TestEphysTrialsFPGA(base.IntegrationTest):
    trials_task = ChoiceWorldTrialsNidq
    """The task to use for trials extraction (may depend on task protocol)."""

    required_files = [
        'ephys/ephys_choice_world_task/ibl_witten_27/2021-01-21/001/*',
        'ephys/ephys_choice_world_task/ibl_witten_13/2019-11-25/001/*',  # FPGA stops before bpod, custom sync
    ]

    def test_frame2ttl_flicker(self):
        session_path = get_session_path(next(self.data_path.glob(self.required_files[0])))
        # dsets, out_files = ephys_fpga.extract_all(session_path, save=True)
        task = self.trials_task(session_path,
                                one=ONE(mode='local'), collection='raw_behavior_data',
                                sync_collection='raw_ephys_data')
        fpga_trials, _ = task.extract_behaviour(save=True)
        # Run the task QC
        qc = task.run_qc(fpga_trials, update=False, plot_qc=False)
        # Aggregate and update Alyx QC fields
        _, myqc, _ = qc.compute_session_status()

        # from ibllib.misc import pprint
        # pprint(myqc)
        assert myqc['_task_stimOn_delays'] > 0.9  # 0.6176
        assert myqc['_task_stimFreeze_delays'] > 0.9
        assert myqc['_task_stimOff_delays'] > 0.9
        assert myqc['_task_stimOff_itiIn_delays'] > 0.9
        assert myqc['_task_stimOn_delays'] > 0.9
        assert myqc['_task_stimOn_goCue_delays'] > 0.9

    def test_missing_first_trial(self):
        """Test extraction when FPGA starts before Bpod.

        In some BWM sessions the FPGA starts during the first trial, such that the first Bpod trial start
        pulse is not recorded by the FPGA. This results in the first trial being missing from the FPGA with
        the added complication of the first trial being a special case in the task: the first Bpod pulse is
        longer than usual to trigger the cameras and therefore looks closer to a valve open pulse.
        """
        # The first trial is +ve feedback (valve open) so valve open pulse re-reassignment should take place
        session_path = get_session_path(next(self.data_path.glob(self.required_files[1])))

        task = self.trials_task(
            session_path, one=ONE(mode='local'), collection='raw_behavior_data', sync_collection='raw_ephys_data')
        with self.assertLogs('ibllib.io.extractors.ephys_fpga', level='DEBUG') as cm:
            fpga_trials, _ = task.extract_behaviour(save=False, tmin=305., tmax=5500.)
            msg = 'Re-reassigning first valve open event'
            self.assertTrue(any(msg in lg for lg in cm.output))
        trials = fpga_trials['table']
        # Previously the first valve open was re-assigned to the first trial and feedback_time was NaN
        np.testing.assert_almost_equal(trials['feedback_times'][0], 305.9525)
        np.testing.assert_almost_equal(trials['intervals_0'].iloc[0], 298.33821117616856)
        bpod_intervals = task.extractor.bpod2fpga(task.extractor.bpod_trials['intervals'][:2, :])
        fpga_intervals = trials[['intervals_0', 'intervals_1']].values[:2]
        np.testing.assert_array_almost_equal(bpod_intervals, fpga_intervals, decimal=4)

        # Test extraction when first trial start and first valve pulses are cut off
        # Here we'd expect the first trial end to be assigned as trial start and the
        # extractor should undo this after accounting for missing trial start TTL
        sync, chmap = task.extractor.load_sync('raw_ephys_data')
        selection = np.logical_and(sync['times'] <= 5500., sync['times'] >= 306.5)
        sync = alfio.AlfBunch({k: v[selection] for k, v in sync.items()})
        with self.assertLogs('ibllib.io.extractors.ephys_fpga', level='DEBUG') as cm:
            # Call build_trials directly to avoid needless re-extraction of Bpod data
            trials = task.extractor.build_trials(sync=sync, chmap=chmap)
            msg = 'Re-reassigning first trial end event'
            self.assertTrue(any(msg in lg for lg in cm.output))
        self.assertTrue(np.isnan(trials['feedback_times'][0]))
        np.testing.assert_almost_equal(trials['intervals'][0, 0], 298.33821117616856)
        fpga_intervals = trials['intervals'][:2]
        np.testing.assert_array_almost_equal(bpod_intervals, fpga_intervals, decimal=4)

        # Finally lets cut off several of the first trials and assert that trial lengths
        # remain consistent with Bpod
        sync, chmap = task.extractor.load_sync('raw_ephys_data')
        selection = np.logical_and(sync['times'] <= 5500., sync['times'] >= 350.)
        sync = alfio.AlfBunch({k: v[selection] for k, v in sync.items()})
        trials = task.extractor.build_trials(sync=sync, chmap=chmap)
        bpod_intervals = task.extractor.bpod2fpga(task.extractor.bpod_trials['intervals'][:2, :])
        fpga_intervals = trials['intervals'][:2]
        self.assertEqual(trials['feedbackType'].size, trials['itiIn_times'].size)
        self.assertEqual(trials['itiIn_times'].size, task.extractor.bpod_trials['intervals'].shape[0])
        np.testing.assert_array_almost_equal(bpod_intervals, fpga_intervals, decimal=4)


class TestEphysTrials_iblrigv8(base.IntegrationTest):

    required_files = [
        'ephys/ephys_choice_world_task/KM_014/2024-05-02/001/*',  # iblrig v8 ephys session
        'ephys/ephys_choice_world_task/CSP004/2019-11-27/001/*'  # non iblrig v8 ephys session
    ]

    def test_contrast_extraction_v8(self):
        session_path = get_session_path(next(self.data_path.glob(self.required_files[0])))

        # Extract trials using task, should be using TrialsTableBiased which does not contain the
        # extractor ProbaContrasts and instead gets ContrastLeft, ContrastRight and ProbabilityLeft from
        # the bpod data itself
        task = ChoiceWorldTrialsNidq(session_path, one=ONE(mode='local'), collection='raw_task_data_00',
                                     sync_collection='raw_ephys_data')
        fpga_trials, _ = task.extract_behaviour(save=False)
        trials = fpga_trials['table']

        # Extract the using the pregenerated session extractor
        out, _ = run_extractor_classes(
            ProbaContrasts, session_path=task.session_path, save=False, task_collection=task.collection)

        # Ensure the values are not the same
        for col in ['contrastLeft', 'contrastRight', 'probabilityLeft']:
            self.assertFalse(np.array_equal(trials[col].values, out[col], equal_nan=True))

    def test_contrast_extraction_lt_v8(self):
        session_path = get_session_path(next(self.data_path.glob(self.required_files[1])))

        # Extract trials using task, should be using TrialsTableEphys which uses the pregenerated session number and
        # extractor ProbaContrasts to get ContrastLeft, ContrastRight and ProbabilityLeft
        task = ChoiceWorldTrialsNidq(session_path, one=ONE(mode='local'), collection='raw_behavior_data',
                                     sync_collection='raw_ephys_data')
        fpga_trials, _ = task.extract_behaviour(save=False)
        trials = fpga_trials['table']

        out, _ = run_extractor_classes(
            ProbaContrasts, session_path=task.session_path, save=False, task_collection=task.collection)

        # Ensure the values are the same
        for col in ['contrastLeft', 'contrastRight', 'probabilityLeft']:
            self.assertTrue(np.array_equal(trials[col].values, out[col], equal_nan=True))


class TestEphysTrials_new_1st_trial(base.IntegrationTest):

    required_files = ['ephys/ephys_1st_trial/2025-05-28/001/*']

    def test_new_1st_trial(self):
        """Starting with version 8.28.0 of IBLRIG, the first trial of the state machine is no longer used for timing
        the initial delay and waiting for the first camera trigger. Instead, the first trial is now identical to the
        remaining trials. For this reason, extractors can now skip some logic for recovering from edge cases specific
        to the extraction of the first trial."""
        session_path = get_session_path(next(self.data_path.glob(self.required_files[0])))

        task = ChoiceWorldTrialsNidq(session_path, one=ONE(mode='local'), collection='raw_task_data_00',
                                     sync_collection='raw_ephys_data')
        fpga_trials, _ = task.extract_behaviour(save=False)
        trials = fpga_trials['table']

        # times from Bpod represent ground truth
        bpod_trials = task.extractor.bpod_extractor.bpod_trials
        bpod_trial_start = [t['behavior_data']['States timestamps']['trial_start'][0][0] for t in bpod_trials]

        # compare Bpod and FPGA trial start times
        times_bpod = np.array(bpod_trial_start) - bpod_trial_start[0]
        times_fpga = trials.intervals_0 - trials.intervals_0[0]
        self.assertTrue(np.isclose(times_bpod, times_fpga, atol=0.0001).all())

        # compare Bpod and FPGA trial durations
        diff_bpod = np.diff(bpod_trial_start)
        diff_fpga = np.diff(trials.intervals_0)
        self.assertTrue(np.isclose(diff_bpod, diff_fpga, atol=0.0001).all())

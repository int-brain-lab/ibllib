"""Tests TaskQC object and Bpod extractors.

NB: FPGA TaskQC extractor is tested in test_ephys_extraction_choiceWorld.

This module uses ZM_1150/2019-05-07/001 ('b1c968ad-4874-468d-b2e4-5ffa9b9964e9').
"""
import unittest.mock

from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsBpod
from one.api import ONE
from one.alf import spec
from ci.tests import base

one = ONE(**base.TEST_DB)


class TestTaskQCObject(base.IntegrationTest):

    required_files = ['training/ZM_1150/2019-05-07/001']

    @classmethod
    def setUpClass(cls):
        cls.one = one
        cls.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        # Make sure the data exists locally
        data_path = base.IntegrationTest.default_data_root()
        cls.session_path = data_path.joinpath('training', 'ZM_1150', '2019-05-07', '001')
        task = ChoiceWorldTrialsBpod(cls.session_path, collection='raw_behavior_data', one=one)
        cls.qc = task.run_qc(update=False)
        assert cls.qc.one is one

    def test_compute(self):
        # Compute metrics
        self.qc.metrics = self.qc.passed = None
        self.qc.compute()
        self.assertTrue(self.qc.metrics is not None)
        self.assertTrue(self.qc.passed is not None)

    def test_run(self):
        # Reset Alyx fields before test
        assert 'test' in self.qc.one.alyx.base_url
        reset = self.qc.update(spec.QC.NOT_SET, override=True)
        assert reset == spec.QC.NOT_SET, 'failed to reset QC field for test'
        extended = self.one.alyx.json_field_write('sessions', field_name='extended_qc',
                                                  uuid=self.eid, data={})
        assert not extended, 'failed to reset extended QC field for test'

        # Test update as False
        outcome, _ = self.qc.run(update=False)
        self.assertEqual(spec.QC.FAIL, outcome)
        extended = self.one.alyx.rest('sessions', 'read',
                                      id=self.eid, no_cache=True)['extended_qc']
        self.assertDictEqual({}, extended, 'unexpected update to extended qc')
        outcome = self.one.alyx.rest('sessions', 'read', id=self.eid, no_cache=True)['qc']
        self.assertEqual(spec.QC.NOT_SET.name, outcome, 'unexpected update to qc')

        # Test update as True
        outcome, results = self.qc.run(update=True)
        self.assertEqual(spec.QC.FAIL, outcome)
        extended = self.one.alyx.rest('sessions', 'read',
                                      id=self.eid, no_cache=True)['extended_qc']
        expected = list(results.keys()) + ['task']
        self.assertCountEqual(expected, extended.keys(), 'unexpected update to extended qc')
        qc_field = self.one.alyx.rest('sessions', 'read', id=self.eid, no_cache=True)['qc']
        self.assertEqual(outcome.name, qc_field, 'unexpected update to qc')

    def test_compute_session_status(self):
        self.qc.metrics = self.qc.passed = None
        with self.assertRaises(AttributeError):
            self.qc.compute_session_status()
        self.qc.compute()
        outcome, results, outcomes = self.qc.compute_session_status()
        self.assertEqual(spec.QC.FAIL, outcome)

        # Check each outcome matches...
        expected_outcomes = {
            '_task_audio_pre_trial': spec.QC.PASS,
            '_task_correct_trial_event_sequence': spec.QC.FAIL,
            '_task_detected_wheel_moves': spec.QC.WARNING,
            '_task_errorCue_delays': spec.QC.WARNING,
            '_task_error_trial_event_sequence': spec.QC.FAIL,
            '_task_goCue_delays': spec.QC.WARNING,
            '_task_iti_delays': spec.QC.NOT_SET,
            '_task_n_trial_events': spec.QC.FAIL,
            '_task_negative_feedback_stimOff_delays': spec.QC.WARNING,
            '_task_positive_feedback_stimOff_delays': spec.QC.WARNING,
            '_task_response_feedback_delays': spec.QC.FAIL,
            '_task_response_stimFreeze_delays': spec.QC.WARNING,
            '_task_reward_volume_set': spec.QC.PASS,
            '_task_reward_volumes': spec.QC.PASS,
            '_task_stimFreeze_delays': spec.QC.WARNING,
            '_task_stimOff_delays': spec.QC.WARNING,
            '_task_stimOff_itiIn_delays': spec.QC.WARNING,
            '_task_stimOn_delays': spec.QC.WARNING,
            '_task_stimOn_goCue_delays': spec.QC.FAIL,
            '_task_stimulus_move_before_goCue': spec.QC.NOT_SET,
            '_task_trial_length': spec.QC.WARNING,
            '_task_wheel_freeze_during_quiescence': spec.QC.PASS,
            '_task_wheel_integrity': spec.QC.PASS,
            '_task_wheel_move_before_feedback': spec.QC.PASS,
            '_task_wheel_move_during_closed_loop': spec.QC.PASS,
            '_task_wheel_move_during_closed_loop_bpod': spec.QC.PASS,
            '_task_passed_trial_checks': spec.QC.NOT_SET
        }
        for k in outcomes:
            with self.subTest(check=k[6:].replace('_', ' ')):
                self.assertEqual(expected_outcomes[k], outcomes[k], f'{k} should be {expected_outcomes[k]}')


if __name__ == '__main__':
    unittest.main(exit=False)

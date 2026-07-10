"""Tests for habituation pipeline tasks.

NB: For ibllib.pipes.behavior_tasks.HabituationTrialsNidq tests see tests.test_ephys_trials.
"""
import logging
import ibllib.pipes.behavior_tasks as btasks
from one.api import ONE

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class HabituationTemplate(base.IntegrationTest):
    required_files = ['tasks/choice_world_habituation/steinmetzlab/Subjects/NR_0020/2022-01-27/001']

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB, mode='local')
        self.session_path = self.data_path.joinpath(self.required_files[0])


class TestHabituationRegisterRaw(HabituationTemplate):

    def test_task(self):
        wf = btasks.HabituationRegisterRaw(self.session_path, one=self.one, collection='raw_behavior_data')
        status = wf.run()
        assert status == 0
        wf.assert_expected_outputs()
        wf.assert_expected_inputs()


class TestHabituationTrialsBpod(HabituationTemplate):

    def test_task(self):
        wf = btasks.HabituationTrialsBpod(self.session_path, one=self.one, collection='raw_behavior_data', save=True)
        status = wf.run(update=False)
        assert status == 0
        wf.assert_expected_outputs()

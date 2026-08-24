import logging
import ibllib.pipes.behavior_tasks as btasks
from one.api import ONE

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class BiasedTemplate(base.IntegrationTest):
    required_files = ['tasks/choice_world_biased/steinmetzlab/Subjects/NR_0020/2022-03-14/001']

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB, mode='local')
        self.session_path = self.data_path.joinpath(self.required_files[0])


class TestBiasedTrialsBpod(BiasedTemplate):
    def test_task(self):
        wf = btasks.ChoiceWorldTrialsBpod(self.session_path, one=self.one, collection='raw_behavior_data')
        status = wf.run(update=False)
        assert status == 0
        wf.assert_expected_outputs()
        wf.assert_expected_inputs()


class TestTrialRegisterRaw(BiasedTemplate):
    def test_task(self):
        wf = btasks.TrialRegisterRaw(self.session_path, one=self.one, collection='raw_behavior_data')
        status = wf.run()
        assert status == 0
        wf.assert_expected_outputs()
        wf.assert_expected_inputs()

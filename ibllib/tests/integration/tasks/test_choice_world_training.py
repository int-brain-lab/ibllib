import logging
import shutil
import ibllib.pipes.behavior_tasks as btasks
from one.api import ONE

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class TrainingTemplate(base.IntegrationTest):
    required_files = ['tasks/choice_world_training/steinmetzlab/Subjects/NR_0020/2022-01-28/001']

    def setUp(self) -> None:
        super().setUp()
        self.one = ONE(**base.TEST_DB, mode='local')
        self.session_path = self.data_path.joinpath(self.required_files[0])


class TestTrainingTrialsBpod(TrainingTemplate):

    def test_task(self):
        wf = btasks.ChoiceWorldTrialsBpod(self.session_path, one=self.one, collection='raw_behavior_data')
        status = wf.run(update=False)
        assert status == 0
        wf.assert_expected_outputs()
        wf.assert_expected_inputs()


class TestTrialRegisterRaw(TrainingTemplate):

    def test_task(self):
        wf = btasks.TrialRegisterRaw(self.session_path, one=self.one, collection='raw_behavior_data')
        status = wf.run()
        assert status == 0
        wf.assert_expected_outputs()


class TestTrainingTrialsBpodSavePath(TrainingTemplate):

    def test_task(self):
        shutil.move(self.session_path.joinpath('raw_behavior_data'), self.session_path.joinpath('raw_lala_data'))
        wf = btasks.ChoiceWorldTrialsBpod(self.session_path, one=self.one, collection='raw_lala_data')
        # force output collection
        wf.output_collection = 'alf/task00'
        self.assertIsNone(wf.protocol_number)
        status = wf.run(update=False)
        assert status == 0
        wf.assert_expected_outputs()
        wf.assert_expected_inputs()
        self.assertTrue(wf.outputs[0].parent, self.session_path.joinpath(wf.output_collection))

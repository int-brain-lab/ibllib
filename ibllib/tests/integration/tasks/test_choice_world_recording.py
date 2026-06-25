import logging
import shutil
import ibllib.pipes.behavior_tasks as btasks
from one.api import ONE

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class RecordingTemplate(base.IntegrationTest):

    required_files = ['tasks/choice_world_ephys/steinmetzlab/Subjects/NR_0020/2022-05-12/001']

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB, mode='local')
        self.session_path = self.data_path.joinpath(self.required_files[0])


class TestTrainingTrialsRecording(RecordingTemplate):

    def test_task(self):
        wf = btasks.ChoiceWorldTrialsNidq(self.session_path, one=self.one, collection='raw_behavior_data',
                                          sync_namespace='spikeglx', sync_collection='raw_ephys_data')
        status = wf.run(update=False, plot_qc=False)
        self.assertEqual(0, status)
        wf.assert_expected_outputs()
        wf.assert_expected_inputs()


class TestTrialRegisterRaw(RecordingTemplate):

    def test_task(self):
        wf = btasks.TrialRegisterRaw(self.session_path, one=self.one, collection='raw_behavior_data')
        status = wf.run()
        self.assertEqual(0, status)
        wf.assert_expected_outputs()

import logging
import shutil
import tempfile
from pathlib import Path

from one.api import ONE
from ibllib.pipes.audio_tasks import AudioCompress, AudioSync

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class TestAudioCompress(base.IntegrationTest):
    def setUp(self) -> None:
        self.data_path = self.default_data_root().joinpath('ephys', 'ephys_video_init', 'ZM_1735', '2019-08-01', '001', 'raw_behavior_data')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1735', '2019-08-01', '001')
        shutil.copytree(self.data_path, self.session_path.joinpath('raw_behavior_data'))
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_compress(self):
        task = AudioCompress(self.session_path, one=self.one, device_collection='raw_behavior_data')
        status = task.run()
        assert status == 0

        self.assertIsNone(next(self.session_path.rglob('*.wav'), None))
        self.assertTrue(next(self.session_path.rglob('*.flac')) == task.outputs[0])

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestAudioSync(base.IntegrationTest):

    def setUp(self) -> None:
        self.data_path = self.default_data_root().joinpath('ephys', 'ephys_video_init', 'ZM_1735', '2019-08-01', '001', 'raw_behavior_data')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1735', '2019-08-01', '001')
        shutil.copytree(self.data_path, self.session_path.joinpath('raw_behavior_data'))
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_audiosync(self):
        task = AudioSync(
            self.session_path, one=self.one, device_collection='raw_behavior_data', collection='raw_behavior_data', sync='bpod')
        status = task.run()
        assert status == 0

        self.assertIsNone(next(self.session_path.rglob('*.wav'), None))

    def test_audiosync_fpga(self):
        task = AudioSync(
            self.session_path, one=self.one, device_collection='raw_behavior_data', collection='raw_behavior_data', sync='fpga')
        status = task.run()
        assert status == 0
        assert task.outputs is None
        print(task.log)
        assert 'Audio Syncing not yet implemented for FPGA' in task.log

    def check_files(self, task):
        for exp_files in task.signature['output_files']:
            file = next(self.session_path.joinpath(exp_files[1]).glob(exp_files[0]), None)
            assert file.exists()
            assert file in task.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

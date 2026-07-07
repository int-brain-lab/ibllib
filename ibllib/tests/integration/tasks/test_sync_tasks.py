import logging
import shutil
import tempfile
from pathlib import Path

from one.api import ONE
from ibllib.pipes.sync_tasks import SyncRegisterRaw, SyncMtscomp, SyncPulses

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class SyncTemplate(base.IntegrationTest):
    required_files = ['widefield/widefieldChoiceWorld/JC076/2022-02-04/001']
    _writable_scope = 'test'

    def setUp(self) -> None:
        super().setUp()
        self.session_path = self.data_path.joinpath(self.required_files[0])
        self.widefield_path = self.session_path.joinpath('raw_widefield_data')
        self.kwargs = dict(sync_collection='raw_widefield_data', sync='nidq', sync_namespace='spikeglx', one=ONE(**base.TEST_DB))

    def copy_folder(self, folder):
        shutil.copytree(self.session_path.joinpath(folder), self.widefield_path)

    def check_files(self, task, ignore_ext='blablabla'):
        for exp_files in task.signature['output_files']:
            if ignore_ext not in exp_files[0]:
                file = next(self.session_path.joinpath(exp_files[1]).glob(exp_files[0]), None)
                assert file.exists()
                assert file in task.outputs


class TestSyncRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.session_path = Path(self._tempdir.name).joinpath('JC076', 'test_date', 'test_sess')
        self.sync_collection = 'raw_device_collection'
        self.sync = 'random'
        self.sync_ext = 'tdms'
        self.sync_namespace = 'pluto'

        self.sync_path = self.session_path.joinpath(self.sync_collection)
        self.sync_path.mkdir(exist_ok=True, parents=True)
        self.daq_file = self.sync_path.joinpath(f'_{self.sync_namespace}_DAQdata.raw.{self.sync_ext}')
        self.wiring_file = self.sync_path.joinpath(f'_{self.sync_namespace}_DAQdata.wiring.json')
        self.daq_file.touch()
        self.wiring_file.touch()

    def test_register(self):
        task = SyncRegisterRaw(self.session_path, sync_collection=self.sync_collection, sync=self.sync, sync_ext=self.sync_ext,
                               sync_namespace=self.sync_namespace, one=ONE(**base.TEST_DB))
        status = task.run()

        self.assertEqual(status, 0)
        self.assertIn(self.daq_file, task.outputs)
        self.assertIn(self.wiring_file, task.outputs)

    def tearDown(self) -> None:
        self._tempdir.cleanup()


class TestSyncMtscomp(SyncTemplate):

    def test_rename_and_compress(self):
        self.copy_folder('rename_compress')
        task = SyncMtscomp(self.session_path, **self.kwargs)
        status = task.run()
        self.assertEqual(status, 0)
        self.check_files(task)

    def test_rename(self):
        self.copy_folder('rename')
        task = SyncMtscomp(self.session_path, **self.kwargs)
        status = task.run()
        self.assertEqual(status, 0)
        self.check_files(task)

    def test_compress(self):
        self.copy_folder('compress')
        task = SyncMtscomp(self.session_path, **self.kwargs)
        status = task.run()
        self.assertEqual(status, 0)
        self.check_files(task)

    def test_register(self):
        self.copy_folder('register')
        task = SyncMtscomp(self.session_path, **self.kwargs)
        status = task.run()
        self.assertEqual(status, 0)
        # Here we don't expect the .cbin file
        self.check_files(task, ignore_ext='.cbin')


class TestSyncPulses(SyncTemplate):

    def test_extract_pulses_bin(self):
        self.copy_folder('compress')
        task = SyncPulses(self.session_path, **self.kwargs)
        status = task.run()
        self.assertEqual(status, 0)
        self.check_files(task)

    def test_extract_pulses_cbin(self):
        self.copy_folder('compress')
        task = SyncMtscomp(self.session_path, **self.kwargs)
        task.run()

        task = SyncPulses(self.session_path, **self.kwargs)
        status = task.run()
        self.assertEqual(status, 0)
        self.check_files(task)

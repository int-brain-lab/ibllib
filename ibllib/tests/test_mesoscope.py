import unittest
import tempfile
import json
from pathlib import Path
from ibllib.pipes.mesoscope_tasks import MesoscopePreprocess


class TestMesoscopePreprocess(unittest.TestCase):

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath('test', '2023-01-31', '001')
        self.img_path = self.session_path.joinpath('raw_imaging_data')
        self.img_path.mkdir(parents=True)
        self.task = MesoscopePreprocess(self.session_path)

    def test_rename_outputs(self):
        

    def test_defaults(self):
        defaults = {
            'data_path': [str(self.img_path)],
            'save_path0': str(self.session_path.joinpath('alf')),
            'move_bin': True,
            'keep_movie_raw': False,
            'delete_bin': False,
            'batch_size': 1000,
            'combined': True
        }
        missing_keys = ['nrois', 'mesoscan', 'nplanes', 'nchannels', 'tau', 'fs', 'dx', 'dy', 'lines']
        with open(self.img_path.joinpath('rawImagingData.meta.json'), 'w') as f:
            json.dump({}, f)
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', level='WARNING') as capture:
            status = self.task.run(run_suite2p=False, rename_files=False)
            self.assertEqual(len(capture.records), len(missing_keys))
            for i in range(len(missing_keys)):
                self.assertEqual(capture.records[i].getMessage(),
                                 f"Setting for {missing_keys[i]} not found in metadata file. Keeping default.")
        self.assertEqual(status, 0)
        self.assertCountEqual(self.task.kwargs, defaults)

    def test_meta(self):
        # test stuff that should be replaced by meta file (write these to meta file first)
        pass

    def test_kwargs(self):
        # test that passing kwargs overwrites the defaults and meta options
        pass

    def tearDown(self) -> None:
        self.td.cleanup()


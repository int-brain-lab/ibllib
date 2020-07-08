import unittest
from pathlib import Path
import tempfile
import shutil

from oneibl.one import ONE

class TestOneOffline(unittest.TestCase):

    def test_one_offline(self) -> None:
        # init: create a temp directory and copy the fixtures
        init_cache_file = Path(__file__).parent.joinpath('fixtures', '.one_cache.parquet')
        td = tempfile.TemporaryDirectory()
        cache_dir = Path(td.name)
        shutil.copyfile(init_cache_file, cache_dir.joinpath(init_cache_file.name))

        # instantiate the one offline object
        one = ONE(offline=True, cache_dir=cache_dir)

        # test the constructor
        self.assertTrue(one._cache.shape[1] == 13)

        # test the load with download false so it returns only file paths
        eid = '4b00df29-3769-43be-bb40-128b1cba6d35'
        dtypes = ['_spikeglx_sync.channels',
                  '_spikeglx_sync.polarities',
                  '_spikeglx_sync.times',
                  '_iblrig_taskData.raw',
                  '_iblrig_taskSettings.raw',
                  'ephysData.raw.meta',
                  'camera.times',
                  'ephysData.raw.wiring']
        files = one.load(eid, dataset_types=dtypes, dclass_output=False, download_only=True,
                         offline=True)
        self.assertTrue(one._cache.shape[1] == len(files))

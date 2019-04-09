import unittest
from pathlib import Path
import shutil

from ibllib.ephys import ephysqc
from ibllib.io import alf


class TestFlagOperations(unittest.TestCase):

    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys')
        if not self.init_folder.exists():
            return
        self.alf_folder = self.init_folder / 'alf'

    def _qc_extract(self):
        # extract a short lf signal RMS
        for fbin in Path(self.init_folder).rglob('*.lf.bin'):
            ephysqc.extract_rmsmap(fbin, out_folder=self.alf_folder)
            rmsmap_lf = alf.load_object(self.alf_folder, '_ibl_ephysRmsMap_lf')
            spec_lf = alf.load_object(self.alf_folder, '_ibl_ephysSpectra_lf')
            ntimes = rmsmap_lf['times'].shape[0]
            nchannels = rmsmap_lf['rms'].shape[1]
            nfreqs = spec_lf['frequencies'].shape[0]
            # makes sure the dimensions are consistend
            self.assertTrue(rmsmap_lf['rms'].shape == (ntimes, nchannels))
            self.assertTrue(spec_lf['power'].shape == (nfreqs, nchannels))

    def test_pipeline(self):
        if not self.init_folder.exists():
            return
        self._qc_extract()

    def tearDown(self):
        if not self.init_folder.exists():
            return
        shutil.rmtree(self.alf_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(exit=False)

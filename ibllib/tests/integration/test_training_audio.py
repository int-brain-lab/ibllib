import unittest

import numpy as np
import one.alf.io as alfio
from ibllib.io.extractors import training_audio as audio

from ci.tests import base


class TestAudioExtraction(base.IntegrationTest):

    required_files = ['Subjects_init/ZM_1085/2019-07-01/002/raw_behavior_data']

    def setUp(self):
        file_wav = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-07-01', '002',
                                           'raw_behavior_data', '_iblrig_micData.raw.wav')
        self.ses_path = file_wav.parents[1]
        if not self.ses_path.exists():
            return

    def test_qc_extract(self):
        # extract audio
        audio.extract_sound(self.ses_path, save=True)
        D = alfio.load_object(self.ses_path / 'raw_behavior_data', 'audioSpectrogram')
        cues = alfio.load_object(self.ses_path / 'raw_behavior_data', 'audioOnsetGoCue')
        self.assertEqual(cues['times_mic'].size, 7)
        self.assertEqual(D['power'].shape[0], D['times_mic'].shape[0])
        self.assertEqual(D['frequencies'].shape[1], D['power'].shape[1])
        # now test the registration of the data

    def tearDown(self):
        path_out = self.ses_path / 'raw_behavior_data'
        for f in path_out.glob('_iblmic_*'):
            f.unlink()


class TestAudioProcessing(base.IntegrationTest):

    def test_detect_go_cues(self):
        fs = 200_000
        w = np.load(self.data_path.joinpath("sound/example_gocue_clicks_error_fs200k.npy"))
        dtect = audio.detect_ready_tone(w, fs, threshold=.2)
        # this example contains 3 go cue times
        self.assertTrue(np.all(dtect == (np.array([188863, 1318916, 1932242]))))


if __name__ == "__main__":
    unittest.main(exit=False)

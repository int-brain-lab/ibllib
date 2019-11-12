import unittest
import tempfile
from pathlib import Path

import ibllib.pipes.extract_session
from ibllib.pipes import experimental_data


class TestCompression(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_empty_flag_file(self):
        with tempfile.TemporaryDirectory() as tdir:
            flag = Path(tdir).joinpath('compress_video.flag')
            flag.touch()
            experimental_data.compress_video(tdir, dry=True)
            self.assertFalse(flag.exists())


class TestExtractors(unittest.TestCase):

    def test_task_names_extractors(self):
        """
        This is to test against regressions
        """
        task_out = [
            ('_iblrig_tasks_biasedChoiceWorld3.7.0', 'biased'),
            ('_iblrig_tasks_biasedScanningChoiceWorld5.2.3', 'biased'),
            ('_iblrig_tasks_trainingChoiceWorld3.6.0', 'training'),
            ('_iblrig_tasks_ephysChoiceWorld5.1.3', 'ephys'),
            ('_iblrig_calibration_frame2TTL4.1.3', None),
            ('_iblrig_tasks_habituationChoiceWorld3.6.0', None),
            ('_iblrig_tasks_scanningOptoChoiceWorld5.0.0', None),
            ('_iblrig_tasks_RewardChoiceWorld4.1.3', None),
            ('_iblrig_calibration_screen4.1.3', None),
            ('_iblrig_tasks_ephys_certification4.1.3', 'sync_ephys'),
        ]
        for to in task_out:
            out = ibllib.pipes.extract_session.get_task_extractor_type(to[0])
            self.assertEqual(out, to[1])


if __name__ == "__main__":
    unittest.main(exit=False)

import unittest

import numpy as np
import pandas as pd

from brainbox.io import one as bbone


class TestIO_ONE(unittest.TestCase):
    """Tests for brainbox.io.one functions that don't require fixtures on disk."""

    @staticmethod
    def _make_tidy_trials():
        """Build a small trials DataFrame spanning several probabilityLeft blocks."""
        return pd.DataFrame(
            {
                'choice': [-1.0, 0.0, 1.0, -1.0, 1.0, 0.0],
                'feedbackType': [1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                # one of contrastLeft/contrastRight holds the value, the other is NaN
                'contrastLeft': [0.25, np.nan, 0.0, np.nan, 0.125, np.nan],
                'contrastRight': [np.nan, 1.0, np.nan, 0.0, np.nan, 0.0625],
                'probabilityLeft': [0.5, 0.5, 0.8, 0.8, 0.2, 0.5],
            }
        )

    def test_tidy_choice_mapping(self):
        """choice: -1/0/+1 map to counter_clockwise/none/clockwise."""
        result = bbone.SessionLoader.apply_tidy_transformations(self._make_tidy_trials())
        expected = ['counter_clockwise', 'none', 'clockwise', 'counter_clockwise', 'clockwise', 'none']
        self.assertEqual(result['choice'].tolist(), expected)

    def test_tidy_feedback_to_boolean(self):
        """feedbackType +1/-1 maps to is_mouse_rewarded True/False."""
        result = bbone.SessionLoader.apply_tidy_transformations(self._make_tidy_trials())
        self.assertEqual(result['is_mouse_rewarded'].tolist(), [True, False, True, True, False, True])

    def test_tidy_gabor_stimulus_side_and_contrast(self):
        """contrastLeft/contrastRight consolidate into side + contrast (percent), including 0% trials."""
        result = bbone.SessionLoader.apply_tidy_transformations(self._make_tidy_trials())
        self.assertEqual(result['gabor_stimulus_side'].tolist(), ['left', 'right', 'left', 'right', 'left', 'right'])
        np.testing.assert_array_almost_equal(
            result['gabor_stimulus_contrast'].to_numpy(dtype=float), [25.0, 100.0, 0.0, 0.0, 12.5, 6.25]
        )

    def test_tidy_block_index_and_type(self):
        """probabilityLeft yields an incrementing block_index and a categorical block_type."""
        result = bbone.SessionLoader.apply_tidy_transformations(self._make_tidy_trials())
        self.assertEqual(result['block_index'].tolist(), [0, 0, 1, 1, 2, 3])
        self.assertEqual(
            result['block_type'].tolist(),
            ['unbiased', 'unbiased', 'left_block', 'left_block', 'right_block', 'unbiased'],
        )

    def test_tidy_does_not_mutate_input(self):
        """The input DataFrame is copied, not modified in place."""
        trials = self._make_tidy_trials()
        before = trials.copy()
        bbone.SessionLoader.apply_tidy_transformations(trials)
        pd.testing.assert_frame_equal(trials, before)

    def test_tidy_nan_probability_left_raises(self):
        """A NaN in probabilityLeft indicates corrupted data and must raise ValueError."""
        trials = self._make_tidy_trials()
        trials.loc[2, 'probabilityLeft'] = np.nan
        self.assertRaises(ValueError, bbone.SessionLoader.apply_tidy_transformations, trials)

    def test_load_iti(self):
        """Test for brainbox.io.one.load_iti function."""
        trials = bbone.alfio.AlfBunch({})
        trials.intervals = np.array(
            [
                [114.52487625, 117.88103707],
                [118.5169474, 122.89742147],
                [123.49302927, 126.12216664],
                [126.68107337, 129.53872083],
                [130.11952807, 133.90539162],
            ]
        )
        trials.stimOff_times = [117.38098379, 122.39736201, 125.62210278, 129.03865947, 133.4053633]
        expected = np.array([1.13596361, 1.09566726, 1.05897059, 1.0808686, np.nan])
        np.testing.assert_array_almost_equal(bbone.load_iti(trials), expected)
        _ = trials.pop('stimOff_times')
        self.assertRaises(ValueError, bbone.load_iti, trials)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

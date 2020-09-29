# Mock dataset
import unittest

import numpy as np

from ibllib.ephys import ephysqc, neuropixel


class TestNeuropixel(unittest.TestCase):

    def test_layouts(self):
        dense = neuropixel.dense_layout()
        assert set(dense.keys()) == set(['x', 'y', 'row', 'col', 'ind'])
        xu = np.unique(dense['x'])
        yu = np.unique(dense['y'])
        assert np.all(np.diff(xu) == 16)
        assert np.all(np.diff(yu) == 20)
        assert xu.size == 4 and yu.size == 384 / 2


class TestFpgaTask(unittest.TestCase):

    def test_impeccable_dataset(self):

        fpga2bpod = np.array([11 * 1e-6, -20])  # bpod starts 20 secs before with 10 ppm drift
        fpga_trials = {
            'intervals': np.array([[0, 9.5], [10, 19.5]]),
            'stimOn_times': np.array([2, 12]),
            'goCue_times': np.array([2.0001, 12.0001]),
            'stimFreeze_times': np.array([4., 14.]),
            'feedback_times': np.array([4.0001, 14.0001]),
            'errorCue_times': np.array([4.0001, np.nan]),
            'valveOpen_times': np.array([np.nan, 14.0001]),
            'stimOff_times': np.array([6.0001, 15.0001]),
            'itiIn_times': np.array([6.0011, 15.000]),
        }

        alf_trials = {
            'goCueTrigger_times_bpod': np.polyval(fpga2bpod, fpga_trials['goCue_times'] - 0.00067),
            'response_times_bpod': np.polyval(fpga2bpod, np.array([4., 14.])),
            'intervals_bpod': np.polyval(fpga2bpod, fpga_trials['intervals']),
            # Times from session start
            'goCueTrigger_times': fpga_trials['goCue_times'] - 0.00067,
            'response_times': np.array([4., 14.]),
            'intervals': fpga_trials['intervals'],
            'stimOn_times': fpga_trials['stimOn_times'],
            'goCue_times': fpga_trials['goCue_times'],
            'feedback_times': fpga_trials['feedback_times'],
        }
        qcs, qct = ephysqc.qc_fpga_task(fpga_trials, alf_trials)
        self.assertTrue(np.all([qcs[k] for k in qcs]))
        self.assertTrue(np.all([np.all(qct[k]) for k in qct]))


if __name__ == "__main__":
    unittest.main(exit=False)

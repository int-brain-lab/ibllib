import unittest
import numpy as np
import pandas as pd
import brainbox.modeling.design_matrix as bdm
import brainbox.modeling.utils as mut
from pathlib import Path


class TestModeling(unittest.TestCase):

    def setUp(self):
        # Params for test
        self.binwidth = 0.02

        # Generate fake trial start, stimon, feedback, and end times
        starts = np.array([0, 1.48, 2.93, 4.67, 6.01, 7.31, 8.68, 9.99, 11.43, 12.86])
        ends = np.array([1.35, 2.09, 3.53, 5.23, 6.58, 7.95, 9.37, 11.31, 12.14, 13.26])
        stons = starts + 0.1
        fdbks = np.array([0.24, 1.64, 3.15, 4.81, 6.23, 7.50, 8.91, 10.16, 11.64, 13.05])

        # Figure out how many bins each trial is and generate non-monotonic trace of fake wheel
        whlpath = Path(__file__).parent.joinpath('fixtures', 'design_wheel_traces_test.p')
        if whlpath.exists():
            fakewheels = np.load(whlpath, allow_pickle=True)

        # Store trialsdf for later use
        self.trialsdf = pd.DataFrame({'trial_start': starts,
                                      'trial_end': ends,
                                      'stim_onset': stons,
                                      'feedback': fdbks,
                                      'wheel_traces': fakewheels})

    def binf(self, x):
        return np.ceil(x / self.binwidth).astype(int)

    def test_dm_construct(self):
        """
        Check whether or not design matrix construction works as intended
        """

        # Design matrix instance
        self.design = bdm.DesignMatrix(self.trialsdf,
                                       vartypes={'trial_start': 'timing',
                                                 'trial_end': 'timing',
                                                 'stim_onset': 'timing',
                                                 'feedback': 'timing',
                                                 'wheel_traces': 'continuous'})

        # Separate bases for wheel and timing
        tbases = mut.raised_cosine(0.2, 3, self.binf)
        wbases = mut.raised_cosine(0.1, 2, self.binf)
        # Add covariates one by one. Add different offsets on timings to test
        self.design.add_covariate_timing('start', 'trial_start', tbases, offset=0.02)
        self.design.add_covariate_timing('stim_on', 'stim_onset', tbases)
        self.design.add_covariate_timing('feedback', 'feedback', tbases, offset=-0.02)
        self.design.add_covariate('wheelpos', self.trialsdf.wheel_traces, wbases, offset=-0.1)
        self.design.compile_design_matrix()  # Finally compile
        # Load target DM
        npy_file = Path(__file__).parent.joinpath('fixtures', 'design_matrix_test.npy')
        if npy_file.exists():
            ref_dm = np.load(npy_file)
            self.assertTrue(np.allclose(self.design.dm, ref_dm))

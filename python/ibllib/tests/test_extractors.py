import unittest
from pathlib import Path
import numpy as np
import logging
import ciso8601

import alf.extractors as ex
from ibllib.io import raw_data_loaders as loaders


class TestExtractTrialData(unittest.TestCase):

    def setUp(self):
        self.session_path = Path(__file__).parent.joinpath('data')
        self.data = loaders.load_data(self.session_path)
        # turn off logging for unit testing as we will purposely go into warning/error cases
        self.logger = logging.getLogger('ibllib').setLevel(50)

    def test_stimOn_times(self):
        st = ex.training_trials.get_stimOn_times('', save=False, data=self.data)
        self.assertTrue(isinstance(st, np.ndarray))

    def test_encoder_positions_duds(self):
        dy = loaders.load_encoder_positions(self.session_path)
        self.assertEqual(dy.bns_ts.dtype.name, 'datetime64[ns]')
        self.assertTrue(dy.shape[0] == 2)

    def test_encoder_events_duds(self):
        dy = loaders.load_encoder_events(self.session_path)
        self.assertEqual(dy.bns_ts.dtype.name, 'datetime64[ns]')
        self.assertTrue(dy.shape[0] == 8)

    def test_interpolation(self):
        # straight test that it returns an usable function
        ta = np.array([0., 1., 2., 3., 4., 5.])
        tb = np.array([0., 1.1, 2.0, 2.9, 4., 5.])
        finterp = ex.time_interpolation(ta, tb)
        self.assertTrue(np.all(finterp(ta) == tb))
        # next test if sizes are not similar
        tc = np.array([0., 1.1, 2.0, 2.9, 4., 5., 6.])
        finterp = ex.time_interpolation(ta, tc)
        self.assertTrue(np.all(finterp(ta) == tb))

    def test_ciso8601(self):
        dt = ciso8601.parse_datetime('2018-01-16T14:21:32')
        self.assertFalse(not dt)

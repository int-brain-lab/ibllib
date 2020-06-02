import unittest
from pathlib import Path
import numpy as np

import ibllib.io.extractors.ephys_fpga as ephys_fpga
import ibllib.io.raw_data_loaders as raw


class TestEphysSyncExtraction(unittest.TestCase):

    def test_bpod_trace_extraction(self):

        t_valve_open_ = np.array([117.12136667, 122.3873, 127.82903333, 140.56083333,
                                  143.55326667, 155.29713333, 164.9186, 167.91133333,
                                  171.39736667, 178.0305, 181.70343333])

        t_trial_start_ = np.array([109.7647, 118.51416667, 123.7964, 129.24503333,
                                   132.97976667, 136.8624, 141.95523333, 144.93636667,
                                   149.5042, 153.08273333, 156.70316667, 164.0096,
                                   166.30633333, 169.28373333, 172.77786667, 176.7828,
                                   179.41063333])
        t_trial_start_[0] = 6.75033333  # rising front for first trial instead of falling
        bpod_fronts_ = np.array([1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                 -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                 -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                 1., -1., 1., -1., 1., -1.])

        bpod_times_ = np.array([6.75033333, 109.7648, 117.12136667, 117.27136667,
                                118.51416667, 118.51426667, 122.3873, 122.5373,
                                123.7964, 123.7965, 127.82903333, 127.97903333,
                                129.24503333, 129.24513333, 132.97976667, 132.97986667,
                                136.8624, 136.8625, 140.56083333, 140.71083333,
                                141.95523333, 141.95533333, 143.55326667, 143.70326667,
                                144.93636667, 144.93646667, 149.5042, 149.5043,
                                153.08273333, 153.08283333, 155.29713333, 155.44713333,
                                156.70316667, 156.70326667, 164.0096, 164.0097,
                                164.9186, 165.0686, 166.30633333, 166.30643333,
                                167.91133333, 168.06133333, 169.28373333, 169.28386667,
                                171.39736667, 171.54736667, 172.77786667, 172.77796667,
                                176.7828, 176.7829, 178.0305, 178.1805,
                                179.41063333, 179.41073333, 181.70343333, 181.85343333,
                                183.12896667, 183.12906667])

        t_trial_start, t_valve_open, _ = ephys_fpga._assign_events_bpod(bpod_times_,
                                                                        bpod_fronts_)
        self.assertTrue(np.all(np.isclose(t_trial_start, t_trial_start_)))
        self.assertTrue(np.all(np.isclose(t_valve_open, t_valve_open_)))

    def test_align_to_trial(self):
        # simple test with one missing at the end
        t_trial_start = np.arange(0, 5) * 10
        t_event = np.arange(0, 5) * 10 + 2
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event)
        self.assertTrue(np.allclose(t_event_nans, t_event, equal_nan=True, atol=0, rtol=0))

        # test with missing values
        t_trial_start = np.array([109, 118, 123, 129, 132, 136, 141, 144, 149, 153])
        t_event = np.array([122, 133, 140, 143, 146, 150, 154])
        t_event_out_ = np.array([np.nan, 122, np.nan, np.nan, 133, 140, 143, 146, 150, 154])
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event)
        self.assertTrue(np.allclose(t_event_out_, t_event_nans, equal_nan=True, atol=0, rtol=0))

        # test with events before initial start trial
        t_trial_start = np.arange(0, 5) * 10
        t_event = np.arange(0, 5) * 10 - 2
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event)
        desired_out = np.array([8., 18., 28., 38., np.nan])
        self.assertTrue(np.allclose(desired_out, t_event_nans, equal_nan=True, atol=0, rtol=0))

        # test with several events per trial, missing events and events before
        t_trial_start = np.array([0, 10, 20, 30, 40])
        t_event = np.array([-1, 2, 4, 12, 35, 42])
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event)
        desired_out = np.array([4, 12., np.nan, 35, 42])
        self.assertTrue(np.allclose(desired_out, t_event_nans, equal_nan=True, atol=0, rtol=0))

        # same test above but this time take the first index instead of last
        t_trial_start = np.array([0, 10, 20, 30, 40])
        t_event = np.array([-1, 2, 4, 12, 35, 42])
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event, take='first')
        desired_out = np.array([2, 12., np.nan, 35, 42])
        self.assertTrue(np.allclose(desired_out, t_event_nans, equal_nan=True, atol=0, rtol=0))

    def test_wheel_trace_from_sync(self):
        pos_ = - np.array([-1, 0, -1, -2, -1, -2]) * (np.pi / ephys_fpga.WHEEL_TICKS)
        ta = np.array([1, 2, 3, 4, 5, 6])
        tb = np.array([0.5, 3.2, 3.3, 3.4, 5.25, 5.5])
        pa = (np.mod(np.arange(6), 2) - 0.5) * 2
        pb = (np.mod(np.arange(6) + 1, 2) - .5) * 2
        t, pos = ephys_fpga._rotary_encoder_positions_from_fronts(ta, pa, tb, pb, coding='x2')
        self.assertTrue(np.all(np.isclose(pos_, pos)))

        pos_ = - np.array([-1, 0, -1, 0, -1, -2]) * (np.pi / ephys_fpga.WHEEL_TICKS)
        tb = np.array([0.5, 3.2, 3.4, 5.25])
        pb = (np.mod(np.arange(4) + 1, 2) - .5) * 2
        t, pos = ephys_fpga._rotary_encoder_positions_from_fronts(ta, pa, tb, pb, coding='x2')
        self.assertTrue(np.all(np.isclose(pos_, pos)))


class TestEphysBehaviorExtraction(unittest.TestCase):
    def setUp(self):
        self.session_path = Path(__file__).parent.joinpath('data', 'session_ephys')

    def test_get_probabilityLeft(self):
        data = raw.load_data(self.session_path)
        settings = raw.load_settings(self.session_path)
        pLeft0, _, _ = ephys_fpga.ProbaContrasts(
            self.session_path).extract(bpod_trials=data, settings=settings)[0]
        self.assertTrue(len(pLeft0) == len(data))
        # Test if only generative prob values in data
        self.assertTrue(all([x in [0.2, 0.5, 0.8] for x in np.unique(pLeft0)]))
        # Test if settings file has empty LEN_DATA result is same
        settings.update({"LEN_BLOCKS": None})
        pLeft1, _, _ = ephys_fpga.ProbaContrasts(
            self.session_path).extract(bpod_trials=data, settings=settings)[0]
        self.assertTrue(all(pLeft0 == pLeft1))
        # Test if only generative prob values in data
        self.assertTrue(all([x in [0.2, 0.5, 0.8] for x in np.unique(pLeft1)]))

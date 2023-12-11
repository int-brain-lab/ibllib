import unittest
from pathlib import Path
import pickle

import numpy as np

from ibllib.io.extractors import ephys_fpga, biased_trials
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.training_wheel import extract_first_movement_times, infer_wheel_units
from ibllib.io.extractors.training_wheel import extract_wheel_moves
import brainbox.behavior.wheel as wh


class TestEphysSyncExtraction(unittest.TestCase):

    def test_bpod_trace_extraction(self):
        """Test ephys_fpga._assign_events_bpod function.

        TODO Remove this test and corresponding function.
        """
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
        """Test ephys_fpga._assign_events_to_trial function."""
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
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event, take='first')
        desired_out = np.array([2, 12., np.nan, 35, 42])
        self.assertTrue(np.allclose(desired_out, t_event_nans, equal_nan=True, atol=0, rtol=0))

        # take second to last
        t_trial_start = np.array([0, 10, 20, 30, 40])
        t_event = np.array([2, 4, 12, 13, 14, 21, 32, 33, 34, 35, 42])
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event, take=-2)
        desired_out = np.array([2, 13, np.nan, 34, np.nan])
        self.assertTrue(np.allclose(desired_out, t_event_nans, equal_nan=True, atol=0, rtol=0))
        t_event_nans = ephys_fpga._assign_events_to_trial(t_trial_start, t_event, take=1)
        desired_out = np.array([4, 13, np.nan, 33, np.nan])
        self.assertTrue(np.allclose(desired_out, t_event_nans, equal_nan=True, atol=0, rtol=0))

        # test errors
        self.assertRaises(ValueError, ephys_fpga._assign_events_to_trial, np.array([0., 2., 1.]), t_event)
        self.assertRaises(ValueError, ephys_fpga._assign_events_to_trial, t_trial_start, np.array([0., 2., 1.]))

    def test_wheel_trace_from_sync(self):
        """Test ephys_fpga._rotary_encoder_positions_from_fronts function."""
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

    def test_time_fields(self):
        """Test for FpgaTrials._time_fields static method."""
        expected = ('intervals', 'fooBar_times_bpod', 'spike_times', 'baz_timestamps')
        fields = ephys_fpga.FpgaTrials._time_fields(expected + ('position', 'timebase', 'fooBaz'))
        self.assertCountEqual(expected, fields)


class TestEphysBehaviorExtraction(unittest.TestCase):
    def setUp(self):
        self.session_path = Path(__file__).parent.joinpath('data', 'session_ephys')

    def test_get_probabilityLeft(self):
        data = raw.load_data(self.session_path)
        settings = raw.load_settings(self.session_path)
        *_, pLeft0, _ = biased_trials.ProbaContrasts(
            self.session_path).extract(bpod_trials=data, settings=settings)[0]
        self.assertTrue(len(pLeft0) == len(data))
        # Test if only generative prob values in data
        self.assertTrue(all([x in [0.2, 0.5, 0.8] for x in np.unique(pLeft0)]))
        # Test if settings file has empty LEN_DATA result is same
        settings.update({"LEN_BLOCKS": None})
        *_, pLeft1, _ = biased_trials.ProbaContrasts(
            self.session_path).extract(bpod_trials=data, settings=settings)[0]
        self.assertTrue(all(pLeft0 == pLeft1))
        # Test if only generative prob values in data
        self.assertTrue(all([x in [0.2, 0.5, 0.8] for x in np.unique(pLeft1)]))


class TestWheelExtraction(unittest.TestCase):

    def setUp(self) -> None:
        self.ta = np.array([2, 4, 6, 8, 12, 14, 16, 18])
        self.pa = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        self.tb = np.array([3, 5, 7, 9, 11, 13, 15, 17])
        self.pb = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    def test_x1_decoding(self):
        p_ = np.array([1, 2, 1, 0])
        t_ = np.array([2, 6, 11, 15])
        t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
            self.ta, self.pa, self.tb, self.pb, ticks=np.pi * 2, coding='x1')
        self.assertTrue(np.all(t == t_))
        self.assertTrue(np.all(p == p_))

    def test_x4_decoding(self):
        p_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]) / 4
        t_ = np.array([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18])
        t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
            self.ta, self.pa, self.tb, self.pb, ticks=np.pi * 2, coding='x4')
        self.assertTrue(np.all(t == t_))
        self.assertTrue(np.all(np.isclose(p, p_)))

    def test_x2_decoding(self):
        p_ = np.array([1, 2, 3, 4, 3, 2, 1, 0]) / 2
        t_ = np.array([2, 4, 6, 8, 12, 14, 16, 18])
        t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
            self.ta, self.pa, self.tb, self.pb, ticks=np.pi * 2, coding='x2')
        self.assertTrue(np.all(t == t_))
        self.assertTrue(np.all(p == p_))


class TestExtractedWheelUnits(unittest.TestCase):
    """Tests the infer_wheel_units function"""

    wheel_radius_cm = 3.1

    def setUp(self) -> None:
        """
        Create the wheel position data for testing: the positions attribute holds a dictionary of
        units, each holding a dictionary of encoding types to test, e.g.

        positions = {
            'rad': {
                'X1': ...,
                'X2': ...,
                'X4': ...
            },
            'cm': {
                'X1': ...,
                'X2': ...,
                'X4': ...
            }
        }
        :return:
        """
        def x(unit, enc=int(1), wheel_radius=self.wheel_radius_cm):
            radius = 1 if unit == 'rad' else wheel_radius
            return 1 / ephys_fpga.WHEEL_TICKS * np.pi * 2 * radius / enc

        # A pseudo-random sequence of integrated fronts
        seq = np.array([-1, 0, 1, 2, 1, 2, 3, 4, 3, 2, 1, 0, -1, -2, 1, -2])
        encs = (1, 2, 4)  # Encoding types to test
        units = ('rad', 'cm')  # Units to test
        self.positions = {unit: {f'X{e}': x(unit, e) * seq for e in encs} for unit in units}

    def test_extract_wheel_moves(self):
        for unit in self.positions.keys():
            for encoding, pos in self.positions[unit].items():
                result = infer_wheel_units(pos)
                self.assertEqual(unit, result[0], f'failed to determine units for {encoding}')
                expected = int(ephys_fpga.WHEEL_TICKS * int(encoding[1]))
                self.assertEqual(expected, result[1],
                                 f'failed to determine number of ticks for {encoding} in {unit}')
                self.assertEqual(encoding, result[2], f'failed to determine encoding in {unit}')


class TestWheelMovesExtraction(unittest.TestCase):

    def setUp(self) -> None:
        """
        Test data is in the form ((inputs), (outputs)) where inputs is a tuple containing a
        numpy array of timestamps and one of positions; outputs is a tuple of outputs from
        the functions.  For details, see help on TestWheel.setUp method in module
        brainbox.tests.test_behavior
        """
        pickle_file = Path(__file__).parents[3].joinpath(
            'brainbox', 'tests', 'fixtures', 'wheel_test.p')
        if not pickle_file.exists():
            self.test_data = None
        else:
            with open(pickle_file, 'rb') as f:
                self.test_data = pickle.load(f)

        # Some trial times for trial_data[1]
        self.trials = {
            'goCue_times': np.array([162.5, 105.6, 55]),
            'feedback_times': np.array([164.3, 108.3, 56])
        }

    def test_extract_wheel_moves(self):
        test_data = self.test_data[1]
        # Wrangle data into expected form
        re_ts = test_data[0][0]
        re_pos = test_data[0][1]

        logname = 'ibllib.io.extractors.training_wheel'
        with self.assertLogs(logname, level='INFO') as cm:
            wheel_moves = extract_wheel_moves(re_ts, re_pos)
            self.assertEqual([f'INFO:{logname}:Wheel in cm units using X2 encoding'], cm.output)

        n = 56  # expected number of movements
        self.assertTupleEqual(wheel_moves['intervals'].shape, (n, 2),
                              'failed to return the correct number of intervals')
        self.assertEqual(wheel_moves['peakAmplitude'].size, n)
        self.assertEqual(wheel_moves['peakVelocity_times'].size, n)

        # Check the first 3 intervals
        ints = np.array(
            [[24.78462599, 25.22562599],
             [29.58762599, 31.15062599],
             [31.64262599, 31.81662599]])
        actual = wheel_moves['intervals'][:3, ]
        self.assertIsNone(np.testing.assert_allclose(actual, ints), 'unexpected intervals')

        # Check amplitudes
        actual = wheel_moves['peakAmplitude'][-3:]
        expected = [0.50255486, -1.70103154, 1.00740789]
        self.assertIsNone(np.testing.assert_allclose(actual, expected), 'unexpected amplitudes')

        # Check peak velocities
        actual = wheel_moves['peakVelocity_times'][-3:]
        expected = [175.13662599, 176.65762599, 178.57262599]
        self.assertIsNone(np.testing.assert_allclose(actual, expected), 'peak times')

        # Test extraction in rad
        re_pos = wh.cm_to_rad(re_pos)
        with self.assertLogs(logname, level='INFO') as cm:
            wheel_moves = ephys_fpga.extract_wheel_moves(re_ts, re_pos)
            self.assertEqual([f'INFO:{logname}:Wheel in rad units using X2 encoding'], cm.output)

        # Check the first 3 intervals.  As position thresholds are adjusted by units and
        # encoding, we should expect the intervals to be identical to above
        actual = wheel_moves['intervals'][:3, ]
        self.assertIsNone(np.testing.assert_allclose(actual, ints), 'unexpected intervals')

    def test_movement_log(self):
        """
        Integration test for inferring the units and decoding type for wheel data input for
        extract_wheel_moves.  Only expected to work for the default wheel diameter.
        """
        ta = np.array([2, 4, 6, 8, 12, 14, 16, 18])
        pa = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        tb = np.array([3, 5, 7, 9, 11, 13, 15, 17])
        pb = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        logname = 'ibllib.io.extractors.training_wheel'

        for unit in ['cm', 'rad']:
            for i in (1, 2, 4):
                encoding = 'X' + str(i)
                r = 3.1 if unit == 'cm' else 1
                # print(encoding, unit)
                t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
                    ta, pa, tb, pb, ticks=1024, coding=encoding.lower(), radius=r)
                expected = f'INFO:{logname}:Wheel in {unit} units using {encoding} encoding'
                with self.assertLogs(logname, level='INFO') as cm:
                    ephys_fpga.extract_wheel_moves(t, p)
                    self.assertEqual([expected], cm.output)

    def test_extract_first_movement_times(self):
        test_data = self.test_data[1]
        wheel_moves = ephys_fpga.extract_wheel_moves(test_data[0][0], test_data[0][1])
        first, is_final, ind = extract_first_movement_times(wheel_moves, self.trials)
        np.testing.assert_allclose(first, [162.48462599, 105.62562599, np.nan])
        np.testing.assert_array_equal(is_final, [False, True, False])
        np.testing.assert_array_equal(ind, [46, 18])


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

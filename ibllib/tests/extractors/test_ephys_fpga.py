import unittest
import tempfile
from pathlib import Path
import pickle
import logging

import numpy as np

import ibllib.io.spikeglx as spikeglx
from ibllib.io.extractors.training_wheel import extract_first_movement_times, infer_wheel_units
from ibllib.io.extractors import ephys_fpga
from ibllib.io.extractors.training_wheel import extract_wheel_moves
import brainbox.behavior.wheel as wh


class TestsFolderStructure(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        pl = Path(self.dir.name) / 'raw_ephys_data' / 'probe_left'
        pr = Path(self.dir.name) / 'raw_ephys_data' / 'probe_right'
        pl.mkdir(parents=True)
        pr.mkdir(parents=True)
        (pl / 'iblrig_ephysData.raw_g0_t0.imec.lf.bin').touch()
        (pl / 'iblrig_ephysData.raw_g0_t0.imec.ap.bin').touch()
        (pr / 'iblrig_ephysData.raw_g0_t0.imec.lf.bin').touch()
        (pr / 'iblrig_ephysData.raw_g0_t0.imec.ap.bin').touch()

    def test_get_ephys_files(self):
        # first test at the root directory level, with a string input
        ephys_files = spikeglx.glob_ephys_files(self.dir.name)
        for ef in ephys_files:
            self.assertTrue(ef.label in ['probe_right', 'probe_left'])
            self.assertTrue(ef.ap.exists() and ef.lf.exists())
        # second test at the ephys directory level, with a pathlib.Path input
        ephys_files = spikeglx.glob_ephys_files(Path(self.dir.name) / 'raw_ephys_data')
        for ef in ephys_files:
            self.assertTrue(ef.label in ['probe_right', 'probe_left'])
            self.assertTrue(ef.ap.exists() and ef.lf.exists())

    def tearDown(self):
        self.dir.cleanup()


class TestSyncExtraction(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(__file__).parents[1] / 'fixtures' / 'io' / 'spikeglx'
        self.meta_files = list(Path.glob(self.workdir, '*.meta'))

    def test_sync_nidq(self):
        self.sync_gen(fn='sample3B_g0_t0.nidq.meta', ns=32, nc=2, sync_depth=8)

    def test_sync_3B(self):
        self.sync_gen(fn='sample3B_g0_t0.imec1.ap.meta', ns=32, nc=385, sync_depth=16)

    def test_sync_3A(self):
        self.sync_gen(fn='sample3A_g0_t0.imec.ap.meta', ns=32, nc=385, sync_depth=16)

    def sync_gen(self, fn, ns, nc, sync_depth):
        # nidq has 1 analog and 1 digital sync channels
        with tempfile.TemporaryDirectory() as tdir:
            ses_path = Path(tdir).joinpath('raw_ephys_data')
            ses_path.mkdir(parents=True, exist_ok=True)
            bin_file = ses_path.joinpath(fn).with_suffix('.bin')
            nidq = spikeglx._mock_spikeglx_file(bin_file, self.workdir / fn,
                                                ns=ns, nc=nc, sync_depth=sync_depth)
            syncs, files = ephys_fpga.extract_sync(tdir)
            self.assertTrue(np.all(syncs[0].channels[slice(1, None, 2)] ==
                                   np.arange(0, nidq['sync_depth'])))
            with self.assertLogs(level='INFO') as log:
                ephys_fpga.extract_sync(tdir)
                self.assertEqual(1, len(log.output))
                self.assertIn('SGLX sync found', log.output[0])


class TestIblChannelMaps(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(__file__).parents[1] / 'fixtures'

    def test_ibl_sync_maps(self):
        s = ephys_fpga.get_ibl_sync_map({'ap': 'toto', 'path': self.workdir}, '3A')
        self.assertEqual(s, ephys_fpga.CHMAPS['3A']['ap'])
        s = ephys_fpga.get_ibl_sync_map({'nidq': 'toto', 'path': self.workdir}, '3B')
        self.assertEqual(s, ephys_fpga.CHMAPS['3B']['nidq'])
        s = ephys_fpga.get_ibl_sync_map({'ap': 'toto', 'path': self.workdir}, '3B')
        self.assertEqual(s, ephys_fpga.CHMAPS['3B']['ap'])


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

        logger = logging.getLogger('ibllib')
        with self.assertLogs(logger, level='INFO') as cm:
            wheel_moves = extract_wheel_moves(re_ts, re_pos)
            self.assertEqual(['INFO:ibllib:Wheel in cm units using X2 encoding'], cm.output)

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
        with self.assertLogs(logger, level='INFO') as cm:
            wheel_moves = ephys_fpga.extract_wheel_moves(re_ts, re_pos)
            self.assertEqual(['INFO:ibllib:Wheel in rad units using X2 encoding'], cm.output)

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
        logger = logging.getLogger('ibllib')

        for unit in ['cm', 'rad']:
            for i in (1, 2, 4):
                encoding = 'X' + str(i)
                r = 3.1 if unit == 'cm' else 1
                # print(encoding, unit)
                t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
                    ta, pa, tb, pb, ticks=1024, coding=encoding.lower(), radius=r)
                expected = 'INFO:ibllib:Wheel in {} units using {} encoding'.format(unit, encoding)
                with self.assertLogs(logger, level='INFO') as cm:
                    ephys_fpga.extract_wheel_moves(t, p)
                    self.assertEqual([expected], cm.output)

    def test_extract_first_movement_times(self):
        test_data = self.test_data[1]
        wheel_moves = ephys_fpga.extract_wheel_moves(test_data[0][0], test_data[0][1])
        first, is_final, ind = extract_first_movement_times(wheel_moves, self.trials)

        np.testing.assert_allclose(first, [162.48462599, 105.62562599, np.nan])
        np.testing.assert_array_equal(is_final, [False, True, False])
        np.testing.assert_array_equal(ind, [46, 18])


class TestEphysFPGA_TTLsExtraction(unittest.TestCase):

    def test_ttl_bpod_gaelle_writes_protocols_but_guido_doesnt_read_them(self):
        bpod_t = np.array([5.423290950005423, 6.397993470006398, 6.468919710006469,
                           7.497916800007498, 7.997933460007998, 8.599239990008599,
                           8.5993399800086, 15.141985650015142, 15.642002310015641,
                           16.215411630016217, 16.215511620016215, 17.104122750017105,
                           17.175015660017174, 18.204012750018205, 18.704029410018705,
                           19.286337840019286, 19.28643783001929, 21.76005711002176,
                           21.83095002002183, 22.85998044002286])
        # when the bpod has started before the ephys, the first pulses may have been missed
        # and the first TTL may be negative/. This needs to yield the same result as if the
        # bpod was started properly
        pol = (np.mod(np.arange(bpod_t.size), 2) - 0.5) * 2
        st, op, iti = ephys_fpga._assign_events_bpod(bpod_t=bpod_t, bpod_polarities=pol)
        st_, op_, iti_ = ephys_fpga._assign_events_bpod(bpod_t=bpod_t[1:], bpod_polarities=pol[1:])
        assert np.all(st == st_) and np.all(op == op_) and np.all(iti_ == iti)

    def test_frame2ttl_flickers(self):
        """
        Frame2ttl can flicker abnormally. One way to detect this is to remove consecutive polarity
        switches under a given threshold
        """
        DISPLAY = False  # for debug purposes
        diff = ephys_fpga.F2TTL_THRESH * np.array([0.5, 10])

        # flicker ends with a polarity switch - downgoing pulse is removed
        t = np.r_[0, np.cumsum(diff[np.array([1, 1, 0, 0, 1])])] + 1
        frame2ttl = {'times': t, 'polarities': np.mod(np.arange(t.size) + 1, 2) * 2 - 1}
        expected = {'times': np.array([1., 1.1, 1.2, 1.31]),
                    'polarities': np.array([1, -1, 1, -1])}
        frame2ttl_ = ephys_fpga._clean_frame2ttl(frame2ttl, display=DISPLAY)
        assert all([np.all(frame2ttl_[k] == expected[k]) for k in frame2ttl_])

        # stand-alone flicker
        t = np.r_[0, np.cumsum(diff[np.array([1, 1, 0, 0, 0, 1])])] + 1
        frame2ttl = {'times': t, 'polarities': np.mod(np.arange(t.size) + 1, 2) * 2 - 1}
        expected = {'times': np.array([1., 1.1, 1.2, 1.215, 1.315]),
                    'polarities': np.array([1, -1, 1, -1, 1])}
        frame2ttl_ = ephys_fpga._clean_frame2ttl(frame2ttl, display=DISPLAY)
        assert all([np.all(frame2ttl_[k] == expected[k]) for k in frame2ttl_])


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

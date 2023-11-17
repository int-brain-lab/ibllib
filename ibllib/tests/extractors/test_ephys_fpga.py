"""Tests for ephys FPGA sync and FPGA wheel extraction."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from ibllib.io.extractors import ephys_fpga
import spikeglx


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
        self.workdir = Path(__file__).parents[1] / 'fixtures' / 'sync_ephys_fpga'
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


class TestEphysFPGA_TTLsExtraction(unittest.TestCase):

    def test_audio_ttl_wiring_camera(self):
        """
        Test ephys_fpga._clean_audio function.

        Test removal of spurious TTLs due to a wrong wiring of the camera onto the soundcard
        example eid: e349a2e7-50a3-47ca-bc45-20d1899854ec
        """
        # on this example it's business as usual and the clean up should not change anything
        audio = {
            'times': np.array([1740.1032, 1740.20176667, 1741.0786, 1741.57713333, 1744.78716667, 1744.88573333]),
            'polarities': np.array([1., -1., 1., -1., 1., -1.]),
        }
        audio_ = ephys_fpga._clean_audio(audio)
        self.assertEqual(audio, audio_)
        # on that example
        audio = {'times': np.array([3399.4090251, 3399.4110249, 3399.4156911, 3399.4176909,
                                    3399.42232377, 3399.42432357, 3399.42898977, 3399.43095624,
                                    3399.43562244, 3399.43762224, 3399.44228844, 3399.44425491,
                                    3399.44892111, 3399.45092091, 3399.45558711, 3399.45755358,
                                    3399.46221978, 3399.46421958, 3399.46888578, 3399.47085225,
                                    3399.47551845, 3399.47751825, 3399.48215112, 3399.48415092,
                                    3399.48881712, 3399.49081692, 3399.49544979, 3399.49744959,
                                    3399.50211579, 3399.50411559, 3400.306602, 3400.3086018,
                                    3400.313268, 3400.3152678, 3400.31990067, 3400.32190047,
                                    3400.32656667, 3400.32856647, 3400.33319934, 3400.33519914,
                                    3400.33986534, 3400.34186514, 3400.34649801, 3400.34849781,
                                    3400.35316401, 3400.35516381, 3400.35979668, 3400.36179648,
                                    3400.36646268, 3400.36846248, 3400.37309535, 3400.37509515,
                                    3400.37976135, 3400.38172782, 3400.38639402, 3400.38839382,
                                    3400.39306002, 3400.39502649, 3400.39969269, 3400.40169249,
                                    3400.40635869, 3400.40832516, 3400.41299136, 3400.41499116,
                                    3400.41962403, 3400.42162383, 3400.42629003, 3400.42828983,
                                    3400.4329227, 3400.4349225, 3400.4395887, 3400.4415885,
                                    3400.44622137, 3400.44822117, 3400.45288737, 3400.45488717,
                                    3400.45952004, 3400.46151984, 3400.46618604, 3400.46818584,
                                    3400.47281871, 3400.47481851, 3400.47948471, 3400.48148451,
                                    3400.48611738, 3400.48811718, 3400.49278338, 3400.49478318,
                                    3400.49941605, 3400.50141585, 3400.50608205, 3400.50808185,
                                    3400.51271472, 3400.51471452, 3400.51938072, 3400.52138052,
                                    3400.52601339, 3400.52801319, 3400.53267939, 3400.53467919,
                                    3400.53931206, 3400.54131186, 3400.54597806, 3400.54797786,
                                    3400.55261073, 3400.55461053, 3400.55927673, 3400.5612432,
                                    3400.5659094, 3400.5679092, 3400.5725754, 3400.57454187,
                                    3400.57920807, 3400.58120787, 3400.58587407, 3400.58784054,
                                    3400.59250674, 3400.59450654, 3400.59917274, 3400.60113921,
                                    3400.60580541, 3400.60780521, 3400.61243808, 3400.61443788,
                                    3400.61910408, 3400.62110388, 3400.62573675, 3400.62773655,
                                    3400.63240275, 3400.63440255, 3400.63903542, 3400.64103522,
                                    3400.64570142, 3400.64770122, 3400.65233409, 3400.65433389,
                                    3400.65900009, 3400.66099989, 3400.66563276, 3400.66763256,
                                    3400.67229876, 3400.67429856, 3400.67893143, 3400.68093123,
                                    3400.68559743, 3400.68759723, 3400.6922301, 3400.6942299,
                                    3400.6988961, 3400.7008959, 3400.70552877, 3400.70752857,
                                    3400.71219477, 3400.71419457, 3400.71882744, 3400.72082724,
                                    3400.72549344, 3400.72749324, 3400.73212611, 3400.73412591,
                                    3400.73879211, 3400.74079191, 3400.74542478, 3400.74742458,
                                    3400.75209078, 3400.75405725, 3400.75872345, 3400.76072325,
                                    3400.76538945, 3400.76735592, 3400.77202212, 3400.77402192,
                                    3400.77868812, 3400.78065459, 3400.78532079, 3400.78732059,
                                    3400.79198679, 3400.79395326, 3400.79861946, 3400.80061926]),
                 'polarities': np.array([1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                         -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                         1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                         -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                         1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                         -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                         1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                         -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                         1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                         -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                         1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                         -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                                         1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.,
                                         -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.])
                 }
        audio_ = ephys_fpga._clean_audio(audio)
        expected = {'times': np.array([3399.4090251, 3399.50411559, 3400.306602, 3400.80061926]),
                    'polarities': np.array([1., -1., 1., -1.])}
        assert all([np.all(audio_[k] == expected[k]) for k in audio_])

    def test_audio_ttl_start_up_down(self):
        """
        If the behaviour starts before FPGA, it is very unlikely but possible that the FPGA
        starts on up state and that the first front of the audio is a downgoing one.
        The extraction should handle both cases seamlessly: cf eid d839491f-55d8-4cbe-a298-7839208ba12b
        """

        def _test_audio(audio):
            ready, error = ephys_fpga._assign_events_audio(audio['times'], audio['polarities'])
            assert np.all(ready == audio['times'][audio['ready_tone']])
            assert np.all(error == audio['times'][audio['error_tone']])
        audio = {
            'times': np.array([1740.1032, 1740.20176667, 1741.0786, 1741.57713333, 1744.78716667, 1744.88573333]),
            'polarities': np.array([1., -1., 1., -1., 1., -1.]),
            'error_tone': np.array([False, False, True, False, False, False]),
            'ready_tone': np.array([True, False, False, False, True, False])
        }
        _test_audio(audio)  # this tests the usual pulses
        _test_audio({k: audio[k][1:] for k in audio})  # this tests when it starts in upstate

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


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

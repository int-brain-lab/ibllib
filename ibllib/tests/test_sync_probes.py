import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from iblutil.util import Bunch

from ibllib.ephys import sync_probes
from ibllib.exceptions import SyncFrontsAnomaly


class TestClassifySyncAnomaly(unittest.TestCase):
    """classify_sync_anomaly on synthetic single-channel front times."""

    def _regular_times(self, n=200, period=1.0):
        return np.arange(n, dtype=float) * period

    def test_clean_returns_none(self):
        times = self._regular_times()
        self.assertIsNone(sync_probes.classify_sync_anomaly(times))

    def test_dropped_edges(self):
        # sustained rate doubling for the last ~15% of pulses, matching the audit's
        # dropped_edges signature (probe free-runs at ~2x nominal rate past a truncation point).
        n_normal = 170
        n_fast = 30
        normal = np.arange(n_normal, dtype=float)
        fast = normal[-1] + np.arange(1, n_fast + 1) * 0.5
        times = np.r_[normal, fast]
        self.assertEqual(sync_probes.classify_sync_anomaly(times), 'dropped_edges')

    def test_duplicate_burst(self):
        # one interval far shorter than nominal -- a bounced/double-triggered edge.
        times = self._regular_times()
        times[100] = times[99] + 1e-3  # squeeze pulse 100 right next to pulse 99
        self.assertEqual(sync_probes.classify_sync_anomaly(times), 'duplicate_burst')

    def test_single_edge_glitch(self):
        # one isolated edge mistimed by a large absolute amount, but not a burst
        # (deviation stays well above the 0.2x-median burst threshold).
        times = self._regular_times()
        times[100:] += 0.05  # 50ms one-off shift, well past glitch_abs_thresh (5ms)
        self.assertEqual(sync_probes.classify_sync_anomaly(times), 'single_edge_glitch')

    def test_too_few_points_is_clean(self):
        self.assertIsNone(sync_probes.classify_sync_anomaly(np.array([0.0, 1.0, 2.0])))


def _make_ephys_file(path, ap):
    return Bunch({'path': Path(path), 'ap': Path(ap)})


class TestVersion3BAnomalyRaising(unittest.TestCase):
    """version3B(raise_on_anomaly=...) wiring, with all file I/O mocked out."""

    def setUp(self):
        n = 200
        nidq_times = np.arange(n, dtype=float)
        probe_times = nidq_times.copy()
        probe_times[100] = probe_times[99] + 1e-3  # duplicate_burst on the probe channel

        nidq_ef = _make_ephys_file('nidq', 'nidq.ap.meta')
        nidq_ef['nidq'] = True
        nidq_ef['sync'] = Bunch({
            'channels': np.full(n, 6),
            'times': nidq_times,
            'polarities': np.ones(n, dtype=int),
        })
        probe_ef = _make_ephys_file('probe00', 'probe00.ap.meta')
        probe_ef['sync'] = Bunch({
            'channels': np.full(n, 6),
            'times': probe_times,
            'polarities': np.ones(n, dtype=int),
        })
        self.ephys_files = [nidq_ef, probe_ef]

        self.patches = [
            mock.patch.object(sync_probes.spikeglx, 'glob_ephys_files', return_value=self.ephys_files),
            mock.patch.object(
                sync_probes.alfio,
                'load_object',
                side_effect=lambda path, *a, **k: nidq_ef['sync'] if path == nidq_ef.path else probe_ef['sync'],
            ),
            mock.patch.object(sync_probes, 'get_ibl_sync_map', return_value={'imec_sync': 6}),
            mock.patch.object(sync_probes, '_get_sr', return_value=30000.0),
            mock.patch.object(sync_probes, '_save_timestamps_npy', return_value=['fake_sync.npy', 'fake_timestamps.npy']),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_raises_when_opted_in(self):
        with self.assertRaises(SyncFrontsAnomaly):
            sync_probes.version3B('fake_session', display=False, raise_on_anomaly=True)

    def test_does_not_raise_by_default(self):
        qc_all, out_files = sync_probes.version3B('fake_session', display=False)
        self.assertFalse(qc_all)
        self.assertTrue(len(out_files) > 0)

    def test_does_not_raise_when_explicitly_disabled(self):
        qc_all, out_files = sync_probes.version3B('fake_session', display=False, raise_on_anomaly=False)
        self.assertFalse(qc_all)


class TestVersion3AAnomalyRaising(unittest.TestCase):
    """version3A(raise_on_anomaly=...) wiring, with sync_probe_front_times mocked to fail qc."""

    def setUp(self):
        n = 200
        ref_times = np.arange(n, dtype=float)
        probe_times = ref_times.copy()

        # .ap.parent (not .path) is what version3A's inner get_sync_fronts passes to
        # alfio.load_object -- give each probe its own parent directory so they resolve
        # to distinct fixture entries below.
        ephys_files = [
            _make_ephys_file('probe00', 'probe00/probe00.ap.meta'),
            _make_ephys_file('probe01', 'probe01/probe01.ap.meta'),
        ]
        sync_by_path = {
            ephys_files[0].ap.parent: Bunch({'channels': np.full(n, 2), 'times': ref_times}),
            ephys_files[1].ap.parent: Bunch({'channels': np.full(n, 2), 'times': probe_times}),
        }

        self.patches = [
            mock.patch.object(sync_probes.spikeglx, 'glob_ephys_files', return_value=ephys_files),
            mock.patch.object(sync_probes, 'alfio'),
            mock.patch.object(sync_probes, 'get_ibl_sync_map', return_value={'frame2ttl': 2}),
            mock.patch.object(sync_probes, '_get_sr', return_value=30000.0),
            mock.patch.object(sync_probes, '_save_timestamps_npy', return_value=['fake_sync.npy', 'fake_timestamps.npy']),
            # force the tolerance check to fail regardless of the (irrelevant here) input data
            mock.patch.object(sync_probes, 'sync_probe_front_times', return_value=(np.array([[0.0, 0.0], [1.0, 1.0]]), False)),
        ]
        for p in self.patches:
            mocked = p.start()
            self.addCleanup(p.stop)
        sync_probes.alfio.load_object.side_effect = lambda path, *a, **k: sync_by_path[path]

    def test_raises_when_opted_in(self):
        with self.assertRaises(SyncFrontsAnomaly):
            sync_probes.version3A('fake_session', display=False, raise_on_anomaly=True)

    def test_raises_by_default(self):
        # raise_on_anomaly defaults to True for version3A (unlike version3B): 3A is a legacy
        # extraction path, so an anomaly here is unlikely and worth failing loudly on.
        with self.assertRaises(SyncFrontsAnomaly):
            sync_probes.version3A('fake_session', display=False)

    def test_does_not_raise_when_explicitly_disabled(self):
        qc_all, out_files = sync_probes.version3A('fake_session', display=False, raise_on_anomaly=False)
        self.assertFalse(qc_all)
        self.assertTrue(len(out_files) > 0)


if __name__ == '__main__':
    unittest.main()

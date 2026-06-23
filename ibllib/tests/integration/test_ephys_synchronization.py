import numpy as np
import matplotlib.pyplot as plt

import one.alf.io as alfio
import spikeglx
import ibllib.ephys.sync_probes as sync_probes

from ci.tests import base


class TestEphysCheckList(base.IntegrationTest):
    def setUp(self):
        self.folder3a = self.data_path.joinpath('ephys/sync/sync_3A')
        self.folder3b = self.data_path.joinpath('ephys/sync/sync_3B')
        self.folder3b_single = self.data_path.joinpath('ephys/sync/sync_3B_single')
        self.folder3a_single = self.data_path.joinpath('ephys/sync/sync_3A_single')
        folder = self.data_path.joinpath('ephys', 'sync')
        for fil in folder.rglob('*.sync.npy'):
            fil.unlink()
        for fil in folder.rglob('*.timestamps.npy'):
            fil.unlink()
        self.addCleanup(plt.close, 'all')

    def test_sync_3A_single(self):
        ses_path = self.folder3a_single.joinpath('sub', '2019-08-09', '004')
        self.assertTrue(sync_probes.version3A(ses_path, display=False))
        self.assertTrue(np.all(np.load(list(
            ses_path.rglob('*.sync.npy'))[0]) == np.array([[0, 0], [1, 1]])))

    def test_sync_3A(self):
        if not self.folder3a.exists():
            return
        # the assertion is already in the files
        # test both residual smoothed and linear
        for ses_path in self.folder3a.rglob('raw_ephys_data'):
            # we switched to sync using frame2ttl on November 2019
            channel = 12 if '2019-11-05' in str(ses_path) else 2
            self.assertTrue(sync_probes.version3A(ses_path.parent, type='linear', tol=2,
                                                  display=False))
            self.assertTrue(sync_probes.version3A(ses_path.parent, display=True))
            dt = _check_session_sync(ses_path, channel=channel)
            self.assertTrue(np.all(np.abs(dt * 30000) < 2))

    def test_sync_3B(self):
        # the assertion is already in the files
        if not self.folder3b.exists():
            return
        """ First session is a pass """
        ses_path = self.folder3b.joinpath("hofer", "raw_ephys_data")
        self.assertTrue(sync_probes.version3B(ses_path.parent, type='linear', tol=10,
                                              display=False))
        self.assertTrue(sync_probes.version3B(ses_path.parent, display=False))
        dt = _check_session_sync(ses_path, 6)
        # import matplotlib.pyplot as plt
        # plt.plot(dt * 30000)
        """ Test a single probe"""
        ses_path_single = self.folder3b_single.joinpath('hofer', 'raw_ephys_data')
        qc, outputs = sync_probes.version3B(ses_path_single.parent, display=False)
        self.assertTrue(qc)
        self.assertTrue(len(outputs) == 2)
        sync_dual_probe0 = np.load(list(ses_path.rglob('*imec0.sync.npy'))[0])
        sync_single_probe0 = np.load(list(ses_path_single.rglob('*imec0.sync.npy'))[0])
        self.assertTrue(np.all(np.equal(sync_dual_probe0, sync_single_probe0)))
        self.assertTrue(np.all(np.abs(dt * 30000) < 2))
        """ Second session has sync issues """
        ses_path = self.folder3b.joinpath('cortexlab', 'KS014', '2019-12-03', '001',
                                          'raw_ephys_data')
        qc, outputs = sync_probes.version3B(ses_path.parent, display=False)
        self.assertFalse(qc)
        self.assertTrue(len(outputs) == 4)
        dt = _check_session_sync(ses_path, 6)
        # which doesn't prevent the sync function to output the desired output
        self.assertTrue(np.all(np.abs(dt * 30000) < 2))


def _check_session_sync(ses_path, channel):
    """
    Resync the original cam pulses
    :param ses_path:
    :return:
    """
    efiles = spikeglx.glob_ephys_files(ses_path, bin_exists=False)
    tprobe = []
    tinterp = []
    for ef in efiles:
        if not ef.get('ap'):
            continue
        sync_events = alfio.load_object(ef.ap.parent, 'sync', short_keys=True)
        # the first step is to construct list arrays with probe sync
        sync_file = ef.ap.parent.joinpath(ef.ap.name.replace('.ap.', '.sync.')).with_suffix('.npy')
        t = sync_events.times[sync_events.channels == channel]
        tsync = sync_probes.apply_sync(sync_file, t, forward=True)
        tprobe.append(t)
        tinterp.append(tsync)
        # the second step is to make sure sample / time_ref files match time / time_ref files
        ts_file = ef.ap.parent.joinpath(ef.ap.name.replace('.ap.', '.timestamps.')
                                        ).with_suffix('.npy')
        fs = spikeglx._get_fs_from_meta(spikeglx.read_meta_data(ef.ap.with_suffix('.meta')))
        tstamp = sync_probes.apply_sync(ts_file, t * fs, forward=True)
        assert np.all(tstamp - tsync < 1e-12)
    return tinterp[0] - tinterp[1]


if __name__ == "__main__":
    import unittest
    unittest.main(exit=False, verbosity=2)

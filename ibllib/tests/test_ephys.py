# Mock dataset
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from ibllib.ephys import ephysqc, neuropixel
from ibllib.tests import TEST_DB
from ibllib.tests.fixtures import utils
from one.api import ONE


class TestNeuropixel(unittest.TestCase):

    def test_layouts(self):
        dense = neuropixel.dense_layout()
        assert set(dense.keys()) == set(['x', 'y', 'row', 'col', 'ind'])
        xu = np.unique(dense['x'])
        yu = np.unique(dense['y'])
        assert np.all(np.diff(xu) == 16)
        assert np.all(np.diff(yu) == 20)
        assert xu.size == 4 and yu.size == 384 / 2

    def tests_headers(self):
        th = neuropixel.trace_header()
        assert set(th.keys()) == set(['x', 'y', 'row', 'col', 'ind', 'adc', 'sample_shift'])


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


class TestEphysQC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = TemporaryDirectory()
        cls.one = ONE(**TEST_DB, cache_dir=cls.tempdir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clear overwritten methods by destroying cached instance
        ONE.cache_clear()
        cls.tempdir.cleanup()

    def setUp(self) -> None:

        self.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        # make a temp probe insertion
        self.pname = 'probe02'
        # Find any existing insertions with this name and delete
        probe_insertions = self.one.alyx.rest('insertions', 'list', session=self.eid, name=self.pname, no_cache=True)
        for pi in probe_insertions:
            self.one.alyx.rest('insertions', 'delete', pi['id'])
        # Create new insertion with this name and add teardown hook to delete it
        probe_insertion = self.one.alyx.rest('insertions', 'create', data={'session': self.eid, 'name': self.pname})
        self.addCleanup(self.one.alyx.rest, 'insertions', 'delete', id=probe_insertion['id'])
        self.pid = probe_insertion['id']
        self.qc = ephysqc.EphysQC(self.pid, one=self.one)

    def tearDown(self) -> None:
        pass

    def test_ensure_data(self):
        # Make sure raises an error when no data present
        utils.create_fake_raw_ephys_data_folder(self.one.eid2path(self.eid), populate=False)
        with self.assertRaises(AssertionError):
            self.qc._ensure_required_data()
        # Make sure it runs through fine when meta files are present
        utils.create_fake_raw_ephys_data_folder(self.one.eid2path(self.eid), populate=True)
        self.qc._ensure_required_data()

    def test_load_data(self):
        # In case that hasn't been run
        utils.create_fake_raw_ephys_data_folder(self.one.eid2path(self.eid), populate=True)
        # Remove the fake bin files because they won't be able to load
        for fbin in ['_spikeglx_ephysData_g0_t0.imec.lf.bin', '_spikeglx_ephysData_g0_t0.imec.ap.bin']:
            self.one.eid2path(self.eid).joinpath('raw_ephys_data', self.pname, fbin).unlink()
        self.qc.load_data()


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

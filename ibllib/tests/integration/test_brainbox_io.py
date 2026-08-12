import logging
import hashlib
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from one.api import One, ONE
from one.alf.io import load_object
import brainbox.io.one as bbone
from neuropixel import trace_header
from iblatlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader, SessionLoader

from ibllib.tests.base import IntegrationTest

_logger = logging.getLogger('ibllib')
_logger.setLevel(10)
br = BrainRegions()


class TestSpikeInterface(unittest.TestCase):

    def test_spike_interface(self):
        """
        Those are the specifications for the spike interface tests to pass.
        :return:
        """
        one = ONE(
            base_url='https://openalyx.internationalbrainlab.org',
            silent=True,
            password='international'
        )
        pid = '80f6ffdd-f692-450f-ab19-cd6d45bfd73e'
        ssl = SpikeSortingLoader(pid=pid, one=one)

        clusters = ssl.load_spike_sorting_object('clusters', revision='2023-12-05')
        assert clusters['uuids'].shape[0] == 733

        clusters = ssl.load_spike_sorting_object('clusters', revision='2024-05-06')
        assert clusters['uuids'].shape[0] == 1091

        spikes, clusters, channels = ssl.load_spike_sorting()
        assert clusters['uuids'].shape[0] == 1091


class TestReadChannels(unittest.TestCase):

    def test_read_channels(self):
        one = ONE(
            base_url='https://openalyx.internationalbrainlab.org',
            silent=True,
            password='international'
        )
        pid = '511afaa5-fdc4-4166-b4c0-4629ec5e652e'
        ssl = SpikeSortingLoader(one=one, pid=pid)
        channels = ssl.load_channels(revision='2024-05-06')
        np.testing.assert_array_equal(
            pd.Series(channels['atlas_id']).value_counts().to_numpy(),
            np.array([206, 130, 48]),
        )


class TestReadSpikeSorting(IntegrationTest):

    required_files = ['brainbox/io/spike_sorting']
    _writable_scope = 'test'

    def setUp(self) -> None:
        super().setUp()
        self.root_path = self.data_path.joinpath('brainbox/io/spike_sorting')
        self.session_path = self.root_path.joinpath('SWC_054/2020-10-05/001')
        print('Building ONE cache from filesystem...')
        self.one = One.setup(self.root_path, silent=True)
        self.pname = 'probe01'
        self.eid = self.one.path2eid(self.session_path)

    def _check(self, spike_times, spike_sorter='pykilosort'):
        if spike_sorter == 'pykilosort':
            hash = 'f66c53aec01333245acc2fd658339ee9fceda5b9'
        elif spike_sorter == 'ks2_preproc_tests':
            hash = 'e833cd9df46aa791fcec48b52ddff7f96f98a6ab'
        elif spike_sorter == '':
            hash = '5628850285f44fba66ea241cd106d8c7c1871754'
        self.assertEqual(hashlib.sha1(spike_times.tobytes()).hexdigest(), hash)

    def _check_spike_clusters(self, spike_clusters, namespace=None):
        if namespace is None:
            hash = '99946f9e8565d1bf5afdd76e9b13b84c4cbb47c8'
        elif namespace == 'av':
            hash = '56d6e9d33bf418fb944005147648084426301b73'
        elif namespace == 'mf':
            hash = '99946f9e8565d1bf5afdd76e9b13b84c4cbb47c8'
        self.assertEqual(hashlib.sha1(spike_clusters.tobytes()).hexdigest(), hash)

    def _check_cluster_depths(self, cluster_depths, namespace=None):
        if namespace is None:
            hash = 'e1c193f46bffbb9e44ff5fe084a65bfb55118987'
        elif namespace == 'av':
            hash = '86846d9a42a8f66127a61b0d8aa2886f224567f8'
        elif namespace == 'mf':
            hash = 'e1c193f46bffbb9e44ff5fe084a65bfb55118987'
        self.assertEqual(hashlib.sha1(cluster_depths.tobytes()).hexdigest(), hash)

    def test_channel_conversion_interpolation(self):
        BUNCH_KEYS = {'x', 'y', 'z', 'acronym', 'atlas_id', 'axial_um', 'lateral_um'}
        ALF_KEYS = {'localCoordinates', 'mlapdv', 'brainLocationIds_ccf_2017'}
        pname = 'probe01'
        alf_channels = load_object(self.session_path.joinpath('alf', pname), 'channels')
        channels = bbone._channels_alf2bunch(alf_channels)
        assert BUNCH_KEYS.issubset(set(channels.keys()))

        h = trace_header(1)
        raw_channels = bbone.channel_locations_interpolation(
            alf_channels, {'localCoordinates': np.c_[h['x'], h['y']]})
        assert set(raw_channels.keys()) == ALF_KEYS
        channels = bbone.channel_locations_interpolation(
            alf_channels, {'localCoordinates': np.c_[h['x'], h['y']]}, brain_regions=br)
        assert set(channels.keys()) == BUNCH_KEYS

        # this function should also be able to take a bunch formatted dict as input
        channels = bbone.channel_locations_interpolation(
            channels, {'localCoordinates': np.c_[h['x'], h['y']]}, brain_regions=br)
        assert set(channels.keys()) == BUNCH_KEYS

    def test_display_spike_sorting(self):
        sl = SpikeSortingLoader(eid=self.eid, pname=self.pname, one=self.one)
        _logger.setLevel(0)
        spikes, _, channels = sl.load_spike_sorting(spike_sorter='')
        fig, ax = sl.raster(spikes, channels)
        plt.close(fig)

    def test_read_spike_sorting(self):
        sl = SpikeSortingLoader(eid=self.eid, pname=self.pname, one=self.one)
        self.assertEqual(sl.pid2ref, '2020-10-05_1_SWC_054_probe01')
        _logger.setLevel(0)
        spikes, clusters, channels = sl.load_spike_sorting(spike_sorter='')
        self._check(spikes['times'], spike_sorter='')
        clusters = sl.merge_clusters(spikes, clusters, channels)
        assert 'acronym' in clusters.keys()

        # load spike sorting for a non default sorter
        spikes, clusters, channels = sl.load_spike_sorting(spike_sorter='ks2_preproc_tests')
        self._check(spikes['times'], spike_sorter='ks2_preproc_tests')

        # load spike sorting using collection
        # this is not recommended as the spike sorter property doesn not match the spike sorting loaded
        spikes, clusters, channels = sl.load_spike_sorting(
            collection=f'alf/{self.pname}/ks2_preproc_tests', enforce_version=False)
        self._check(spikes['times'], spike_sorter='ks2_preproc_tests')

        # makes sure this is the pykilosort that is returned by default1
        spikes, clusters, channels = bbone._load_spike_sorting(
            eid=self.eid, one=self.one, collection=f'alf/*{self.pname}/*', return_channels=True)
        self._check(spikes[self.pname]['times'])

        # Tests for loading of manually curated datasets
        # For None and 'mf' it should load the default spikesorted data (as mf data doesn't exist). For
        # namespace = 'av' it should read in the av clusters objects and replace the spikes.clusters with the av version.
        for namespace in [None, 'av', 'mf']:
            with self.subTest(namespace=namespace):
                spikes, clusters, channels = sl.load_spike_sorting(enforce_version=False, namespace=namespace,
                                                                   dataset_types=['clusters.curatedLabels'])
                self._check(spikes['times'], spike_sorter='pykilosort')
                self._check_spike_clusters(spikes['clusters'], namespace=namespace)
                self._check_cluster_depths(clusters['depths'], namespace=namespace)
                if namespace == 'av':
                    assert 'curatedLabels' in clusters.keys()
                else:
                    assert 'curatedLabels' not in clusters.keys()

        # Check that it isn't possible to load spikesorting with good_units=True and namespace not None
        spikes, clusters, channels = sl.load_spike_sorting(enforce_version=False, namespace='av',
                                                           good_units=True)
        assert spikes is None
        assert clusters is None
        assert channels is None

        # this dataset contains no raw data whatsoever
        self.assertEqual(len(sl.download_raw_electrophysiology('lf')), 0)

    def test_samples2times(self):
        #  eid original alyx '56b57c38-2699-4091-90a8-aba35103155e'
        # relative path: brainbox/io/spike_sorting/SWC_054/2020-10-05/001'
        pname = 'probe01'
        one = self.one
        eid = one.path2eid(self.session_path)
        sl = SpikeSortingLoader(eid=eid, pname=pname, one=one)
        _logger.setLevel(0)
        spikes, _, _ = sl.load_spike_sorting(spike_sorter='', dataset_types=['spikes.samples'])
        self.assertTrue(np.all(np.abs(sl.samples2times(spikes.samples) - spikes.times) < 1e11))


class TestSessionLoader(IntegrationTest):

    required_files = ['ephys/choice_world_init/KS022/2019-12-10/001']

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.root_path = cls.default_data_root().joinpath('ephys', 'choice_world_init')
        if not cls.root_path.exists():
            return
        cls.session_path = cls.root_path.joinpath('KS022', '2019-12-10', '001')
        print('Building ONE cache from filesystem...')
        cls.one = One.setup(cls.root_path, silent=True)
        cls.sess_loader = SessionLoader(one=cls.one, session_path=cls.session_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._writable_tempdir is None and hasattr(cls, 'root_path') and cls.root_path.exists():
            for file in cls.root_path.glob('*.pqt'):
                file.unlink()
        super().tearDownClass()

    def test_load_trials_data(self):
        expected = [
            'stimOff_times', 'goCueTrigger_times', 'intervals_bpod_0', 'intervals_bpod_1',
            'probabilityLeft', 'contrastRight', 'firstMovement_times', 'goCue_times', 'feedbackType', 'choice',
            'contrastLeft', 'stimOn_times', 'rewardVolume', 'feedback_times', 'response_times',
            'intervals_0', 'intervals_1'
        ]
        self.sess_loader.load_trials()
        self.assertCountEqual(expected, self.sess_loader.trials.columns)
        self.assertEqual((626, 17), self.sess_loader.trials.shape)

    def test_load_wheel(self):
        self.sess_loader.load_wheel(fs=100, corner_frequency=20, order=8)
        self.assertCountEqual(['times', 'position', 'velocity', 'acceleration'], self.sess_loader.wheel.columns)

    def test_load_pose(self):
        self.sess_loader.load_pose(likelihood_thr=0.9, views=['left', 'body'])
        self.assertIsInstance(self.sess_loader.pose, dict)
        self.assertCountEqual(['leftCamera', 'bodyCamera'], self.sess_loader.pose.keys())
        self.assertIn('times', self.sess_loader.pose['leftCamera'].columns)
        self.assertCountEqual(['times', 'tail_start_x', 'tail_start_y', 'tail_start_likelihood'],
                              self.sess_loader.pose['bodyCamera'].columns)
        self.assertEqual((4000, 4), self.sess_loader.pose['bodyCamera'].shape)
        self.assertEqual((4000, 34), self.sess_loader.pose['leftCamera'].shape)

        all_nan = [c for c in self.sess_loader.pose['leftCamera'].columns if
                   all(np.isnan(self.sess_loader.pose['leftCamera'][c]))]
        self.assertCountEqual(['pupil_bottom_r_x', 'pupil_bottom_r_y'], all_nan)

        self.sess_loader.load_pose(likelihood_thr=0.5, views=['left'])
        all_nan = [c for c in self.sess_loader.pose['leftCamera'].columns if
                   all(np.isnan(self.sess_loader.pose['leftCamera'][c]))]
        self.assertTrue(len(all_nan) == 0)

    def test_load_motion_energy(self):
        self.sess_loader.load_motion_energy()
        self.assertIsInstance(self.sess_loader.motion_energy, dict)
        self.assertCountEqual(['leftCamera', 'rightCamera', 'bodyCamera'], self.sess_loader.motion_energy.keys())
        self.assertCountEqual(['times', 'whiskerMotionEnergy'], self.sess_loader.motion_energy['leftCamera'].columns)
        self.assertCountEqual(['times', 'whiskerMotionEnergy'], self.sess_loader.motion_energy['rightCamera'].columns)
        self.assertCountEqual(['times', 'bodyMotionEnergy'], self.sess_loader.motion_energy['bodyCamera'].columns)
        self.assertCountEqual([158377, 396504, 79095], [df.shape[0] for df in self.sess_loader.motion_energy.values()])

        self.sess_loader.load_motion_energy(views=['left'])
        self.assertCountEqual(['leftCamera'], self.sess_loader.motion_energy.keys())

    def test_load_pupil(self):
        self.sess_loader.load_pupil()
        self.assertCountEqual(['pupilDiameter_raw', 'pupilDiameter_smooth'], self.sess_loader.pupil.columns)
        self.assertEqual(4000, self.sess_loader.pupil.shape[0])

        with self.assertRaises(ValueError):
            self.sess_loader.load_pupil(snr_thresh=20)
            self.assertTrue(self.sess_loader.pupil.empty)

    def test_load_session_data(self):
        # Instantiate new session loader
        self.sess_loader = SessionLoader(one=self.one, session_path=self.session_path)
        self.sess_loader.load_session_data()
        self.assertTrue(all(self.sess_loader.data_info['is_loaded']))

        # Make sure data is not reloaded
        with self.assertLogs(_logger, level='DEBUG') as cm:
            self.sess_loader.load_session_data()
            self.assertEqual([
                'DEBUG:ibllib:Not loading trials data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading wheel data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading pose data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading motion_energy data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading pupil data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading licks data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading pawstates data, is already loaded and reload=False.'

            ], cm.output)
        # Make sure data IS reloaded
        with self.assertLogs(_logger, level='INFO') as cm:
            self.sess_loader.load_session_data(reload=True)
            self.assertEqual([
                'INFO:ibllib:Loading trials data',
                'INFO:ibllib:Loading wheel data',
                'INFO:ibllib:Loading pose data',
                'INFO:ibllib:Loading motion_energy data',
                'INFO:ibllib:Loading pupil data',
                'INFO:ibllib:Pupil diameter not available, trying to compute on the fly.',
                'INFO:ibllib:Loading licks data',
                'INFO:ibllib:Loading pawstates data',
            ], cm.output)

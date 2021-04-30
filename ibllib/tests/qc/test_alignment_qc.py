import unittest
from pathlib import Path
import numpy as np
import copy

from oneibl.one import ONE
from ibllib.atlas import AllenAtlas
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.qc.alignment_qc import AlignmentQC
from ibllib.pipes.histology import register_track


EPHYS_SESSION = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
one = ONE(username='test_user', password='TapetesBloc18',
          base_url='https://test.alyx.internationalbrainlab.org')
brain_atlas = AllenAtlas(25)


class TestProbeInsertion(unittest.TestCase):

    def test_creation(self):
        probe = ['probe00', 'probe01']
        create_alyx_probe_insertions(session_path=EPHYS_SESSION, model='3B2', labels=probe,
                                     one=one, force=True)
        insertion = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION)
        assert(len(insertion) == 2)
        assert (insertion[0]['json']['qc'] == 'NOT_SET')
        assert (len(insertion[0]['json']['extended_qc']) == 0)

        one.alyx.rest('insertions', 'delete', id=insertion[0]['id'])
        one.alyx.rest('insertions', 'delete', id=insertion[1]['id'])


class TestHistologyQc(unittest.TestCase):

    def test_session_creation(self):
        pass

    def test_probe_qc(self):
        pass


class TestTracingQc(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        probe = ['probe00', 'probe01']
        create_alyx_probe_insertions(session_path=EPHYS_SESSION, model='3B2', labels=probe,
                                     one=one, force=True)
        cls.probe00_id = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION,
                                       name='probe00')[0]['id']
        cls.probe01_id = one.alyx.rest('insertions', 'list', session=EPHYS_SESSION,
                                       name='probe01')[0]['id']
        data = np.load(Path(Path(__file__).parent.parent.
                            joinpath('fixtures', 'qc', 'data_alignmentqc_existing.npz')),
                       allow_pickle=True)
        cls.xyz_picks = np.array(data['xyz_picks']) / 1e6

    def test_tracing_exists(self):
        register_track(self.probe00_id, picks=self.xyz_picks, one=one, overwrite=True,
                       channels=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe00_id)

        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['tracing_exists'] == 1)

    def test_tracing_not_exists(self):
        register_track(self.probe01_id, picks=None, one=one, overwrite=True,
                       channels=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe01_id)
        assert (insertion['json']['qc'] == 'CRITICAL')
        assert (insertion['json']['extended_qc']['tracing_exists'] == 0)

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe01_id)
        one.alyx.rest('insertions', 'delete', id=cls.probe00_id)


class TestAlignmentQcExisting(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        data = np.load(Path(Path(__file__).parent.parent.
                            joinpath('fixtures', 'qc', 'data_alignmentqc_existing.npz')),
                       allow_pickle=True)
        cls.xyz_picks = data['xyz_picks'].tolist()
        cls.alignments = data['alignments'].tolist()
        # Manipulate so one alignment disagrees
        cls.alignments['2020-06-26T16:40:14_Karolina_Socha'][1] = \
            list(np.array(cls.alignments['2020-06-26T16:40:14_Karolina_Socha'][1]) + 0.0001)
        cls.cluster_chns = data['cluster_chns']
        insertion = data['insertion'].tolist()
        insertion['json'] = {'xyz_picks': cls.xyz_picks}
        probe_insertion = one.alyx.rest('insertions', 'create', data=insertion)
        cls.probe_id = probe_insertion['id']
        cls.trajectory = data['trajectory'].tolist()
        cls.trajectory.update({'probe_insertion': cls.probe_id})

    def setUp(self):
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        if traj:
            self.prev_traj_id = traj[0]['id']

    def test_01_no_alignments(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (len(insertion['json']['extended_qc']) == 0)

    def test_02_one_alignment(self):
        alignments = {'2020-06-26T16:40:14_Karolina_Socha':
                      self.alignments['2020-06-26T16:40:14_Karolina_Socha']}
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        _ = one.alyx.rest('trajectories', 'create', data=trajectory)
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['alignment_count'] == 1)
        assert (insertion['json']['extended_qc']['alignment_stored'] ==
                '2020-06-26T16:40:14_Karolina_Socha')
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 0)

    def test_03_alignments_disagree(self):
        alignments = {'2020-06-26T16:40:14_Karolina_Socha':
                      self.alignments['2020-06-26T16:40:14_Karolina_Socha'],
                      '2020-06-12T00:39:15_nate': self.alignments['2020-06-12T00:39:15_nate']}
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        traj = one.alyx.rest('trajectories', 'update', id=self.prev_traj_id, data=trajectory)
        traj_id = traj['id']
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        align_qc.load_data(prev_alignments=traj['json'], xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['alignment_count'] == 2)
        assert (insertion['json']['extended_qc']['alignment_stored'] ==
                '2020-06-26T16:40:14_Karolina_Socha')
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 0)
        assert (insertion['json']['extended_qc']['alignment_qc'] < 0.8)
        assert (np.isclose(insertion['json']['extended_qc']['alignment_qc'], 0.782216))
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        assert(traj_id == traj[0]['id'])

    def test_04_alignments_agree(self):
        alignments = {'2020-06-19T10:52:36_noam.roth':
                      self.alignments['2020-06-19T10:52:36_noam.roth'],
                      '2020-06-12T00:39:15_nate': self.alignments['2020-06-12T00:39:15_nate']}
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        traj = one.alyx.rest('trajectories', 'update', id=self.prev_traj_id, data=trajectory)
        assert(self.prev_traj_id == traj['id'])
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(cluster_chns=self.cluster_chns)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['alignment_count'] == 2)
        assert (insertion['json']['extended_qc']['alignment_stored'] ==
                '2020-06-19T10:52:36_noam.roth')
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['alignment_resolved_by'] == 'qc')
        assert (insertion['json']['extended_qc']['alignment_qc'] > 0.8)
        assert (np.isclose(insertion['json']['extended_qc']['alignment_qc'], 0.952319))
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        assert(self.prev_traj_id == traj[0]['id'])

    def test_05_not_latest_alignments_agree(self):
        alignments = copy.deepcopy(self.alignments)
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        traj = one.alyx.rest('trajectories', 'update', id=self.prev_traj_id, data=trajectory)
        assert(self.prev_traj_id == traj['id'])
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=traj['json'], xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns)
        align_qc.resolved = 0
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['alignment_count'] == 4)
        assert (insertion['json']['extended_qc']['alignment_stored'] ==
                '2020-06-19T10:52:36_noam.roth')
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['alignment_resolved_by'] == 'qc')
        assert (insertion['json']['extended_qc']['alignment_qc'] > 0.8)
        assert (np.isclose(insertion['json']['extended_qc']['alignment_qc'], 0.952319))
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        assert(self.prev_traj_id != traj[0]['id'])

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe_id)


class TestAlignmentQcManual(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = np.load(Path(Path(__file__).parent.parent.
                            joinpath('fixtures', 'qc', 'data_alignmentqc_manual.npz')),
                       allow_pickle=True)
        cls.xyz_picks = (data['xyz_picks'] * 1e6).tolist()
        cls.alignments = data['alignments'].tolist()
        cls.cluster_chns = data['cluster_chns']

        data = np.load(Path(Path(__file__).parent.parent.
                            joinpath('fixtures', 'qc', 'data_alignmentqc_existing.npz')),
                       allow_pickle=True)
        insertion = data['insertion'].tolist()
        insertion['json'] = {'xyz_picks': cls.xyz_picks}
        probe_insertion = one.alyx.rest('insertions', 'create', data=insertion)
        cls.probe_id = probe_insertion['id']
        cls.trajectory = data['trajectory'].tolist()
        cls.trajectory.update({'probe_insertion': cls.probe_id})
        cls.trajectory.update({'json': cls.alignments})
        cls.traj = one.alyx.rest('trajectories', 'create', data=cls.trajectory)

    def setUp(self) -> None:
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        if traj:
            self.prev_traj_id = traj[0]['id']

    def test_01_normal_computation(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=self.traj['json'],
                           xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['alignment_count'] == 3)
        assert (insertion['json']['extended_qc']['alignment_stored'] ==
                '2020-09-28T15:57:25_mayo')
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 0)
        assert (insertion['json']['extended_qc']['alignment_qc'] < 0.8)
        assert (np.isclose(insertion['json']['extended_qc']['alignment_qc'], 0.604081))
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        assert(self.prev_traj_id == traj[0]['id'])

    def test_02_manual_resolution_latest(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=self.traj['json'],
                           xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns)
        align_qc.resolve_manual('2020-09-28T15:57:25_mayo', update=True, upload_alyx=True,
                                upload_flatiron=False)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['alignment_count'] == 3)
        assert (insertion['json']['extended_qc']['alignment_stored'] ==
                '2020-09-28T15:57:25_mayo')
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['alignment_resolved_by'] == 'experimenter')
        assert (insertion['json']['extended_qc']['alignment_qc'] < 0.8)
        assert (np.isclose(insertion['json']['extended_qc']['alignment_qc'], 0.604081))
        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        assert(self.prev_traj_id == traj[0]['id'])

    def test_03_manual_resolution_not_latest(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=self.traj['json'],
                           xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns)
        align_qc.resolve_manual('2020-09-28T10:03:06_alejandro', update=True, upload_alyx=True,
                                upload_flatiron=False, force=True)
        insertion = one.alyx.rest('insertions', 'read', id=self.probe_id)
        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['alignment_count'] == 3)
        assert (insertion['json']['extended_qc']['alignment_stored'] ==
                '2020-09-28T10:03:06_alejandro')
        assert (insertion['json']['extended_qc']['alignment_resolved'] == 1)
        assert (insertion['json']['extended_qc']['alignment_resolved_by'] == 'experimenter')
        assert (insertion['json']['extended_qc']['alignment_qc'] < 0.8)
        assert (np.isclose(insertion['json']['extended_qc']['alignment_qc'], 0.604081))

        traj = one.alyx.rest('trajectories', 'list', probe_id=self.probe_id,
                             provenance='Ephys aligned histology track')
        assert(self.prev_traj_id != traj[0]['id'])

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe_id)


class TestUploadToFlatIron(unittest.TestCase):
    @unittest.skip("Skip FTP upload test")
    @classmethod
    def setUpClass(cls) -> None:
        data = np.load(Path(Path(__file__).parent.parent.
                            joinpath('fixtures', 'qc', 'data_alignmentqc_manual.npz')),
                       allow_pickle=True)
        cls.xyz_picks = (data['xyz_picks'] * 1e6).tolist()
        cls.alignments = data['alignments'].tolist()
        cls.cluster_chns = data['cluster_chns']

        data = np.load(Path(Path(__file__).parent.parent.
                            joinpath('fixtures', 'qc', 'data_alignmentqc_existing.npz')),
                       allow_pickle=True)
        insertion = data['insertion'].tolist()
        insertion['json'] = {'xyz_picks': cls.xyz_picks}
        probe_insertion = one.alyx.rest('insertions', 'create', data=insertion)
        cls.probe_id = probe_insertion['id']
        cls.probe_name = probe_insertion['name']
        cls.trajectory = data['trajectory'].tolist()
        cls.trajectory.update({'probe_insertion': cls.probe_id})
        cls.trajectory.update({'json': cls.alignments})
        cls.traj = one.alyx.rest('trajectories', 'create', data=cls.trajectory)

        align_qc = AlignmentQC(cls.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=cls.traj['json'],
                           xyz_picks=np.array(cls.xyz_picks) / 1e6,
                           cluster_chns=cls.cluster_chns)
        cls.file_paths = align_qc.resolve_manual('2020-09-28T15:57:25_mayo', update=True,
                                                 upload_alyx=True, upload_flatiron=True)
        print(cls.file_paths)

    def test_data_content(self):
        alf_path = one.path_from_eid(EPHYS_SESSION).joinpath('alf', self.probe_name)
        channels_mlapdv = np.load(alf_path.joinpath('channels.mlapdv.npy'))
        assert(np.all(np.abs(channels_mlapdv) > 0))
        channels_id = np.load(alf_path.joinpath('channels.brainLocationIds_ccf_2017.npy'))
        assert(channels_mlapdv.shape[0] == channels_id.shape[0])

        clusters_mlapdv = np.load(alf_path.joinpath('clusters.mlapdv.npy'))
        assert(np.all(np.abs(clusters_mlapdv) > 0))
        clusters_id = np.load(alf_path.joinpath('clusters.brainLocationIds_ccf_2017.npy'))
        assert(clusters_mlapdv.shape[0] == clusters_id.shape[0])
        assert(np.all(np.in1d(clusters_mlapdv, channels_mlapdv)))
        assert (np.all(np.in1d(clusters_id, channels_id)))
        clusters_acro = np.load(alf_path.joinpath('clusters.brainLocationAcronyms_ccf_2017.npy'),
                                allow_pickle=True)
        assert(clusters_acro.shape == clusters_id.shape)

    def test_upload_to_flatiron(self):
        for file in self.file_paths:
            file_registered = one.alyx.rest('datasets', 'list', session=EPHYS_SESSION,
                                            dataset_type=file.stem)
            data_id = file_registered[0]['url'][-36:]
            assert(len(file_registered) == 1)
            one.alyx.rest('datasets', 'delete', id=data_id)

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe_id)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

import unittest
from pathlib import Path
import re
from inspect import getmembers, ismethod

import numpy as np
import copy
import random
import string
import datetime

from one.api import ONE
from neuropixel import trace_header

from ibllib.tests import TEST_DB
from ibllib.atlas import AllenAtlas
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.qc.alignment_qc import AlignmentQC
from ibllib.pipes.histology import register_track
from one.registration import RegistrationClient


EPHYS_SESSION = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
one = ONE(**TEST_DB)
brain_atlas = AllenAtlas(25)

refch_3a = np.array([36, 75, 112, 151, 188, 227, 264, 303, 340, 379])
th = trace_header(version=1)
SITES_COORDINATES = np.delete(np.c_[th['x'], th['y']], refch_3a, axis=0)


class TestTracingQc(unittest.TestCase):
    probe01_id = None
    probe00_id = None

    @classmethod
    def setUpClass(cls) -> None:
        probe = [''.join(random.choices(string.ascii_letters, k=5)),
                 ''.join(random.choices(string.ascii_letters, k=5))]
        ins = create_alyx_probe_insertions(session_path=EPHYS_SESSION, model='3B2', labels=probe,
                                           one=one, force=True)
        cls.probe00_id, cls.probe01_id = (x['id'] for x in ins)
        data = np.load(Path(Path(__file__).parent.parent.
                            joinpath('fixtures', 'qc', 'data_alignmentqc_existing.npz')),
                       allow_pickle=True)
        cls.xyz_picks = np.array(data['xyz_picks']) / 1e6

    def test_tracing_exists(self):
        register_track(self.probe00_id, picks=self.xyz_picks, one=one, overwrite=True,
                       channels=False, brain_atlas=brain_atlas)
        insertion = one.alyx.get('/insertions/' + self.probe00_id, clobber=True)

        assert (insertion['json']['qc'] == 'NOT_SET')
        assert (insertion['json']['extended_qc']['tracing_exists'] == 1)

    def test_tracing_not_exists(self):
        register_track(self.probe01_id, picks=None, one=one, overwrite=True,
                       channels=False, brain_atlas=brain_atlas)
        insertion = one.alyx.get('/insertions/' + self.probe01_id, clobber=True)
        assert (insertion['json']['qc'] == 'CRITICAL')
        assert (insertion['json']['extended_qc']['tracing_exists'] == 0)

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe01_id)
        one.alyx.rest('insertions', 'delete', id=cls.probe00_id)


class TestAlignmentQcExisting(unittest.TestCase):
    probe_id = None
    prev_traj_id = None

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
        insertion['name'] = ''.join(random.choices(string.ascii_letters, k=5))
        insertion['json'] = {'xyz_picks': cls.xyz_picks}
        date = str(datetime.date(2019, np.random.randint(1, 12), np.random.randint(1, 28)))
        _, eid = RegistrationClient(one).create_new_session('ZM_1150', date=date)
        cls.eid = str(eid)
        # Currently the task protocol of a session must contain 'ephys' in order to create an insertion!
        one.alyx.rest('sessions', 'partial_update', id=cls.eid, data={'task_protocol': 'ephys'})
        insertion['session'] = cls.eid
        probe_insertion = one.alyx.rest('insertions', 'create', data=insertion)
        cls.probe_id = probe_insertion['id']
        cls.trajectory = data['trajectory'].tolist()
        cls.trajectory.update({'probe_insertion': cls.probe_id})

    def test_alignments(self):
        checks = getmembers(self, lambda x: ismethod(x) and re.match(r'^_\d{2}_.*', x.__name__))
        # Run numbered functions in order
        for _, fn in sorted(checks, key=lambda x: x[0]):
            self._get_prev_traj_id()
            fn()

    def _get_prev_traj_id(self):
        traj = one.alyx.get('/trajectories?'
                            f'&probe_id={self.probe_id}'
                            '&provenance=Ephys aligned histology track', clobber=True)
        if traj:
            self.prev_traj_id = traj[0]['id']

    def _01_no_alignments(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        insertion = one.alyx.get(f'/insertions/{self.probe_id}', clobber=True)
        self.assertEqual('NOT_SET', insertion['json']['qc'])
        self.assertTrue(len(insertion['json']['extended_qc']) == 0)

    def _02_one_alignment(self):
        alignments = {'2020-06-26T16:40:14_Karolina_Socha':
                      self.alignments['2020-06-26T16:40:14_Karolina_Socha']}
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        _ = one.alyx.rest('trajectories', 'create', data=trajectory)
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        _verify(self,
                alignment_count=1,
                alignment_stored='2020-06-26T16:40:14_Karolina_Socha',
                alignment_resolved=False)

    def _03_alignments_disagree(self):
        alignments = {'2020-06-26T16:40:14_Karolina_Socha':
                      self.alignments['2020-06-26T16:40:14_Karolina_Socha'],
                      '2020-06-12T00:39:15_nate': self.alignments['2020-06-12T00:39:15_nate']}
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        traj = one.alyx.rest('trajectories', 'update', id=self.prev_traj_id, data=trajectory)
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=traj['json'], xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns, depths=SITES_COORDINATES[:, 1],
                           chn_coords=SITES_COORDINATES)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)

        _verify(self, alignment_qc=0.782216, alignment_resolved=False,
                alignment_count=2, alignment_stored='2020-06-26T16:40:14_Karolina_Socha',
                trajectory_created=False)

    def _04_alignments_agree(self):
        alignments = {'2020-06-19T10:52:36_noam.roth':
                      self.alignments['2020-06-19T10:52:36_noam.roth'],
                      '2020-06-12T00:39:15_nate': self.alignments['2020-06-12T00:39:15_nate']}
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        traj = one.alyx.rest('trajectories', 'update', id=self.prev_traj_id, data=trajectory)
        self.assertEqual(self.prev_traj_id, traj['id'])
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(cluster_chns=self.cluster_chns, depths=SITES_COORDINATES[:, 1],
                           chn_coords=SITES_COORDINATES)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)

        _verify(self, alignment_resolved='qc', alignment_qc=0.952319, trajectory_created=False,
                alignment_count=2, alignment_stored='2020-06-19T10:52:36_noam.roth')

    def _05_not_latest_alignments_agree(self):
        alignments = copy.deepcopy(self.alignments)
        trajectory = copy.deepcopy(self.trajectory)
        trajectory.update({'json': alignments})
        traj = one.alyx.rest('trajectories', 'update', id=self.prev_traj_id, data=trajectory)
        self.assertEqual(self.prev_traj_id, traj['id'])
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=traj['json'], xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns, depths=SITES_COORDINATES[:, 1],
                           chn_coords=SITES_COORDINATES)
        align_qc.resolved = 0
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)

        _verify(self, alignment_resolved='qc', alignment_qc=0.952319, alignment_count=4,
                alignment_stored='2020-06-19T10:52:36_noam.roth', trajectory_created=True)

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe_id)
        one.alyx.rest('sessions', 'delete', id=cls.eid)


class TestAlignmentQcManual(unittest.TestCase):
    probe_id = None
    prev_traj_id = None

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
        insertion['name'] = ''.join(random.choices(string.ascii_letters, k=5))
        insertion['json'] = {'xyz_picks': cls.xyz_picks}

        date = str(datetime.date(2018, np.random.randint(1, 12), np.random.randint(1, 28)))
        _, eid = RegistrationClient(one).create_new_session('ZM_1150', date=date)
        cls.eid = str(eid)
        insertion['session'] = cls.eid
        probe_insertion = one.alyx.rest('insertions', 'create', data=insertion)
        cls.probe_id = probe_insertion['id']
        cls.trajectory = data['trajectory'].tolist()
        cls.trajectory.update({'probe_insertion': cls.probe_id})
        cls.trajectory.update({'json': cls.alignments})
        cls.traj = one.alyx.rest('trajectories', 'create', data=cls.trajectory)

    def test_alignments(self):
        checks = getmembers(self, lambda x: ismethod(x) and re.match(r'^_\d{2}_.*', x.__name__))
        # Run numbered functions in order
        for _, fn in sorted(checks, key=lambda x: x[0]):
            self._get_prev_traj_id()
            fn()

    def _get_prev_traj_id(self):
        traj = one.alyx.get('/trajectories?'
                            f'&probe_id={self.probe_id}'
                            '&provenance=Ephys aligned histology track', clobber=True)
        if traj:
            self.prev_traj_id = traj[0]['id']

    def _01_normal_computation(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=self.traj['json'],
                           xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns,
                           depths=SITES_COORDINATES[:, 1],
                           chn_coords=SITES_COORDINATES)
        align_qc.run(update=True, upload_alyx=True, upload_flatiron=False)
        _verify(self,
                alignment_resolved=False,
                alignment_stored='2020-09-28T15:57:25_mayo',
                alignment_count=3,
                trajectory_created=False,
                alignment_qc=0.604081)

    def _02_manual_resolution_latest(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=self.traj['json'],
                           xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns,
                           depths=SITES_COORDINATES[:, 1],
                           chn_coords=SITES_COORDINATES)
        align_qc.resolve_manual('2020-09-28T15:57:25_mayo', update=True, upload_alyx=True,
                                upload_flatiron=False)
        _verify(self,
                alignment_resolved='experimenter',
                alignment_stored='2020-09-28T15:57:25_mayo',
                alignment_count=3,
                trajectory_created=False,
                alignment_qc=0.604081)

    def _03_manual_resolution_not_latest(self):
        align_qc = AlignmentQC(self.probe_id, one=one, brain_atlas=brain_atlas, channels=False)
        align_qc.load_data(prev_alignments=self.traj['json'],
                           xyz_picks=np.array(self.xyz_picks) / 1e6,
                           cluster_chns=self.cluster_chns,
                           depths=SITES_COORDINATES[:, 1],
                           chn_coords=SITES_COORDINATES)
        align_qc.resolve_manual('2020-09-28T10:03:06_alejandro', update=True, upload_alyx=True,
                                upload_flatiron=False, force=True)
        _verify(self,
                alignment_resolved='experimenter',
                alignment_stored='2020-09-28T10:03:06_alejandro',
                alignment_count=3,
                trajectory_created=True,
                alignment_qc=0.604081)

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe_id)
        one.alyx.rest('sessions', 'delete', id=cls.eid)


def _verify(tc, alignment_resolved=None, alignment_count=None,
            alignment_stored=None, trajectory_created=False, alignment_qc=None):
    """
    For a given test case with a `probe_id` attribute, check that Alyx returns insertion records
    that match the provided parameters.
    :param tc: An instance of TestAlignmentQcManual or TestAlignmentQcExisting
    :param alignment_resolved: Check the alignment_resolved is true or false
    :param alignment_count: Check the alignment count matches the one given
    :param alignment_stored: Check the alignment stored key matches the one given
    :param trajectory_created: Check whether a new trajectory exists on Alyx
    :param alignment_qc: Check the stored QC value is close to the provided one
    :return:
    """
    QC_THRESH = 0.8  # Expected alignment QC threshold
    insertion = one.alyx.get(f'/insertions/{tc.probe_id}', clobber=True)
    tc.assertEqual('NOT_SET', insertion['json']['qc'])
    if alignment_count is not None:
        tc.assertEqual(alignment_count, insertion['json']['extended_qc']['alignment_count'])
    if alignment_stored is not None:
        tc.assertEqual(alignment_stored,
                       insertion['json']['extended_qc']['alignment_stored'])
    if alignment_resolved:
        tc.assertEqual(alignment_resolved,
                       insertion['json']['extended_qc']['alignment_resolved_by'])
        tc.assertEqual(1, insertion['json']['extended_qc']['alignment_resolved'])
    elif alignment_resolved is False:
        tc.assertEqual(0, insertion['json']['extended_qc']['alignment_resolved'])
    if alignment_qc:
        tc.assertEqual(insertion['json']['extended_qc']['alignment_qc'] < QC_THRESH,
                       alignment_qc < QC_THRESH)
        tc.assertTrue(np.isclose(insertion['json']['extended_qc']['alignment_qc'], alignment_qc))
    if tc.prev_traj_id:
        traj = one.alyx.get('/trajectories?'
                            f'&probe_id={tc.probe_id}'
                            '&provenance=Ephys aligned histology track', clobber=True)
        tc.assertNotEqual(tc.prev_traj_id == traj[0]['id'], trajectory_created)


class TestUploadToFlatIron(unittest.TestCase):
    probe_id = None

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
                           cluster_chns=cls.cluster_chns,
                           depths=SITES_COORDINATES[:, 1],
                           chn_coords=SITES_COORDINATES)
        cls.file_paths = align_qc.resolve_manual('2020-09-28T15:57:25_mayo', update=True,
                                                 upload_alyx=True, upload_flatiron=True)
        print(cls.file_paths)

    def test_data_content(self):
        alf_path = one.path_from_eid(EPHYS_SESSION).joinpath('alf', self.probe_name)
        channels_mlapdv = np.load(alf_path.joinpath('channels.mlapdv.npy'))
        self.assertTrue(np.all(np.abs(channels_mlapdv) > 0))
        channels_id = np.load(alf_path.joinpath('channels.brainLocationIds_ccf_2017.npy'))
        self.assertEqual(channels_mlapdv.shape[0], channels_id.shape[0])

    def test_upload_to_flatiron(self):
        for file in self.file_paths:
            file_registered = one.alyx.get(f'/datasets?&session={EPHYS_SESSION}'
                                           f'&dataset_type={file.stem}')
            data_id = file_registered[0]['url'][-36:]
            self.assertEqual(len(file_registered), 1)
            one.alyx.rest('datasets', 'delete', id=data_id)

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('insertions', 'delete', id=cls.probe_id)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

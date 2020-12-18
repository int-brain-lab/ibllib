import json
from pathlib import Path
import unittest
import uuid
from datetime import timedelta

import numpy as np

from oneibl.one import ONE
from brainbox.core import intersect2d, ismember2d, ismember
from brainbox.io.parquet import uuid2np, np2uuid, rec2col, np2str
import brainbox.io.video as video


class TestParquet(unittest.TestCase):

    def test_uuids_conversions(self):
        str_uuid = 'a3df91c8-52a6-4afa-957b-3479a7d0897c'
        one_np_uuid = np.array([-411333541468446813, 8973933150224022421])
        two_np_uuid = np.tile(one_np_uuid, [2, 1])
        # array gives a list
        self.assertTrue(all(map(lambda x: x == str_uuid, np2str(two_np_uuid))))
        # single uuid gives a string
        self.assertTrue(np2str(one_np_uuid) == str_uuid)

    def test_rec2col(self):
        json_fixture = Path(__file__).parent.joinpath('fixtures', 'parquet_records.json')
        with open(json_fixture, 'r') as fid:
            datasets = json.loads(fid.read())
        # test with includes / joins and uuid fields in both join and includes
        include = ['id', 'hash', 'dataset_type', 'name', 'file_size', 'collection']
        uuid_fields = ['id', 'eid']
        join = {'subject': 'Bernard', 'lab': 'thelab',
                'eid': '150f92bc-e755-4f54-96c1-84e1eaf832b4'}
        arr = rec2col(datasets, include=include, uuid_fields=uuid_fields, join=join)
        self.assertTrue(np.all(np.array([arr[k].size for k in arr]) == 5))
        self.assertTrue(len(arr.keys()) == len(include) + len(uuid_fields) + len(join.keys()))
        # test single dictionary
        arr_single = rec2col(datasets[0], include=include, uuid_fields=uuid_fields, join=join)
        self.assertTrue(np.all(arr.to_df().iloc[0] == arr_single.to_df()))
        # test empty
        arr_empty = rec2col([], include=include, uuid_fields=uuid_fields, join=join)
        self.assertTrue(arr_empty.to_df().size == 0)

        # the empty float fields should be serialized as NaNs when coerced into double
        [ds.update({'float_field': None}) for ds in datasets]
        arr = rec2col(datasets, uuid_fields=uuid_fields, join=join,
                      types={'float_field': np.double})
        self.assertTrue(np.all(np.isnan(arr['float_field'])))

    def test_uuids_intersections(self):
        ntotal = 500
        nsub = 17
        nadd = 3

        eids = uuid2np([uuid.uuid4() for _ in range(ntotal)])

        np.random.seed(42)
        isel = np.floor(np.argsort(np.random.random(nsub)) / nsub * ntotal).astype(np.int16)
        sids = np.r_[eids[isel, :], uuid2np([uuid.uuid4() for _ in range(nadd)])]
        np.random.shuffle(sids)

        # check the intersection
        v, i0, i1 = intersect2d(eids, sids)
        assert np.all(eids[i0, :] == sids[i1, :])
        assert np.all(np.sort(isel) == np.sort(i0))

        v_, i0_, i1_ = np.intersect1d(eids[:, 0], sids[:, 0], return_indices=True)
        assert np.setxor1d(v_, v[:, 0]).size == 0
        assert np.setxor1d(i0, i0_).size == 0
        assert np.setxor1d(i1, i1_).size == 0

        for a, b in zip(ismember2d(sids, eids), ismember(sids[:, 0], eids[:, 0])):
            assert np.all(a == b)

        # check conversion to numpy back and forth
        uuids = [uuid.uuid4() for _ in np.arange(4)]
        np_uuids = uuid2np(uuids)
        assert np2uuid(np_uuids) == uuids


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.one = ONE(
            base_url="https://test.alyx.internationalbrainlab.org",
            username="test_user",
            password="TapetesBloc18"
        )

    def setUp(self) -> None:
        self.eid = '8dd0fcb0-1151-4c97-ae35-2e2421695ad7'
        self.url = ('http://ibl.flatironinstitute.org/mainenlab/'
                    'Subjects/ZM_1743/2019-06-14/001/raw_video_data/'
                    '_iblrig_leftCamera.raw.71cfeef2-2aa5-46b5-b88f-ca07e3d92474.mp4')

    def test_label_from_path(self):
        # Test file path
        session_path = self.one.path_from_eid(self.eid)
        video_path = session_path / 'raw_video_data' / '_iblrig_bodyCamera.raw.mp4'
        label = video.label_from_path(video_path)
        self.assertEqual('body', label)
        # Test URL
        label = video.label_from_path(self.url)
        self.assertEqual('left', label)
        # Test file name
        label = video.label_from_path('_iblrig_rightCamera.raw.mp4')
        self.assertEqual('right', label)
        # Test wrong file
        label = video.label_from_path('_iblrig_taskSettings.raw.json')
        self.assertIsNone(label)

    def test_url_from_eid(self):
        actual = video.url_from_eid(self.eid, 'left', self.one)
        self.assertEqual(self.url, actual)
        actual = video.url_from_eid(self.eid, one=self.one)
        expected = {'left': self.url}
        self.assertEqual(expected, actual)
        actual = video.url_from_eid(self.eid, label=('left', 'right'), one=self.one)
        expected = {'left': self.url, 'right': None}
        self.assertEqual(expected, actual)

    def test_get_video_meta(self):
        actual = video.get_video_meta(self.url, self.one)
        expected = {
            'length': 144120,
            'fps': 30,
            'width': 1280,
            'height': 1024,
            'duration': timedelta(seconds=4804),
            'size': 495090155
        }
        self.assertEqual(expected, actual)

    def test_get_video_frame(self):
        actual = video.get_video_frame(self.url, 0)
        expected = np.array([152, 44, 206, 97, 0], dtype=np.uint8)
        self.assertTrue(np.all(actual[0, :5, 0] == expected))

    def test_get_video_frames_preload(self):
        actual = video.get_video_frames_preload(self.url, [0, 3, 4])
        expected = np.array([152, 44, 206, 97, 0], dtype=np.uint8)
        self.assertEqual((3, 1024, 1280, 3), actual.shape)
        self.assertTrue(np.all(actual[0, 0, :5, 0] == expected))
        actual = video.get_video_frames_preload(self.url, [0, 3, 4],
                                                mask=np.s_[0, :5, 0], as_list=True)
        self.assertIsInstance(actual, list)
        self.assertEqual(3, len(actual))
        self.assertEqual((5,), actual[0].shape)

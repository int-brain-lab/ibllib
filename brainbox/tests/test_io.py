import json
from pathlib import Path
import unittest
import uuid

import numpy as np

from brainbox.numerical import ismember, ismember2d, intersect2d
from brainbox.io.parquet import uuid2np, np2uuid, rec2col, np2str


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

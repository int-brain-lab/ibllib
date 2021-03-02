import unittest
import numpy as np
import copy

from oneibl.dataclass import SessionDataInfo


class TestSearch(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        dc1 = SessionDataInfo(
            dataset_type='first dtype',
            dataset_id='first uuid data',
            local_path='/first/path',
            eid='first uuid session',
            url='first url',
            data=np.array([1, 2, 3]),
        )
        dc2 = SessionDataInfo(
            dataset_type='second dtype',
            dataset_id='second uuid data',
            local_path='/second/path',
            eid='second uuid session',
            url='second url',
            data=np.array([1, 2, 3]),
        )
        self.dc1 = dc1
        self.dc2 = dc2
        self.dce = SessionDataInfo()

    def test_append(self):
        dc1 = copy.copy(self.dc1)
        dc2 = copy.copy(self.dc2)
        dcall = copy.copy(dc1)
        # append real dictionaries
        dcall.append(dc2)
        self.assertEqual(dcall.dataset_type, [dc1.dataset_type, dc2.dataset_type])
        self.assertEqual(dcall.data, [dc1.data, dc2.data])

    def test_append_empty(self):
        # append with an empty value should reflect the length
        dcall = copy.copy(self.dc1)
        dcall.append(self.dce)
        self.assertEqual(dcall.dataset_type, [self.dc1.dataset_type, None])
        self.assertEqual(dcall.data, [self.dc1.data, None])

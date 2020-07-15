import unittest
import tempfile
from pathlib import Path

import numpy as np

from brainbox import core


class TestIsmember(unittest.TestCase):

    def test_ismember(self):
        def _check_ismember(a, b, lia_, locb_):
            lia, locb = core.ismember(a, b)
            self.assertTrue(np.all(a[lia] == b[locb]))
            self.assertTrue(np.all(lia_ == lia))
            self.assertTrue(np.all(locb_ == locb))

        b = np.array([0, 1, 3, 4, 4])
        a = np.array([1, 4, 5, 4])
        lia_ = np.array([True, True, False, True])
        locb_ = np.array([1, 3, 3])
        _check_ismember(a, b, lia_, locb_)

        b = np.array([0, 4, 3, 1, 4])
        a = np.array([1, 4, 5, 4])
        lia_ = np.array([True, True, False, True])
        locb_ = np.array([3, 1, 1])
        _check_ismember(a, b, lia_, locb_)

        b = np.array([0, 1, 3, 4])
        a = np.array([1, 4, 5])
        lia_ = np.array([True, True, False])
        locb_ = np.array([1, 3])
        _check_ismember(a, b, lia_, locb_)


class TestBunch(unittest.TestCase):

    def test_sync(self):
        """
        This test is just to document current use in libraries in case of refactoring
        """
        sd = core.Bunch({'label': 'toto', 'ap': None, 'lf': 8})
        self.assertTrue(sd['label'] is sd.label)
        self.assertTrue(sd['ap'] is sd.ap)
        self.assertTrue(sd['lf'] is sd.lf)
        sda = core.Bunch({'label': np.array('toto'), 'ap': np.array(None), 'lf': np.array(8)})
        dfa = sda.to_df()
        self.assertTrue(sda is dfa)
        sdb = core.Bunch({'label': np.array(['toto', 'tata']),
                          'ap': np.array([None, 1]),
                          'lf': np.array([10, 8])})
        dfb = sdb.to_df()
        for k in sdb:
            self.assertTrue(np.all(sdb[k] == dfb[k].values))

    def test_bunch_io(self):
        a = np.random.rand(50, 1)
        b = np.random.rand(50, 1)
        abunch = core.Bunch({'a': a, 'b': b})

        with tempfile.TemporaryDirectory() as td:
            npz_file = Path(td).joinpath('test_bunch.npz')
            abunch.save(npz_file)
            another_bunch = core.Bunch.load(npz_file)
            [self.assertTrue(np.all(abunch[k]) == np.all(another_bunch[k])) for k in abunch]
            npz_filec = Path(td).joinpath('test_bunch_comp.npz')
            abunch.save(npz_filec, compress=True)
            another_bunch = core.Bunch.load(npz_filec)
            [self.assertTrue(np.all(abunch[k]) == np.all(another_bunch[k])) for k in abunch]


if __name__ == "__main__":
    unittest.main(exit=False)

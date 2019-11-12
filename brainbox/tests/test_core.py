import unittest
import numpy as np
from brainbox import core


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


if __name__ == "__main__":
    unittest.main(exit=False)

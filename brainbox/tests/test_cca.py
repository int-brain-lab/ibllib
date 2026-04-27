import unittest

import numpy as np
import matplotlib.pylab as plt

from brainbox.population import cca


class TestBunch(unittest.TestCase):
    def test_plotting(self):
        """
        This test is just to document current use in libraries in case of refactoring
        """
        corrs = np.array([0.6, 0.2, 0.1, 0.001])
        errs = np.array([0.1, 0.05, 0.04, 0.0005])
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        cca.plot_correlations(corrs, errs, ax=ax1, color='blue')
        cca.plot_correlations(corrs * 0.1, errs, ax=ax1, color='orange')

    # Shuffle data
    # ...
    # fig, ax1 = plt.subplots(1,1,figsize(10,10))
    # plot_correlations(corrs, ... , ax=ax1, color='blue')
    # plot_correlations(shuffled_coors, ..., ax=ax1, color='red')
    # plt.show()


if __name__ == '__main__':
    unittest.main(exit=False)

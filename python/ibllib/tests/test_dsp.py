import unittest
import numpy as np

import ibllib.dsp.fourier as ft


class TestFFT(unittest.TestCase):

    def test_freduce(self):
        fs = np.fft.fftfreq(5)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))

        fs = np.fft.fftfreq(6)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))


if __name__ == "__main__":
    unittest.main(exit=False)

import unittest
import numpy as np

import ibllib.dsp.fourier as ft


class TestFFT(unittest.TestCase):

    def test_freduce(self):
        fs = np.fft.fftfreq(5)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))

        fs = np.fft.fftfreq(6)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))

    def test_fexpand(self):
        # test odd input
        res = np.random.rand(11)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 11)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test even input
        res = np.random.rand(12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with a 2 dimensional input
        res = np.random.rand(2, 12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with a 3 dimensional input
        res = np.random.rand(3, 5, 12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))

    def test_fscale(self):
        # test for an even number of samples
        res = [0, 100, 200, 300, 400, 500, -400, -300, -200, -100],
        self.assertTrue(np.all(np.abs(ft.fscale(10, 0.001) - res) < 1e-6))
        # test for an odd number of samples
        res = [0, 90.9090909090909, 181.818181818182, 272.727272727273, 363.636363636364,
               454.545454545455, -454.545454545455, -363.636363636364, -272.727272727273,
               -181.818181818182, -90.9090909090909],
        self.assertTrue(np.all(np.abs(ft.fscale(11, 0.001) - res) < 1e-6))

    def test_filter_lp(self):
        ts = np.random.rand(500) * 0.1
        ft.lp(ts, 1, [.1, .2])


if __name__ == "__main__":
    unittest.main(exit=False)

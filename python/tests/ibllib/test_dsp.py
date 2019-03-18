import unittest
import numpy as np
import numpy.matlib as mat

import ibllib.dsp.fourier as ft


class TestFFT(unittest.TestCase):

    def test_freduce(self):
        # test with 1D arrays
        fs = np.fft.fftfreq(5)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))
        fs = np.fft.fftfreq(6)
        self.assertTrue(np.all(ft.freduce(fs) == fs[:-2]))

        # test 2D arrays along both dimensions
        fs = mat.repmat(ft.fscale(500, 0.001), 4, 1)
        self.assertTrue(ft.freduce(fs).shape == (4, 251))
        self.assertTrue(ft.freduce(np.transpose(fs), axis=0).shape == (251, 4))

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
        # test with a 2 dimensional input along last dimension
        res = np.random.rand(2, 12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with a 3 dimensional input along last dimension
        res = np.random.rand(3, 5, 12)
        X = ft.freduce(np.fft.fft(res))
        R = np.real(np.fft.ifft(ft.fexpand(X, 12)))
        self.assertTrue(np.all((res - R) < 1e-6))
        # test with 2 dimensional input along first dimension
        fs = np.transpose(mat.repmat(ft.fscale(500, 0.001, half_sided=True), 4, 1))
        self.assertTrue(ft.fexpand(fs, 500, axis=0).shape == (500, 4))

    def test_fscale(self):
        # test for an even number of samples
        res = [0, 100, 200, 300, 400, 500, -400, -300, -200, -100],
        self.assertTrue(np.all(np.abs(ft.fscale(10, 0.001) - res) < 1e-6))
        # test for an odd number of samples
        res = [0, 90.9090909090909, 181.818181818182, 272.727272727273, 363.636363636364,
               454.545454545455, -454.545454545455, -363.636363636364, -272.727272727273,
               -181.818181818182, -90.9090909090909],
        self.assertTrue(np.all(np.abs(ft.fscale(11, 0.001) - res) < 1e-6))

    def test_filter_lp_hp(self):
        # test 1D time serie: subtracting lp filter removes DC
        ts1 = np.random.rand(500)
        out1 = ft.lp(ts1, 1, [.1, .2])
        self.assertTrue(np.mean(ts1 - out1) < 0.001)
        # test 2D case along the last dimension
        ts = mat.repmat(ts1, 11, 1)
        out = ft.lp(ts, 1, [.1, .2])
        self.assertTrue(np.allclose(out, out1))
        # test 2D case along the first dimension
        ts = mat.repmat(ts1[:, np.newaxis], 1, 11)
        out = ft.lp(ts, 1, [.1, .2], axis=0)
        self.assertTrue(np.allclose(np.transpose(out), out1))
        # test 1D time serie: subtracting lp filter removes DC
        out2 = ft.hp(ts1, 1, [.1, .2])
        self.assertTrue(np.allclose(out1, ts1 - out2))


if __name__ == "__main__":
    unittest.main(exit=False)

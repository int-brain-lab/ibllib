import unittest
import numpy as np
import numpy.matlib as mat

import ibllib.dsp.fourier as ft
from ibllib.dsp import WindowGenerator, rms, rises, falls, fronts


class TestFFT(unittest.TestCase):

    def test_imports(self):
        import ibllib.dsp as dsp
        self.assertTrue(len([dsp.lp,
                             dsp.fexpand,
                             dsp.hp,
                             dsp.fscale,
                             dsp.freduce,
                             dsp.rms]) == 6)

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
        fs = np.transpose(mat.repmat(ft.fscale(500, 0.001, one_sided=True), 4, 1))
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


class TestWindowGenerator(unittest.TestCase):

    def test_window_simple(self):
        wg = WindowGenerator(ns=500, nswin=100, overlap=50)
        sl = list(wg.firstlast)
        self.assertTrue(wg.nwin == len(sl) == 9)
        self.assertTrue(np.all(np.array([s[0] for s in sl]) == np.arange(0, wg.nwin) * 50))
        self.assertTrue(np.all(np.array([s[1] for s in sl]) == np.arange(0, wg.nwin) * 50 + 100))

        wg = WindowGenerator(ns=500, nswin=100, overlap=10)
        sl = list(wg.firstlast)
        first = np.array([0, 90, 180, 270, 360, 450])
        last = np.array([100, 190, 280, 370, 460, 500])
        self.assertTrue(wg.nwin == len(sl) == 6)
        self.assertTrue(np.all(np.array([s[0] for s in sl]) == first))
        self.assertTrue(np.all(np.array([s[1] for s in sl]) == last))

    def test_nwindows_computation(self):
        for m in np.arange(0, 100):
            wg = WindowGenerator(ns=500 + m, nswin=87 + m, overlap=11 + m)
            sl = list(wg.firstlast)
            self.assertTrue(wg.nwin == len(sl))

    def test_firstlast_slices(self):
        # test also the indexing versus direct slicing
        my_sig = np.random.rand(500,)
        wg = WindowGenerator(ns=500, nswin=100, overlap=50)
        # 1) get the window by
        my_rms = np.zeros((wg.nwin,))
        for first, last in wg.firstlast:
            my_rms[wg.iw] = rms(my_sig[first:last])
        # test with slice_array method
        my_rms_ = np.zeros((wg.nwin,))
        for wsig in wg.slice_array(my_sig):
            my_rms_[wg.iw] = rms(wsig)
        self.assertTrue(np.all(my_rms_ == my_rms))
        # test with the slice output
        my_rms_ = np.zeros((wg.nwin,))
        for sl in wg.slice:
            my_rms_[wg.iw] = rms(my_sig[sl])
        self.assertTrue(np.all(my_rms_ == my_rms))

    def test_tscale(self):
        wg = WindowGenerator(ns=500, nswin=100, overlap=50)
        ts = wg.tscale(fs=1000)
        self.assertTrue(ts[0] == (100 - 1) / 2 / 1000)
        self.assertTrue((np.allclose(np.diff(ts), 0.05)))

    def test_rises_falls(self):
        # test 1D case with a long pulse and a dirac
        a = np.zeros(500,)
        a[80:120] = 1
        a[200] = 1
        # rising fronts
        self.assertTrue(all(rises(a) == np.array([80, 200])))
        # falling fronts
        self.assertTrue(all(falls(a) == np.array([120, 201])))
        # both
        ind, val = fronts(a)
        self.assertTrue(all(ind == np.array([80, 120, 200, 201])))
        self.assertTrue(all(val == np.array([1, -1, 1, -1])))

        # test a 2D case with 2 long pulses and a dirac
        a = np.zeros((2, 500))
        a[0, 80:120] = 1
        a[0, 200] = 1
        a[1, 280:320] = 1
        a[1, 400] = 1
        # rising fronts
        self.assertTrue(np.all(rises(a) == np.array([[0, 0, 1, 1], [80, 200, 280, 400]])))
        # falling fronts
        self.assertTrue(np.all(falls(a) == np.array([[0, 0, 1, 1], [120, 201, 320, 401]])))
        # both
        ind, val = fronts(a)
        self.assertTrue(all(ind[0] == np.array([0, 0, 0, 0, 1, 1, 1, 1])))
        self.assertTrue(all(ind[1] == np.array([80, 120, 200, 201, 280, 320, 400, 401])))
        self.assertTrue(all(val == np.array([1, -1, 1, -1, 1, -1, 1, -1])))


if __name__ == "__main__":
    unittest.main(exit=False)

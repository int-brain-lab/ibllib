import unittest
import numpy as np
import scipy.signal

import ibllib.dsp.fourier as ft
from ibllib.dsp import WindowGenerator, rms, rises, falls, fronts, smooth, shift, fit_phase,\
    fcn_cosine
from ibllib.dsp.utils import parabolic_max, sync_timestamps


class TestSyncTimestamps(unittest.TestCase):

    def test_timestamps_lin(self):
        np.random.seed(4132)
        n = 50
        drift = 17.14
        offset = 34.323
        tsa = np.cumsum(np.random.random(n) * 10)
        tsb = tsa * (1 + drift / 1e6) + offset

        # test linear drift
        _fcn, _drift = sync_timestamps(tsa, tsb)
        assert np.all(np.isclose(_fcn(tsa), tsb))
        assert np.isclose(drift, _drift)

        # test missing indices on a
        imiss = np.setxor1d(np.arange(n), [1, 2, 34, 35])
        _fcn, _drift, _ia, _ib = sync_timestamps(tsa[imiss], tsb, return_indices=True)
        assert np.all(np.isclose(_fcn(tsa[imiss[_ia]]), tsb[_ib]))

        # test missing indices on b
        _fcn, _drift, _ia, _ib = sync_timestamps(tsa, tsb[imiss], return_indices=True)
        assert np.all(np.isclose(_fcn(tsa[_ia]), tsb[imiss[_ib]]))

        # test missing indices on both
        imiss2 = np.setxor1d(np.arange(n), [14, 17])
        _fcn, _drift, _ia, _ib = sync_timestamps(tsa[imiss], tsb[imiss2], return_indices=True)
        assert np.all(np.isclose(_fcn(tsa[imiss[_ia]]), tsb[imiss2[_ib]]))


class TestParabolicMax(unittest.TestCase):
    # expected values
    maxi = np.array([np.NaN, 0, 3.04166667, 3.04166667, 5, 5])
    ipeak = np.array([np.NaN, 0, 5.166667, 2.166667, 0, 7])
    # input
    x = np.array([
        [0, 0, 0, 0, 0, np.NaN, 0, 0],  # some NaNs
        [0, 0, 0, 0, 0, 0, 0, 0],  # all flat
        [0, 0, 0, 0, 1, 3, 2, 0],
        [0, 1, 3, 2, 0, 0, 0, 0],
        [5, 1, 3, 2, 0, 0, 0, 0],  # test first sample
        [0, 1, 3, 2, 0, 0, 0, 5],  # test last sample
    ])

    def test_error_cases(self):
        pass

    def test_2d(self):
        ipeak_, maxi_ = parabolic_max(self.x)
        self.assertTrue(np.all(np.isclose(self.maxi, maxi_, equal_nan=True)))
        self.assertTrue(np.all(np.isclose(self.ipeak, ipeak_, equal_nan=True)))

    def test_1d(self):
        # look over the 2D array as 1D chunks
        for i, x in enumerate(self.x):
            ipeak_, maxi_ = parabolic_max(x)
            self.assertTrue(np.all(np.isclose(self.ipeak[i], ipeak_, equal_nan=True)))
            self.assertTrue(np.all(np.isclose(self.maxi[i], maxi_, equal_nan=True)))


class TestDspMisc(unittest.TestCase):

    def test_dsp_cosine_func(self):
        x = np.linspace(0, 40)
        fcn = fcn_cosine(bounds=[20, 30])
        y = fcn(x)
        self.assertTrue(y[0] == 0 and y[-1] == 1 and np.all(np.diff(y) >= 0))


class TestPhaseRegression(unittest.TestCase):

    def test_fit_phase1d(self):
        w = np.zeros(500)
        w[1] = 1
        self.assertTrue(np.isclose(fit_phase(w, .002), .002))

    def test_fit_phase2d(self):
        w = np.zeros((500, 2))
        w[1, 0], w[2, 1] = (1, 1)
        self.assertTrue(np.all(np.isclose(fit_phase(w, .002, axis=0), np.array([.002, .004]))))
        self.assertTrue(np.all(np.isclose(fit_phase(w.transpose(), .002), np.array([.002, .004]))))


class TestShift(unittest.TestCase):

    def test_shift_1d(self):
        ns = 500
        w = scipy.signal.ricker(ns, 10)
        self.assertTrue(np.all(np.isclose(shift(w, 1), np.roll(w, 1))))

    def test_shift_2d(self):
        ns = 500
        w = scipy.signal.ricker(ns, 10)
        w = np.tile(w, (100, 1)).transpose()
        self.assertTrue(np.all(np.isclose(shift(w, 1, axis=0), np.roll(w, 1, axis=0))))
        self.assertTrue(np.all(np.isclose(shift(w, 1, axis=1), np.roll(w, 1, axis=1))))


class TestSmooth(unittest.TestCase):

    def test_smooth_lp(self):
        np.random.seed(458)
        a = np.random.rand(500,)
        a_ = smooth.lp(a, [0.1, 0.15])
        res = ft.hp(np.pad(a_, 100, mode='edge'), 1, [0.1, 0.15])[100:-100]
        self.assertTrue((rms(a) / rms(res)) > 500)


class TestFFT(unittest.TestCase):

    def test_spectral_convolution(self):
        sig = np.random.randn(20, 500)
        w = np.hanning(25)
        c = ft.convolve(sig, w)
        s = np.convolve(sig[0, :], w)
        self.assertTrue(np.all(np.isclose(s, c[0, :-1])))

        c = ft.convolve(sig, w, mode='same')
        s = np.convolve(sig[0, :], w, mode='same')
        self.assertTrue(np.all(np.isclose(c[0, :], s)))

        c = ft.convolve(sig, w[:-1], mode='same')
        s = np.convolve(sig[0, :], w[:-1], mode='same')
        self.assertTrue(np.all(np.isclose(c[0, :], s)))

    def test_nech_optim(self):
        self.assertTrue(ft.ns_optim_fft(2048) == 2048)
        self.assertTrue(ft.ns_optim_fft(65532) == 65536)

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
        fs = np.tile(ft.fscale(500, 0.001), (4, 1))
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
        fs = np.transpose(np.tile(ft.fscale(500, 0.001, one_sided=True), (4, 1)))
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
        ts = np.tile(ts1, (11, 1))
        out = ft.lp(ts, 1, [.1, .2])
        self.assertTrue(np.allclose(out, out1))
        # test 2D case along the first dimension
        ts = np.tile(ts1[:, np.newaxis], (1, 11))
        out = ft.lp(ts, 1, [.1, .2], axis=0)
        self.assertTrue(np.allclose(np.transpose(out), out1))
        # test 1D time serie: subtracting lp filter removes DC
        out2 = ft.hp(ts1, 1, [.1, .2])
        self.assertTrue(np.allclose(out1, ts1 - out2))

    def test_dft(self):
        # test 1D complex
        x = np.array([1, 2 - 1j, -1j, -1 + 2j])
        X = ft.dft(x)
        assert np.all(np.isclose(X, np.fft.fft(x)))
        # test 1D real
        x = np.random.randn(7)
        X = ft.dft(x)
        assert np.all(np.isclose(X, np.fft.rfft(x)))
        # test along the 3 dimensions of a 3D array
        x = np.random.rand(10, 11, 12)
        for axis in np.arange(3):
            X_ = np.fft.rfft(x, axis=axis)
            assert np.all(np.isclose(X_, ft.dft(x, axis=axis)))
        # test 2D irregular grid
        _n0, _n1, nt = (10, 11, 30)
        x = np.random.rand(_n0 * _n1, nt)
        X_ = np.fft.fft(np.fft.fft(x.reshape(_n0, _n1, nt), axis=0), axis=1)
        r, c = [v.flatten() for v in np.meshgrid(np.arange(
            _n0) / _n0, np.arange(_n1) / _n1, indexing='ij')]
        nk, nl = (_n0, _n1)
        X = ft.dft2(x, r, c, nk, nl)
        assert np.all(np.isclose(X, X_))


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

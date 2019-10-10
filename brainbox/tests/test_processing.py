from brainbox import core, processing
import unittest
import numpy as np


class TestProcessing(unittest.TestCase):

    def test_sync(self):
        # Test casting non-uniformly-sampled data to a evenly-sampled TimeSeries.
        # Begin by defining sampling intervals of random half-normally distributed length
        times = np.cumsum(np.abs(np.random.normal(loc=4., scale=6., size=100)))
        # take sample values as though the value was increasing as a cube of sample time
        samples = times**3
        # Use cubic interpolation to resample to uniform interval
        cubes = core.TimeSeries(times=times, values=samples, columns=('cubic',))
        resamp = processing.sync(0.1, timeseries=cubes, interp='cubic', fillval='extrapolate')
        # Check that the sync function is returning a new time series object
        self.assertTrue(isinstance(resamp, core.TimeSeries))
        # Test that all returned sample times are uniformly spaced
        # We need to use np.isclose because of floating point arithematic problems instead of ==0.1
        # Since the actual diff returns 0.09999999999999964
        self.assertTrue(np.all(np.isclose(np.diff(resamp.times), 0.1)))
        # Check that we're within a margin of error on the interpolation
        err_margin = 1e-3  # Maximum percent error allowed
        err_percs = np.abs(resamp.times**3 - resamp.values.T) / (resamp.times**3)
        self.assertTrue(np.all(err_percs < err_margin))
        # Make a second timeseries of square-law increasing samples
        times2 = np.cumsum(np.abs(np.random.normal(loc=2., scale=1., size=200)))
        samples2 = times2**2
        squares = core.TimeSeries(times=times2, values=samples2, columns=('square',))
        # Use cubic interpolation again, this time on both timeseries
        resamp2 = processing.sync(0.1, timeseries=[squares, cubes], interp='cubic',
                                  fillval='extrapolate')
        # Check that the new TS has both squares and cubes as keys and attribs
        self.assertTrue(hasattr(resamp2, 'cubic'))
        self.assertTrue(hasattr(resamp2, 'square'))
        # Check that both timeseries are fully contained in the resampled TS
        self.assertTrue(cubes.times.min() >= resamp2.times.min())
        self.assertTrue(cubes.times.max() <= resamp2.times.max())
        self.assertTrue(squares.times.min() >= resamp2.times.min())
        self.assertTrue(squares.times.max() <= resamp2.times.max())
        # Check that all interpolated values are within the margin of error against the known func
        sq_errperc = np.abs(resamp2.times**2 - resamp2.square) / resamp2.times**2
        cu_errperc = np.abs(resamp2.times**3 - resamp2.cubic) / resamp2.times**3
        self.assertTrue(np.all(sq_errperc < err_margin) & np.all(cu_errperc < err_margin))

        # Now check the numpy array behavior of sync.
        # Try running sync on the cubic times and values only.
        resamp = processing.sync(0.1, times=times, values=samples, interp='cubic',
                                 fillval='extrapolate')
        # Do all the tests we did for the instance created using TimeSeries objects
        self.assertTrue(isinstance(resamp, core.TimeSeries))
        self.assertTrue(np.all(np.isclose(np.diff(resamp.times), 0.1)))
        err_margin = 1e-3  # Maximum percent error allowed
        err_percs = np.abs(resamp.times**3 - resamp.values.T) / (resamp.times**3)
        self.assertTrue(np.all(err_percs < err_margin))
        # Try the multiple-arrays case in which we pass two times and two values
        resamp2 = processing.sync(0.1, times=(times, times2), values=(samples, samples2),
                                  interp='cubic', fillval='extrapolate')
        self.assertTrue(times.min() >= resamp2.times.min())
        self.assertTrue(times.max() <= resamp2.times.max())
        self.assertTrue(times2.min() >= resamp2.times.min())
        self.assertTrue(times2.max() <= resamp2.times.max())

    def test_bincount_2d(self):
        # first test simple with indices
        x = np.array([0, 1, 1, 2, 2, 3, 3, 3])
        y = np.array([3, 2, 2, 1, 1, 0, 0, 0])
        r, xscale, yscale = processing.bincount2D(x, y, xbin=1, ybin=1)
        r_ = np.zeros_like(r)
        # sometimes life would have been simpler in c:
        for ix, iy in zip(x, y):
            r_[iy, ix] += 1
        self.assertTrue(np.all(np.equal(r_, r)))
        # test with negative values
        y = np.array([3, 2, 2, 1, 1, 0, 0, 0]) - 5
        r, xscale, yscale = processing.bincount2D(x, y, xbin=1, ybin=1)
        self.assertTrue(np.all(np.equal(r_, r)))
        # test unequal bins
        r, xscale, yscale = processing.bincount2D(x / 2, y / 2, xbin=1, ybin=2)
        r_ = np.zeros_like(r)
        for ix, iy in zip(np.floor(x / 2), np.floor((y / 2 + 2.5) / 2)):
            r_[int(iy), int(ix)] += 1
        self.assertTrue(np.all(r_ == r))
        # test with weights
        w = np.ones_like(x) * 2
        r, xscale, yscale = processing.bincount2D(x / 2, y / 2, xbin=1, ybin=2, weights=w)
        self.assertTrue(np.all(r_ * 2 == r))
        # test aggregation instead of binning
        x = np.array([0, 1, 1, 2, 2, 4, 4, 4])
        y = np.array([4, 2, 2, 1, 1, 0, 0, 0])
        r, xscale, yscale = processing.bincount2D(x, y)
        self.assertTrue(np.all(xscale == yscale) and np.all(xscale == np.array([0, 1, 2, 4])))
        # test aggregation on a fixed scale
        r, xscale, yscale = processing.bincount2D(x + 10, y + 10, xbin=np.arange(5) + 10,
                                                  ybin=np.arange(3) + 10)
        self.assertTrue(np.all(xscale == np.arange(5) + 10))
        self.assertTrue(np.all(yscale == np.arange(3) + 10))
        self.assertTrue(np.all(r.shape == (3, 5)))


def test_get_unit_bunches():
    pass


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main(exit=False)

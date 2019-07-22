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
        resamp = processing.sync(cubes, 0.1, interp='linear', fillval='extrapolate')
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
        resamp2 = processing.sync([squares, cubes], 0.1, interp='cubic', fillval='extrapolate')
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


if __name__ == "__main__":
    unittest.main(exit=False)

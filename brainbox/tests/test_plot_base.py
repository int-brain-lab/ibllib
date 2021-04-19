import unittest

import numpy as np

from brainbox.plot_base import (DefaultPlot, ImagePlot, ScatterPlot, ProbePlot, LinePlot,
                                scatter_xyc_plot, arrange_channels2banks)


class TestPlotBase(unittest.TestCase):

    def setUp(self):
        """
        Test Basic functionality of the plot class method
        """
        self.x = np.arange(100, dtype=float)
        self.y = np.arange(0, 1000, 10, dtype=float) + 500
        self.z = np.arange(100, dtype=float) - 500
        self.c = np.arange(100, dtype=float) + 1000
        self.img = np.random.rand(100, 100)

    def test_default(self):

        data = {'x': self.x, 'y': self.y, 'z': self.z, 'c': self.c}
        plot_default = DefaultPlot('default', data)

        # Check the default max min lims computations
        plot_default.set_xlim()
        assert(plot_default.xlim == (0, 99))
        plot_default.set_ylim()
        assert(plot_default.ylim == (500, 1490))
        plot_default.set_zlim()
        assert(plot_default.zlim == (-500, -401))
        plot_default.set_clim()
        assert(plot_default.clim == (1000, 1099))

        # Check it deals with nans properly (i.e ignores them)
        plot_default.data['x'][0:5] = np.nan
        plot_default.set_xlim()
        assert(plot_default.xlim == (5, 99))

        # Check that when you specify range that is added
        plot_default.set_xlim((0, 150))
        assert(plot_default.xlim == (0, 150))
        plot_default.set_clim((1050, 1100))
        assert(plot_default.clim == (1050, 1100))

        # Test the _set_default method
        out = plot_default._set_default(val=None, default=2)
        assert(out == 2)
        out = plot_default._set_default(val=5, default=2)
        assert(out == 5)

        # Test instantiation of titles and labels etc
        assert(not plot_default.labels.title)
        assert(not plot_default.labels.xlabel)
        assert(not plot_default.labels.ylabel)

        plot_default.set_labels(title='mytitle', xlabel='myxlabel')
        assert(plot_default.labels.title == 'mytitle')
        assert(plot_default.labels.xlabel == 'myxlabel')
        assert(not plot_default.labels.ylabel)

        # Test adding vertical with default options
        assert(len(plot_default.vlines) == 0)
        plot_default.add_lines(10, 'v')
        assert(len(plot_default.vlines) == 1)
        assert(plot_default.vlines[0].lim == plot_default.ylim)
        assert(plot_default.vlines[0].style == '--')
        assert(plot_default.vlines[0].width == 3)
        assert(plot_default.vlines[0].color == 'k')

        # Test adding horizontal line with specified options
        plot_default.add_lines(10, 'h', lim=(40, 80), style='-', width=8, color='g')
        assert(len(plot_default.hlines) == 1)
        assert(plot_default.hlines[0].lim == (40, 80))
        assert(plot_default.hlines[0].style == '-')
        assert(plot_default.hlines[0].width == 8)
        assert(plot_default.hlines[0].color == 'g')

        # Test conversion to dict
        plot_dict = plot_default.convert2dict()
        keys_to_expect = ['data', 'plot_type', 'hlines', 'vlines', 'labels', 'xlim', 'ylim',
                          'zlim', 'clim']
        assert(all(key in plot_dict for key in keys_to_expect))

    def test_image(self):
        # Instantiation without specifying x and y
        plot_image = ImagePlot(self.img)
        assert(plot_image.plot_type == 'image')
        assert(np.all(plot_image.data.c == self.img))
        assert(np.all(plot_image.data.x == np.arange(self.img.shape[0])))
        assert(np.all(plot_image.data.y == np.arange(self.img.shape[1])))

        # Test instantiation specifying x and y
        plot_image = ImagePlot(self.img, x=self.x, y=self.y)
        assert(plot_image.xlim == (0, 99))
        assert(plot_image.ylim == (500, 1490))
        assert(plot_image.clim == (np.min(self.img), np.max(self.img)))
        plot_image.set_scale()
        assert(plot_image.scale == (0.99, 9.9))
        plot_image.set_offset()
        assert (plot_image.offset == (0, 500))

        # Test instantiating with incorrect dimensions gives assert error
        with self.assertRaises(AssertionError):
            ImagePlot(self.img, x=self.x, y=self.y[:-10])

    def test_probe(self):

        # Specific to pyqtgraph implementation
        x = [np.array((0, 1)), np.array((1, 2)), np.array((2, 3))]
        y = [self.y, self.x, self.z]
        img = [self.c[np.newaxis, :], self.c[np.newaxis, :], self.x[np.newaxis, :]]

        plot_probe = ProbePlot(img, x, y)
        # Make sure the automatic limits are set properly to the min max within the whole list
        assert(plot_probe.xlim == (0, 3))
        assert(plot_probe.ylim == (-500, 1490))
        assert(plot_probe.clim == (0, 1099))

        # Make sure we can set chosen values too
        plot_probe.set_xlim((-100, 100))
        assert(plot_probe.xlim == (-100, 100))

        # Make sure scales are set correctly for each item in list
        plot_probe.set_scale()
        assert(plot_probe.scale[0] == (1, 9.9))
        assert(plot_probe.scale[1] == (1, 0.99))
        assert (plot_probe.scale[2] == (1, 0.99))

        plot_probe.set_offset()
        assert(plot_probe.offset[0] == (0, 500))
        assert(plot_probe.offset[1] == (1, 0))
        assert (plot_probe.offset[2] == (2, -500))

    def test_scatter(self):
        plot_scatter = ScatterPlot(self.x, self.y, c=self.c)
        assert(plot_scatter.plot_type == 'scatter')
        # Test that defaults are set correctly
        assert(plot_scatter.color == 'b')
        assert(not plot_scatter.marker_size)
        assert(plot_scatter.marker_type == 'o')
        assert(plot_scatter.opacity == 1)
        assert(not plot_scatter.line_color)
        assert(not plot_scatter.line_width)
        assert(plot_scatter.line_style == '-')

        # Test instantiation with wrong dimension gives error
        with self.assertRaises(AssertionError):
            ScatterPlot(x=self.x, y=self.y[:-10])

    def test_line(self):
        plot_line = LinePlot(self.x, self.y)
        assert(plot_line.plot_type == 'line')
        # Test that defaults are set correctly
        assert(plot_line.line_color == 'k')
        assert(plot_line.line_width == 2)
        assert(plot_line.line_style == '-')
        assert(not plot_line.marker_size)
        assert(not plot_line.marker_type)

        with self.assertRaises(AssertionError):
            LinePlot(x=self.x, y=self.y[:-10])


class TestScatterXYC(unittest.TestCase):
    def setUp(self):
        """
        Test Basic functionality of the plot class method
        """
        self.x = np.arange(100, dtype=float)
        self.y = np.arange(100, dtype=float)
        self.c = np.arange(100, dtype=float)

    def test_without_RGB_conversion(self):
        plot_scatter = scatter_xyc_plot(self.x, self.y, self.c, cmap='binary', rgb=False)
        # color remains at default
        assert(plot_scatter.color == 'b')

    def test_RGB_conversion(self):
        plot_scatter = scatter_xyc_plot(self.x, self.y, self.c, cmap='binary', rgb=True)
        assert(len(plot_scatter.color) == len(plot_scatter.data.x))
        assert(np.all(plot_scatter.color[0] == (1, 1, 1, 1)))
        assert(np.all(plot_scatter.color[-1] == (0, 0, 0, 1)))

    def test_RGB_conversion_with_clim(self):
        plot_scatter = scatter_xyc_plot(self.x, self.y, self.c, cmap='binary', clim=(0, 50),
                                        rgb=True)
        assert(len(plot_scatter.color) == len(plot_scatter.data.x))
        assert(np.all(plot_scatter.color[0] == (1, 1, 1, 1)))
        assert(np.all(plot_scatter.color[50] == (0, 0, 0, 1)))
        assert (np.all(plot_scatter.color[-1] == (0, 0, 0, 1)))


class TestArrangeChannels2Bank(unittest.TestCase):

    def setUp(self):
        """
        Test arrange_channels2bank function
        """
        self.unique_y = np.arange(0, 200, 20)
        y = np.r_[self.unique_y, self.unique_y, self.unique_y]
        x = np.r_[np.ones((len(self.unique_y))) * 5, np.ones((len(self.unique_y))) * 15,
                  np.ones((len(self.unique_y))) * 25]
        self.chn_coords = np.c_[x, y]
        self.data = np.random.rand((len(y)))
        self.depth = np.linspace(0, 3000, len(y))

    def test_no_pad(self):
        # For pyqtgraph implementation
        data_bnk, x_bnk, y_bnk = arrange_channels2banks(self.data, self.chn_coords, pad=False,
                                                        x_offset=10)
        # Test the data has been distributed as expected
        assert(len(data_bnk) == 3)
        assert(data_bnk[0].shape == (1, 10))
        assert(np.all(data_bnk[0] == self.data[0:10]))
        # Test that the x values are as expected
        assert(len(x_bnk) == 3)
        assert(x_bnk[0].shape == (2,))
        assert(np.all(x_bnk[0] == (0, 10)))
        # Test thet the y values are as expected
        assert(len(y_bnk) == 3)
        assert(y_bnk[0].shape == (10,))
        assert(np.all(y_bnk[0] == self.unique_y))

    def test_pad(self):
        data_bnk, x_bnk, y_bnk = arrange_channels2banks(self.data, self.chn_coords, pad=True,
                                                        x_offset=10)
        assert(len(data_bnk) == 3)
        # Check dimensions are correct and data has been assigned correctly
        assert(data_bnk[2].shape == (3, 12))
        assert(np.all(data_bnk[2][1, 1:-1] == self.data[-10:]))
        # Check data has been padded in all directions
        assert(np.all(np.isnan(data_bnk[2][0, :])))
        assert(np.all(np.isnan(data_bnk[2][2, :])))
        assert(np.all(np.isnan(data_bnk[2][:, 0])))
        assert(np.all(np.isnan(data_bnk[2][:, -1])))

        # Test that the x values are as expected
        assert(len(x_bnk) == 3)
        assert(x_bnk[2].shape == (3,))
        assert(np.all(x_bnk[2] == (20, 30, 40)))

        # Test that the x values are as expected
        assert(len(y_bnk) == 3)
        assert(y_bnk[2].shape == (12,))
        assert(np.all(y_bnk[2] == np.r_[np.min(self.unique_y) - np.diff(self.unique_y)[0],
                                        self.unique_y, np.max(self.unique_y) +
                                        np.diff(self.unique_y)[-1]]))

    def test_with_depth(self):
        data_bnk, x_bnk, y_bnk = arrange_channels2banks(self.data, self.chn_coords,
                                                        depth=self.depth, pad=True, x_offset=10)
        assert(np.all(y_bnk[0][1:-1] == self.depth[:10]))
        assert(np.all(y_bnk[2][1:-1] == self.depth[-10:]))

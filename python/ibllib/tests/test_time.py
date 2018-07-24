import unittest
import ibllib.time
import datetime


class TestUtils(unittest.TestCase):

    def test_format_date_range(self):
        date_range = ['2018-03-01', '2018-03-24']
        date_range_out = ['2018-03-01', '2018-03-25']
        # test the string input
        self.assertTrue(ibllib.time.format_date_range(date_range) == date_range_out)
        # test the date input
        date_range = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in date_range]
        self.assertTrue(ibllib.time.format_date_range(date_range) == date_range_out)

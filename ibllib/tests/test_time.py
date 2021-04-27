import unittest
import ibllib.time
import datetime
import pandas as pd


class TestUtils(unittest.TestCase):

    def test_format_date_range(self):
        date_range = ['2018-03-01', '2018-03-24']
        date_range_out = ['2018-03-01', '2018-03-24']
        # test the string input
        self.assertTrue(ibllib.time.format_date_range(date_range) == date_range_out)
        # test the date input
        date_range = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in date_range]
        self.assertTrue(ibllib.time.format_date_range(date_range) == date_range_out)
        # test input validation
        date_range[-1] = date_range_out[-1]  # noqa [datetime, str]
        with self.assertRaises(ValueError):
            ibllib.time.format_date_range(date_range)

    def test_isostr2date(self):
        # test the full string
        a = ibllib.time.isostr2date('2018-03-01T12:34:56.99999')
        self.assertTrue(a == datetime.datetime(2018, 3, 1, 12, 34, 56, 999990))
        # test UTC offset
        # a = ibllib.time.isostr2date('2018-03-01T12:34:56+02:00')  # FAILS!
        # if ms is rounded, test without the F field
        b = ibllib.time.isostr2date('2018-03-01T12:34:56')
        self.assertTrue(b == datetime.datetime(2018, 3, 1, 12, 34, 56))
        # test a mixed list input
        c = ['2018-03-01T12:34:56.99999', '2018-03-01T12:34:56']
        d = ibllib.time.isostr2date(c)
        self.assertTrue((d[0] == a) and (d[1] == b))
        # test with pandas series
        e = ibllib.time.isostr2date(pd.Series(c))
        self.assertTrue((e[0] == a) and (e[1] == b))

    def test_date2isostr(self):
        expected = '2018-08-14T00:00:00'
        day = datetime.date(2018, 8, 14)
        self.assertEqual(expected, ibllib.time.date2isostr(day))
        dt = datetime.datetime(2018, 8, 14)
        self.assertEqual(expected, ibllib.time.date2isostr(dt))


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

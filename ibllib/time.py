# library of small functions
import datetime
import numpy as np


def isostr2date(isostr):
    """
    Convert strings representing dates into datetime.datetime objects aimed ad Django REST API
    ISO 8601: '2018-05-22T14:35:22.99585' or '2018-05-22T14:35:22'

    :param isostr: a string, list of strings or panda Series / numpy arrays containing strings
    :return: a scalar, list of
    """
    # NB this is intended for scalars or small list. See the ciso8601 pypi module instead for
    # a performance implementation
    if not isinstance(isostr, str):
        return [isostr2date(el) for el in isostr]

    format = '%Y-%m-%dT%H:%M:%S'
    if '.' in isostr:
        format += '.%f'
    if '+' in isostr:
        format += '.%f'
    return datetime.datetime.strptime(isostr, format)


def date2isostr(adate):
    # NB this is intended for scalars or small list. See the ciso8601 pypi module instead for
    # a performance implementation
    if type(adate) is datetime.date:
        adate = datetime.datetime.fromordinal(adate.toordinal())
    return datetime.datetime.isoformat(adate)


def format_date_range(date_range):
    if all([isinstance(d, str) for d in date_range]):
        date_range = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in date_range]
    elif not all([isinstance(d, datetime.date) for d in date_range]):
        raise ValueError('Date range doesn''t have proper format: list of 2 strings "yyyy-mm-dd" ')
    # the django filter is implemented in datetime and assumes the beginning of the day (last day
    # is excluded by default
    date_range = [d.strftime('%Y-%m-%d') for d in date_range]
    return date_range


def convert_pgts(time):
    """Convert PointGray cameras timestamps to seconds.
    Use convert then uncycle"""
    # offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds


def uncycle_pgts(time):
    """Unwrap the converted seconds of a PointGray camera timestamp series."""
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128

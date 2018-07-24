# library of small functions
import datetime


def isostr2date(isostr):
    return datetime.datetime.strptime(isostr, '%Y-%m-%dT%H:%M:%S.%f')


def date2isostr(adate):
    return datetime.datetime.isoformat(adate)


def format_date_range(date_range):
    if all([isinstance(d, str) for d in date_range]):
        date_range = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in date_range]
    elif not all([isinstance(d, datetime.date) for d in date_range]):
        raise ValueError('Date range doesn''t have proper format: list of 2 strings "yyyy-mm-dd" ')
    # the django filter is implemented in datetime and assumes the beginning of the day (last day
    # is excluded by default
    date_range[1] += datetime.timedelta(days=1)
    date_range = [d.strftime('%Y-%m-%d') for d in date_range]
    return date_range
# library of small functions
import datetime


def isostr2date(isostr):
    '''
    Convert strings representing dates into datetime.datetime objects
    ISO 8601: '2018-05-22T14:35:22.99585' or '2018-05-22T14:35:22'

    :param isostr: a string, list of strings or panda Series / numpy arrays containing strings
    :return: a scalar, list of
    '''
    if not isinstance(isostr, str):
        return [isostr2date(el) for el in isostr]
    if '.' not in isostr:
        return datetime.datetime.strptime(isostr, '%Y-%m-%dT%H:%M:%S')
    else:
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
    date_range = [d.strftime('%Y-%m-%d') for d in date_range]
    return date_range

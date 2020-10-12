"""
A module for processing experiment references in a human readable way

Three pieces of information are required to uniquely identify an experimental session: subject
nickname, the date, and the sequence number (whether the session was the first, second,
etc. on that date).

Alyx and ONE use uuids (a.k.a. eids) to uniquely identify sessions, however these are not
readable.  This module converts between these uuids an readable references.

References may be strings in the form yyyy-mm-dd_n_subject, which may be easily sorted,
or as bunches (dicts) of the form {'subject': str, 'date': datetime.date, 'sequence', int}.
"""
import re
import functools
from pathlib import Path
from typing import Union, List, Iterable as Iter
from datetime import datetime
from collections.abc import Iterable, Mapping

from oneibl.one import ONE
from brainbox.core import Bunch

__all__ = [
    'ref2eid', 'ref2dict', 'ref2path', 'eid2path', 'eid2ref', 'path2ref', 'ref2dj', 'is_exp_ref'
]


def parse_values(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parse = kwargs.pop('parse', True)
        ref = func(*args, **kwargs)
        if parse:
            if isinstance(ref['date'], str):
                if len(ref['date']) == 10:
                    ref['date'] = datetime.strptime(ref['date'], '%Y-%m-%d').date()
                else:
                    ref['date'] = datetime.fromisoformat(ref['date']).date()
            ref['sequence'] = int(ref['sequence'])
        return ref
    return wrapper_decorator


def recurse(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        first, *args = args
        if isinstance(first, Iterable) and not isinstance(first, (str, Mapping)):
            return [func(item, *args, **kwargs) for item in first]
        else:
            return func(first, *args, **kwargs)
    return wrapper_decorator


@recurse
def ref2eid(ref: Union[Mapping, str, Iter], one=None) -> Union[str, List]:
    """
    Returns experiment uuid, given one or more experiment references
    :param ref: One or more objects with keys ('subject', 'date', 'sequence'), or strings with the
    form yyyy-mm-dd_n_subject
    :param one: An instance of ONE
    :return: an experiment uuid string

    Examples:
    >>> base = 'https://test.alyx.internationalbrainlab.org'
    >>> one = ONE(username='test_user', password='TapetesBloc18', base_url=base)
    Connected to...
    >>> ref = {'date': datetime(2018, 7, 13).date(), 'sequence': 1, 'subject': 'flowers'}
    >>> ref2eid(ref, one=one)
    '4e0b3320-47b7-416e-b842-c34dc9004cf8'
    >>> ref2eid(['2018-07-13_1_flowers', '2019-04-11_1_KS005'], one=one)
    ['4e0b3320-47b7-416e-b842-c34dc9004cf8',
     '7dc3c44b-225f-4083-be3d-07b8562885f4']
    """
    if not one:
        one = ONE()
    ref = ref2dict(ref, parse=False)  # Ensure dict
    session = one.search(
        subjects=ref['subject'],
        date_range=(str(ref['date']), str(ref['date'])),
        number=ref['sequence'])
    assert len(session) == 1, 'session not found'
    return session[0]


@recurse
@parse_values
def ref2dict(ref: Union[str, Mapping, Iter]) -> Union[Bunch, List]:
    """
    Returns a Bunch (dict-like) from a reference string (or list thereof)
    :param ref: One or more objects with keys ('subject', 'date', 'sequence')
    :return: A Bunch in with keys ('subject', 'sequence', 'date')

    Examples:
    >>> ref2dict('2018-07-13_1_flowers')
    {'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'}
    >>> ref2dict('2018-07-13_001_flowers', parse=False)
    {'date': '2018-07-13', 'sequence': '001', 'subject': 'flowers'}
    >>> ref2dict(['2018-07-13_1_flowers', '2020-01-23_002_ibl_witten_01'])
    [{'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'},
     {'date': datetime.date(2020, 1, 23), 'sequence': 2, 'subject': 'ibl_witten_01'}]
    """
    if isinstance(ref, (Bunch, dict)):
        return Bunch(ref)  # Short circuit
    ref = dict(zip(['date', 'sequence', 'subject'], ref.split('_', 2)))
    return Bunch(ref)


@recurse
def ref2path(ref: Union[str, Mapping, Iter], one=None, offline: bool = False) -> Union[Path, List]:
    """
    Convert one or more experiment references to session path(s)
    :param ref: One or more objects with keys ('subject', 'date', 'sequence'), or strings with the
    form yyyy-mm-dd_n_subject
    :param one: An instance of ONE
    :param offline: Return path without connecting to database (unimplemented)
    :return: a Path object for the experiment session

    Examples:
    >>> base = 'https://test.alyx.internationalbrainlab.org'
    >>> one = ONE(username='test_user', password='TapetesBloc18', base_url=base)
    Connected to...
    >>> ref = {'subject': 'flowers', 'date': datetime(2018, 7, 13).date(), 'sequence': 1}
    >>> ref2path(ref, one=one)
    WindowsPath('E:/FlatIron/zadorlab/Subjects/flowers/2018-07-13/001')
    >>> ref2path(['2018-07-13_1_flowers', '2019-04-11_1_KS005'], one=one)
    [WindowsPath('E:/FlatIron/zadorlab/Subjects/flowers/2018-07-13/001'),
     WindowsPath('E:/FlatIron/cortexlab/Subjects/KS005/2019-04-11/001')]
    """
    if not one:
        one = ONE()
    if offline:
        raise NotImplementedError  # Requires lab name :(
        # root = Path(one._get_cache_dir(None))
        # path = root / ref.subject / str(ref.date) / ('%03d' % ref.sequence)
    else:
        ref = ref2dict(ref, parse=False)
        eid, (d,) = one.search(
            subjects=ref['subject'],
            date_range=(str(ref['date']), str(ref['date'])),
            number=ref['sequence'],
            details=True)
        path = d.get('local_path')
        if not path:
            root = Path(one._get_cache_dir(None)) / 'Subjects' / d['lab']
            return root / d['subject'] / d['start_time'][:10] / ('%03d' % d['number'])
        else:
            return Path(path)


@recurse
def eid2path(eid: Union[str, Iter], one=None, offline: bool = False) -> Union[Path, List]:
    """
    Returns a local path from an eid, regardless of whether the path exists locally
    :param eid: An experiment uuid
    :param one: An instance of ONE
    :param offline: If True, do not connect to database (not implemented)
    :return: a Path instance

    Examples:
    >>> base = 'https://test.alyx.internationalbrainlab.org'
    >>> one = ONE(username='test_user', password='TapetesBloc18', base_url=base)
    Connected to...
    >>> eid = '4e0b3320-47b7-416e-b842-c34dc9004cf8'
    >>> eid2path(eid, one=one)
    WindowsPath('E:/FlatIron/zadorlab/Subjects/flowers/2018-07-13/001')
    >>> eid2path([eid, '7dc3c44b-225f-4083-be3d-07b8562885f4'], one=one)
    [WindowsPath('E:/FlatIron/zadorlab/Subjects/flowers/2018-07-13/001'),
     WindowsPath('E:/FlatIron/cortexlab/Subjects/KS005/2019-04-11/001')]
    """
    if not one:
        one = ONE()
    if offline:
        raise NotImplementedError
        # path = one.path_from_eid(eid, offline=True)
    else:
        d = one.get_details(eid)
        path = d.get('local_path')
        if not path:
            root = Path(one._get_cache_dir(None)) / d['lab'] / 'Subjects'
            path = root / d['subject'] / d['start_time'][:10] / ('%03d' % d['number'])
    return path


@recurse
def eid2ref(eid: Union[str, Iter], one=None, as_dict=True, parse=True) \
        -> Union[str, Mapping, List]:
    """
    Get human-readable session ref from path
    :param eid: The experiment uuid to find reference for
    :param one: An ONE instance
    :param as_dict: If false a string is returned in the form 'subject_sequence_yyyy-mm-dd'
    :param parse: If true, the reference date and sequence are parsed from strings to their
    respective data types
    :return: one or more objects with keys ('subject', 'date', 'sequence'), or strings with the
    form yyyy-mm-dd_n_subject

    Examples:
    >>> base = 'https://test.alyx.internationalbrainlab.org'
    >>> one = ONE(username='test_user', password='TapetesBloc18', base_url=base)
    Connected to...
    >>> eid = '4e0b3320-47b7-416e-b842-c34dc9004cf8'
    >>> eid2ref(eid, one=one)
    {'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1}
    >>> eid2ref(eid, parse=False, one=one)
    {'subject': 'flowers', 'date': '2018-07-13', 'sequence': '001'}
    >>> eid2ref(eid, as_dict=False, one=one)
    '2018-07-13_1_flowers'
    >>> eid2ref(eid, as_dict=False, parse=False, one=one)
    '2018-07-13_001_flowers'
    >>> eid2ref([eid, '7dc3c44b-225f-4083-be3d-07b8562885f4'], one=one)
    [{'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1},
     {'subject': 'KS005', 'date': datetime.date(2019, 4, 11), 'sequence': 1}]
    """
    if not one:
        one = ONE()

    d = one.get_details(eid)
    if parse:
        date = datetime.fromisoformat(d['start_time']).date()
        ref = {'subject': d['subject'], 'date': date, 'sequence': d['number']}
        format_str = '{date:%Y-%m-%d}_{sequence:d}_{subject:s}'
    else:
        date = d['start_time'][:10]
        ref = {'subject': d['subject'], 'date': date, 'sequence': '%03d' % d['number']}
        format_str = '{date:s}_{sequence:s}_{subject:s}'
    return Bunch(ref) if as_dict else format_str.format(**ref)


@recurse
@parse_values
def path2ref(path_str: Union[str, Path, Iter]) -> Union[Bunch, List]:
    """
    Returns a human readable experiment reference, given a session path.  The path need not exist.
    :param path_str: A path to a given session
    :return: one or more objects with keys ('subject', 'date', 'sequence')

    Examples:
    >>> path_str = Path('E:/FlatIron/Subjects/zadorlab/flowers/2018-07-13/001')
    >>> path2ref(path_str)
    {'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1}
    >>> path2ref(path_str, parse=False)
    {'subject': 'flowers', 'date': '2018-07-13', 'sequence': '001'}
    >>> path_str2 = Path('E:/FlatIron/Subjects/churchlandlab/CSHL046/2020-06-20/002')
    >>> path2ref([path_str, path_str2])
    [{'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1},
     {'subject': 'CSHL046', 'date': datetime.date(2020, 6, 20), 'sequence': 2}]
    """
    pattern = r'(?P<subject>[\w-]+)([\\/])(?P<date>\d{4}-\d{2}-\d{2})(\2)(?P<sequence>\d{3})'
    match = re.search(pattern, str(path_str)).groupdict()
    return Bunch(match)


def ref2dj(ref: Union[str, Mapping, Iter]):
    """
    Return an ibl-pipeline sessions table, restricted by experiment reference(s)
    :param ref: one or more objects with keys ('subject', 'date', 'sequence'), or strings with the
    form yyyy-mm-dd_n_subject
    :return: an acquisition.Session table

    Examples:
    >>> ref2dj('2020-06-20_2_CSHL046').fetch1()
    Connecting...
    {'subject_uuid': UUID('dffc24bc-bd97-4c2a-bef3-3e9320dc3dd7'),
     'session_start_time': datetime.datetime(2020, 6, 20, 13, 31, 47),
     'session_number': 2,
     'session_date': datetime.date(2020, 6, 20),
     'subject_nickname': 'CSHL046'}
    >>> len(ref2dj({'date':'2020-06-20', 'sequence':'002', 'subject':'CSHL046'}))
    1
    >>> len(ref2dj(['2020-06-20_2_CSHL046', '2019-11-01_1_ibl_witten_13']))
    2
    """
    from ibl_pipeline import subject, acquisition
    sessions = acquisition.Session.proj('session_number', session_date='date(session_start_time)')
    sessions = sessions * subject.Subject.proj('subject_nickname')

    ref = ref2dict(ref)  # Ensure dict-like

    @recurse
    def restrict(r):
        date, sequence, subject = dict(sorted(r.items())).values()  # Unpack sorted
        restriction = {
            'subject_nickname': subject,
            'session_number': sequence,
            'session_date': date}
        return restriction

    return sessions & restrict(ref)


@recurse
def is_exp_ref(ref: Union[str, Mapping, Iter]) -> Union[bool, List[bool]]:
    """
    Returns True is ref is a valid experiment reference
    :param ref: one or more objects with keys ('subject', 'date', 'sequence'), or strings with the
    form yyyy-mm-dd_n_subject
    :return: True if ref is valid

    Examples:
    >>> ref = {'date': datetime(2018, 7, 13).date(), 'sequence': 1, 'subject': 'flowers'}
    >>> is_exp_ref(ref)
    True
    >>> is_exp_ref('2018-07-13_001_flowers')
    True
    >>> is_exp_ref('invalid_ref')
    False
    """
    if isinstance(ref, (Bunch, dict)):
        if not {'subject', 'date', 'sequence'}.issubset(ref):
            return False
        ref = '{date}_{sequence}_{subject}'.format(**ref)
    elif not isinstance(ref, str):
        return False
    return re.compile(r'\d{4}(-\d{2}){2}_(\d{1}|\d{3})_\w+').match(ref) is not None


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)

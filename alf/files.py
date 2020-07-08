"""
Module for identifying and parsing ALF file names.

An ALF file has the following components (those in brackets are optional):
    (_namespace_)object.attribute(_timescale)(.extra.parts).ext

For more information, see the following documentation:
    https://docs.internationalbrainlab.org/en/latest/04_reference.html#alf

Created on Tue Sep 11 18:06:21 2018

@author: Miles
"""
import re

ALF_EXP = (
    r'(?P<namespace>(?:^_{1})\w+(?:_{1}))?'
    r'(?P<object>\w+)\.'
    r'(?P<attribute>[a-zA-Z]+)'
    r'(?P<timescale>(?:_{1})\w+)*'
    r'(?P<extra>[\.\w]+)*\.'
    r'(?P<extension>\w+$)')


def is_valid(filename):
    """
    Returns a True for a given file name if it is an ALF file, otherwise returns False

    Examples:
        >>> is_valid('trials.feedbackType.npy')
        True
        >>> is_valid('spike_trian.npy')
        False
        >>> is_valid('channels._phy_ids.csv')
        False

    Args:
        filename (str): The name of the file

    Returns:
        bool
    """
    return re.match(ALF_EXP, filename) is not None


def alf_parts(filename, as_dict=False):
    """
    Return the parsed elements of a given ALF filename.

    Examples:
        >>> alf_parts('_namespace_obj.times_timescale.extra.foo.ext')
        ('_namespace_', 'obj', 'times', '_timescale', '.extra.foo', 'ext')
        >>> alf_parts('spikes.clusters.npy', as_dict=True)
        {'namespace': None,
         'object': 'spikes',
         'attribute': 'clusters',
         'timescale': None,
         'extra': None,
         'extension': 'npy'}
        >>> alf_parts('spikes.times_ephysClock.npy')
        (None, 'spikes', 'times', '_ephysClock', None, 'npy')
        >>> alf_parts('_iblmic_audioSpectrogram.frequencies.npy')
        ('_iblmic_', 'audioSpectrogram', 'frequencies', None, None, 'npy')

    Args:
        filename (str): The name of the file
        as_dict (bool): when True a dict of matches is returned

    Returns:
        namespace (str): The _namespace_ or None if not present
        object (str): ALF object
        attribute (str): The ALF attribute
        timescale (str): The ALF _timescale or None if not present
        extra (str): Any extra parts to the filename, or None if not present
        extension (str): The file extension
    """
    m = re.match(ALF_EXP, filename)
    if not m:
        raise ValueError('Invalid ALF filename')
    return m.groupdict() if as_dict else m.groups()


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)

def _compare_version_tag(v1, v2, fcn):
    v1_ = ''.join(['{:03d}'.format(int(v)) for v in v1.split('.')])
    v2_ = ''.join(['{:03d}'.format(int(v)) for v in v2.split('.')])
    return fcn(v1_, v2_)


def gt(v1, v2):
    """
    check if v1 > v2

    :param v1: version string, in the form "1.23.3"
    :type v1: str
    :param v2: version string, in the form "1.23.3"
    :type v2: str
    :return: bool
    """
    return _compare_version_tag(v1, v2, str.__gt__)


def ge(v1, v2):
    """
    check if v1 >= v2

    :param v1: version string, in the form "1.23.3"
    :type v1: str
    :param v2: version string, in the form "1.23.3"
    :type v2: str
    :return: bool
    """
    return _compare_version_tag(v1, v2, str.__ge__)


def lt(v1, v2):
    """
    check if v1 < v2

    :param v1: version string, in the form "1.23.3"
    :type v1: str
    :param v2: version string, in the form "1.23.3"
    :type v2: str
    :return: bool
    """
    return _compare_version_tag(v1, v2, str.__lt__)


def le(v1, v2):
    """
    check if v1 <= v2

    :param v1: version string, in the form "1.23.3"
    :type v1: str
    :param v2: version string, in the form "1.23.3"
    :type v2: str
    :return: bool
    """
    return _compare_version_tag(v1, v2, str.__le__)


def eq(v1, v2):
    """
    check if v1 == v2

    :param v1: version string, in the form "1.23.3"
    :type v1: str
    :param v2: version string, in the form "1.23.3"
    :type v2: str
    :return: bool
    """
    return _compare_version_tag(v1, v2, str.__eq__)

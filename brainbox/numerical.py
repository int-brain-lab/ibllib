import numpy as np
from numba import jit


def ismember(a, b):
    """
    equivalent of np.isin but returns indices as in the matlab ismember function
    returns an array containing logical 1 (true) where the data in A is B
    also returns the location of members in b such as a[lia] == b[locb]
    :param a: 1d - array
    :param b: 1d - array
    :return: isin, locb
    """
    lia = np.isin(a, b)
    aun, _, iuainv = np.unique(a[lia], return_index=True, return_inverse=True)
    _, ibu, iau = np.intersect1d(b, aun, return_indices=True)
    locb = ibu[iuainv]
    return lia, locb


def ismember2d(a, b):
    """
    Equivalent of np.isin but returns indices as in the matlab ismember function
    returns an array containing logical 1 (true) where the data in A is B
    also returns the location of members in b such as a[lia, :] == b[locb, :]
    :param a: 2d array
    :param b: 2d array
    :return: isin, locb
    """
    amask = np.ones(a.shape[0], dtype=np.bool)
    ia = np.zeros(a.shape, dtype=np.bool)
    ib = np.zeros(a.shape, dtype=np.int32) - 1
    ina = np.zeros(a.shape[0], dtype=np.bool)
    bind = np.zeros(a.shape[0], dtype=np.int32) - 1
    # get a 1d ismember first for each column
    for n in np.arange(a.shape[1]):
        iaa, ibb = ismember(a[amask, n], b[:, n])
        ia[amask, n] = iaa
        ib[np.where(amask)[0][iaa], n] = ibb
    # those that have at least one mismatch are not in
    amask[~np.all(ia, axis=1)] = False
    # those that get the same index for all column do not need further testing
    ifound = np.where(amask)[0][np.sum(np.abs(np.diff(ib[amask], axis=1)), axis=1) == 0]
    ina[ifound] = True
    amask[ifound] = False
    bind[ifound] = ib[ifound, 0]
    # the remaining ones have to be check manually (almost never happens for uuids)
    for iaa in np.where(amask)[0]:
        ibb = find_first_2d(b, a[iaa, :])
        if ibb:
            ina[iaa] = True
            bind[iaa] = ibb
    return ina, bind[ina]


def intersect2d(a0, a1, assume_unique=False):
    """
    Performs intersection on multiple columns arrays a0 and a1
    :param a0:
    :param a1:
    :return: intesection
    :return: index of a0 such as intersection = a0[ia, :]
    :return: index of b0 such as intersection = b0[ib, :]
    """
    _, i0, i1 = np.intersect1d(a0[:, 0], a1[:, 0],
                               return_indices=True, assume_unique=assume_unique)
    for n in np.arange(1, a0.shape[1]):
        _, ii0, ii1 = np.intersect1d(a0[i0, n], a1[i1, n],
                                     return_indices=True, assume_unique=assume_unique)
        i0 = i0[ii0]
        i1 = i1[ii1]
    return a0[i0, :], i0, i1


@jit(nopython=True)
def find_first_2d(mat, val):
    """
    Returns first index where
    The purpose of this function is performance: uses low level numba and avoids looping
    through the full array
    :param mat: np.array
    :param val: values to search for
    :return: index or empty array
    """
    for i in np.arange(mat.shape[0]):
        if np.all(mat[i] == val):
            return i

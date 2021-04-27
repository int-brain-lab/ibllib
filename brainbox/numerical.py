from typing import TypeVar, Sequence, Union, Optional, Type

import numpy as np
from numba import jit

D = TypeVar('D', bound=np.generic)
Array = Union[np.ndarray, Sequence]


def between_sorted(sorted_v, bounds=None):
    """
    Given a vector of sorted values, returns a boolean vector indices True when the value
    is between bounds. If multiple bounds are given, returns the equivalent OR of individual
    bounds tuple
    Especially useful for spike times
      indices = between_sorted(spike_times, [tstart, tstop])
    :param sorted_v: vector containing sorted values (won't check)
    :param bounds: minimum included value and maximum included value
        can be a list[tstart, tstop] or an array of dimension (n, 2)
    :return:
    """
    bounds = np.array(bounds)
    starts, stops = (np.take(bounds, 0, axis=-1), np.take(bounds, 1, axis=-1))
    sbounds = np.logical_and(starts <= sorted_v[-1], stops >= sorted_v[0])
    starts = starts[sbounds]
    stops = stops[sbounds]
    sel = sorted_v * 0
    sel[np.searchsorted(sorted_v, starts)] = 1
    istops = np.searchsorted(sorted_v, stops, side='right')
    sel[istops[istops < sorted_v.size]] += -1
    return np.cumsum(sel).astype(bool)


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
    amask = np.ones(a.shape[0], dtype=bool)
    ia = np.zeros(a.shape, dtype=bool)
    ib = np.zeros(a.shape, dtype=np.int32) - 1
    ina = np.zeros(a.shape[0], dtype=bool)
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
    :param assume_unique: If True, the input arrays are both assumed to be unique,
    which can speed up the calculation.
    :return: intersection
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


def within_ranges(x: np.ndarray, ranges: Array, labels: Optional[Array] = None,
                  mode: str = 'vector', dtype: Type[D] = 'int8') -> np.ndarray:
    """
    Detects which points of the input vector lie within one of the ranges specified in the ranges.
    Returns an array the size of x with a 1 if the corresponding point is within a range.

    The function uses a stable sort algorithm (timsort) to find the edges within the input array.
    Edge behaviour is inclusive.

    Ranges are [(start0, stop0), (start1, stop1), etc.] or n-by-2 numpy array.
    The ranges may be optionally assigned a row in 'matrix' mode or a numerical label in 'vector'
    mode. Labels must have a length of n.  Overlapping ranges have a value that is the sum of the
    relevant range labels (ones in 'matrix' mode).

    If mode is 'vector' (default) it will give a vector, specifying the range of each point.
    If mode is 'matrix' it will give a matrix output where each range is assigned a particular row
    index with 1 if the point belongs to that range label.  Multiple ranges can be assigned to a
    particular row, e.g. [0, 0,1] would give a 2-by-N matrix with the first two ranges in the
    first row.  Points within more than one range are given a value > 1

    Parameters
    ----------
    x : array_like
        An array whose points are tested against the ranges.  multi-dimensional arrays are
        flattened to 1D
    ranges : array_like
        A list of tuples or N-by-2 array of ranges to test, where N is the number of ranges,
        i.e. [[start0, stop0],
        [start1, stop1]]
    labels : vector, list
        If mode is 'vector'; a list of integer labels to demarcate which points lie within each
        range.  In 'matrix' mode; a list of column indices (ranges can share indices).
        The number of labels should match the number of ranges.  If None, ones are used for all
        ranges.
    mode : {'matrix', 'vector'}
        The type of output to return.  If 'matrix' (default), an N-by-M matrix is returned where N
        is the size of x and M corresponds to the max index in labels, e.g. with labels=[0,1,2],
        the output matrix would have 3 columns.  If 'vector' a vector the size of x is returned.
    dtype : str, numeric or boolean type
        The data type of the returned array.  If type is bool, the labels in vector mode will be
        ignored.  Default is int8.


    Returns
    -------
    A vector of size like x where zeros indicate that the points do not lie within ranges (
    'vector' mode) or a matrix where out.shape[0] == x.size and out.shape[1] == max(labels) + 1.

    Examples
    -------
    # Assert that points in ranges are mutually exclusive
    np.all(within_ranges(x, ranges) <= 1)

    Tests
    -----
    >>> import numpy as np
    >>> within_ranges(np.arange(11), [(1, 2), (5, 8)])
    array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0], dtype=int8)
    >>> ranges = np.array([[1, 2], [5, 8]])
    >>> within_ranges(np.arange(10) + 1, ranges, labels=np.array([0,1]), mode='matrix')
    array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=int8)
    >>> within_ranges(np.arange(11), [(1,2), (5,8), (4,6)], labels=[0,1,1], mode='matrix')
    array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 2, 2, 1, 1, 0, 0]], dtype=int8)
    >>> within_ranges(np.arange(10) + 1, ranges, np.array([3,1]), mode='vector')
    array([3, 3, 0, 0, 1, 1, 1, 1, 0, 0], dtype=int8)
    >>> within_ranges(np.arange(11), [(1,2), (5,8), (4,6)], dtype=bool)
    array([False,  True,  True, False,  True,  True,  True,  True,  True,
           False, False])
    """
    # Flatten
    x = x.ravel()

    # Ensure ranges are numpy
    ranges = np.array(ranges)

    # Get size info
    n_points = x.size
    n_ranges = ranges.shape[0]

    if labels is None:
        # In 'matrix' mode default row index is 0
        labels = np.zeros((n_ranges,), dtype='uint32')
        if mode == 'vector':  # Otherwise default numerical label is 1
            labels += 1
    assert len(labels) >= n_ranges, 'range labels do not match number of ranges'
    n_labels = np.unique(labels).size

    # If no ranges given, short circuit function and return zeros
    if n_ranges == 0:
        return np.zeros_like(x, dtype=dtype)

    # Check end comes after start in each case
    assert np.all(np.diff(ranges, axis=1) > 0), 'ranges ends must all be greater than starts'

    # Make array containing points, starts and finishes

    # This order means it will be inclusive
    to_sort = np.concatenate((ranges[:, 0], x, ranges[:, 1]))
    # worst case O(n*log(n)) but will be better than  this as most of the array is ordered;
    # memory overhead ~n/2
    idx = np.argsort(to_sort, kind='stable')

    # Make delta array containing 1 for every start and -1 for every stop
    # with one row for each range label
    if mode == 'matrix':
        delta_shape = (n_labels, n_points + 2 * n_ranges)
        delta = np.zeros(delta_shape, dtype='int8')

        delta[labels, np.arange(n_ranges)] = 1
        delta[labels, n_points + n_ranges + np.arange(n_ranges)] = -1

        # Arrange in order
        delta_sorted = delta[:, idx]

        # Take cumulative sums
        summed = np.cumsum(delta_sorted, axis=1)

        # Reorder back to original order
        reordered = np.zeros(delta_shape, dtype=dtype)
        reordered[:, idx] = summed.reshape(delta_shape[0], -1)
        return reordered[:, np.arange(n_ranges, n_points + n_ranges)]

    elif mode == 'vector':
        delta_shape = (n_points + 2 * n_ranges,)
        r_delta = np.zeros(delta_shape, dtype='int32')
        r_delta[np.arange(n_ranges)] = labels
        r_delta[n_points + n_ranges + np.arange(n_ranges)] = -labels

        # Arrange in order
        r_delta_sorted = r_delta[idx]

        # Take cumulative sum
        r_summed = np.cumsum(r_delta_sorted)

        # Reorder back to original
        r_reordered = np.zeros_like(r_summed, dtype=dtype)
        r_reordered[idx] = r_summed

        return r_reordered[np.arange(n_ranges, n_points + n_ranges)]
    else:
        raise ValueError('unknown mode type, options are "matrix" and "vector"')

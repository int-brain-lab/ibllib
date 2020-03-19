"""
Elephant - Electrophysiology Analysis Toolkit
Liscence: BSD 3-Clause

Copyright (c) 2014-2019, Elephant authors and contributors
All rights reserved.
"""


class BinnedSpikeTrain(object):
    """
    Class which calculates a binned spike train and provides methods to
    transform the binned spike train to a boolean matrix or a matrix with
    counted time points.
    A binned spike train represents the occurrence of spikes in a certain time
    frame.
    I.e., a time series like [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] is
    represented as [0, 0, 1, 3, 4, 5, 6]. The outcome is dependent on given
    parameter such as size of bins, number of bins, start and stop points.
    A boolean matrix represents the binned spike train in a binary (True/False)
    manner. Its rows represent the number of spike trains and the columns
    represent the binned index position of a spike in a spike train.
    The calculated matrix entry containing `True` indicates a spike.
    A matrix with counted time points is calculated the same way, but its
    entries contain the number of spikes that occurred in the given bin of the
    given spike train.
    Parameters
    ----------
    spiketrains : neo.SpikeTrain or list of neo.SpikeTrain or np.ndarray
        Spike train(s) to be binned.
    binsize : pq.Quantity, optional
        Width of a time bin.
        Default: None
    num_bins : int, optional
        Number of bins of the binned spike train.
        Default: None
    t_start : pq.Quantity, optional
        Time of the left edge of the first bin (left extreme; included).
        Default: None
    t_stop : pq.Quantity, optional
        Time of the right edge of the last bin (right extreme; excluded).
        Default: None
    Raises
    ------
    AttributeError
        If less than 3 optional parameters are `None`.
    TypeError
        If `spiketrains` is an np.ndarray with dimensionality different than
        NxM or
        if type of `num_bins` is not an `int` or `num_bins` < 0.
    ValueError
        When number of bins calculated from `t_start`, `t_stop` and `binsize`
        differs from provided `num_bins` or
        if `t_stop` of any spike train is smaller than any `t_start` or
        if any spike train does not cover the full [`t_start`, t_stop`] range.
    Warns
    -----
    UserWarning
        If some spikes fall outside of [`t_start`, `t_stop`] range
    See also
    --------
    _convert_to_binned
    spike_indices
    to_bool_array
    to_array
    Notes
    -----
    There are four minimal configurations of the optional parameters which have
    to be provided, otherwise a `ValueError` will be raised:
    * `t_start`, `num_bins`, `binsize`
    * `t_start`, `num_bins`, `t_stop`
    * `t_start`, `bin_size`, `t_stop`
    * `t_stop`, `num_bins`, `binsize`
    If `spiketrains` is a `neo.SpikeTrain` or a list thereof, it is enough to
    explicitly provide only one parameter: `num_bins` or `binsize`. The
    `t_start` and `t_stop` will be calculated from given `spiketrains` (max
    `t_start` and min `t_stop` of `neo.SpikeTrain`s).
    Missing parameter will be calculated automatically.
    All parameters will be checked for consistency. A corresponding error will
    be raised, if one of the four parameters does not match the consistency
    requirements.
    """

    def __init__(self, spiketrains, binsize=None, num_bins=None, t_start=None,
                 t_stop=None):
        """
        Defines a BinnedSpikeTrain class
        """
        self.is_spiketrain = _check_neo_spiketrain(spiketrains)
        if not self.is_spiketrain:
            self.is_binned = _check_binned_array(spiketrains)
        else:
            self.is_binned = False
        # Converting spiketrains to a list, if spiketrains is one
        # SpikeTrain object
        if isinstance(spiketrains,
                      neo.SpikeTrain) and self.is_spiketrain:
            spiketrains = [spiketrains]

        # Link to input
        self.lst_input = spiketrains
        # Set given parameter
        self.t_start = t_start
        self.t_stop = t_stop
        self.num_bins = num_bins
        self.binsize = binsize
        # Empty matrix for storage, time points matrix
        self._mat_u = None
        # Variables to store the sparse matrix
        self._sparse_mat_u = None
        # Check all parameter, set also missing values
        if self.is_binned:
            self.num_bins = np.shape(spiketrains)[1]
        self._calc_start_stop(spiketrains)
        self._check_init_params(
            self.binsize, self.num_bins, self.t_start, self.t_stop)
        self._check_consistency(spiketrains, self.binsize, self.num_bins,
                                self.t_start, self.t_stop)
        # Now create sparse matrix
        self._convert_to_binned(spiketrains)

        if self.is_spiketrain:
            n_spikes = sum(map(len, spiketrains))
            n_spikes_binned = self.get_num_of_spikes()
            if n_spikes != n_spikes_binned:
                warnings.warn("Binning discarded {n} last spike(s) in the "
                              "input spiketrain.".format(
                                  n=n_spikes - n_spikes_binned))

    @property
    def matrix_rows(self):
        return self._sparse_mat_u.shape[0]

    @property
    def matrix_columns(self):
        return self._sparse_mat_u.shape[1]

    # =========================================================================
    # There are four cases the given parameters must fulfill, or a `ValueError`
    # will be raised:
    # t_start, num_bins, binsize
    # t_start, num_bins, t_stop
    # t_start, bin_size, t_stop
    # t_stop, num_bins, binsize
    # =========================================================================

    def _check_init_params(self, binsize, num_bins, t_start, t_stop):
        """
        Checks given parameters.
        Calculates also missing parameter.
        Parameters
        ----------
        binsize : pq.Quantity
            Size of bins
        num_bins : int
            Number of bins
        t_start: pq.Quantity
            Start time for the binned spike train
        t_stop: pq.Quantity
            Stop time for the binned spike train
        Raises
        ------
        TypeError
            If type of `num_bins` is not an `int`.
        ValueError
            When `t_stop` is smaller than `t_start`.
        """
        # Check if num_bins is an integer (special case)
        if num_bins is not None:
            if not np.issubdtype(type(num_bins), int):
                raise TypeError("num_bins is not an integer!")
        # Check if all parameters can be calculated, otherwise raise ValueError
        if t_start is None:
            self.t_start = _calc_tstart(num_bins, binsize, t_stop)
        elif t_stop is None:
            self.t_stop = _calc_tstop(num_bins, binsize, t_start)
        elif num_bins is None:
            self.num_bins = _calc_num_bins(binsize, t_start, t_stop)
        elif binsize is None:
            self.binsize = _calc_binsize(num_bins, t_start, t_stop)

    def _calc_start_stop(self, spiketrains):
        """
        Calculates `t_start`, `t_stop` from given spike trains.
        The start and stop points are calculated from given spike trains only
        if they are not calculable from given parameters or the number of
        parameters is less than three.
        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list or np.ndarray of neo.SpikeTrain
        """
        if self._count_params() is False:
            start, stop = _get_start_stop_from_input(spiketrains)
            if self.t_start is None:
                self.t_start = start
            if self.t_stop is None:
                self.t_stop = stop

    def _count_params(self):
        """
        Checks the number of explicitly provided parameters and returns `True`
        if the count is greater or equal `3`.
        The calculation of the binned matrix is only possible if there are at
        least three parameters (fourth parameter will be calculated out of
        them).
        This method checks if the necessary parameters are not `None` and
        returns `True` if the count is greater or equal to `3`.
        Returns
        -------
        bool
            True, if the count of not None parameters is greater or equal to
            `3`, False otherwise.
        """
        return sum(x is not None for x in
                   [self.t_start, self.t_stop, self.binsize,
                    self.num_bins]) >= 3

    def _check_consistency(self, spiketrains, binsize, num_bins, t_start,
                           t_stop):
        """
        Checks the given parameters for consistency
        Raises
        ------
        AttributeError
            If there is an insufficient number of parameters.
        TypeError
            If `num_bins` is not an `int` or is <0.
        ValueError
            If an inconsistency regarding the parameters appears, e.g.
            `t_start` > `t_stop`.
        """
        if self._count_params() is False:
            raise AttributeError("Too few parameters given. Please provide "
                                 "at least one of the parameter which are "
                                 "None.\n"
                                 "t_start: %s, t_stop: %s, binsize: %s, "
                                 "num_bins: %s" % (
                                     self.t_start,
                                     self.t_stop,
                                     self.binsize,
                                     self.num_bins))
        if self.is_spiketrain:
            t_starts = [elem.t_start for elem in spiketrains]
            t_stops = [elem.t_stop for elem in spiketrains]
            max_tstart = max(t_starts)
            min_tstop = min(t_stops)
            if max_tstart >= min_tstop:
                raise ValueError("Starting time of each spike train must be "
                                 "smaller than each stopping time")
            if t_start < max_tstart or t_start > min_tstop:
                raise ValueError(
                    'some spike trains are not defined in the time given '
                    'by t_start')
            if not (t_start < t_stop <= min_tstop):
                raise ValueError(
                    'too many / too large time bins. Some spike trains are '
                    'not defined in the ending time')
        if num_bins != int((
            (t_stop - t_start).rescale(
                binsize.units) / binsize).magnitude):
            raise ValueError(
                "Inconsistent arguments t_start (%s), " % t_start +
                "t_stop (%s), binsize (%d) " % (t_stop, binsize) +
                "and num_bins (%d)" % num_bins)
        if num_bins - int(num_bins) != 0 or num_bins < 0:
            raise TypeError(
                "Number of bins (num_bins) is not an integer or < 0: " + str(
                    num_bins))

    @property
    def bin_edges(self):
        """
        Returns all time edges as a quantity array with :attr:`num_bins` bins.
        The borders of all time steps between :attr:`t_start` and
        :attr:`t_stop` with a step :attr:`binsize`. It is crucial for many
        analyses that all bins have the same size, so if
        :attr:`t_stop` - :attr:`t_start` is not divisible by :attr:`binsize`,
        there will be some leftover time at the end
        (see https://github.com/NeuralEnsemble/elephant/issues/255).
        The length of the returned array should match :attr:`num_bins`.
        Returns
        -------
        bin_edges : pq.Quantity
            All edges in interval [:attr:`t_start`, :attr:`t_stop`] with
            :attr:`num_bins` bins are returned as a quantity array.
        """
        t_start = self.t_start.rescale(self.binsize.units).magnitude
        bin_edges = np.linspace(t_start, t_start + self.num_bins *
                                self.binsize.magnitude,
                                num=self.num_bins + 1, endpoint=True)
        return pq.Quantity(bin_edges, units=self.binsize.units)

    @property
    def bin_centers(self):
        """
        Returns each center time point of all bins between :attr:`t_start` and
        :attr:`t_stop` points.
        The center of each bin of all time steps between start and stop.
        Returns
        -------
        bin_edges : pq.Quantity
            All center edges in interval (:attr:`start`, :attr:`stop`).
        """
        return self.bin_edges[:-1] + self.binsize / 2

    def to_sparse_array(self):
        """
        Getter for sparse matrix with time points.
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix, version with spike counts.
        See also
        --------
        scipy.sparse.csr_matrix
        to_array
        """
        return self._sparse_mat_u

    def to_sparse_bool_array(self):
        """
        Getter for boolean version of the sparse matrix, calculated from
        sparse matrix with counted time points.
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix, binary, boolean version.
        See also
        --------
        scipy.sparse.csr_matrix
        to_bool_array
        """
        # Return sparse Matrix as a copy
        tmp_mat = self._sparse_mat_u.copy()
        tmp_mat[tmp_mat.nonzero()] = 1
        return tmp_mat.astype(bool)

    def get_num_of_spikes(self, axis=None):
        """
        Compute the number of binned spikes.
        Parameters
        ----------
        axis : int, optional
            If `None`, compute the total num. of spikes.
            Otherwise, compute num. of spikes along axis.
            If axis is `1`, compute num. of spikes per spike train (row).
            Default is `None`.
        Returns
        -------
        n_spikes_per_row : int or np.ndarray
            The number of binned spikes.
        """
        if axis is None:
            return self._sparse_mat_u.sum(axis=axis)
        n_spikes_per_row = self._sparse_mat_u.sum(axis=axis)
        n_spikes_per_row = np.asarray(n_spikes_per_row)[:, 0]
        return n_spikes_per_row

    @property
    def spike_indices(self):
        """
        A list of lists for each spike train (i.e., rows of the binned matrix),
        that in turn contains for each spike the index into the binned matrix
        where this spike enters.
        In contrast to `to_sparse_array().nonzero()`, this function will report
        two spikes falling in the same bin as two entries.
        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> import quantities as pq
        >>> st = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
        ...                   t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(st, num_bins=10, binsize=1 * pq.s,
        ...                           t_start=0 * pq.s)
        >>> print(x.spike_indices)
        [[0, 0, 1, 3, 4, 5, 6]]
        >>> print(x.to_sparse_array().nonzero()[1])
        [0 1 3 4 5 6]
        >>> print(x.to_array())
        [[2, 1, 0, 1, 1, 1, 1, 0, 0, 0]]
        """
        spike_idx = []
        for row in self._sparse_mat_u:
            # Extract each non-zeros column index and how often it exists,
            # i.e., how many spikes fall in this column
            n_cols = np.repeat(row.indices, row.data)
            spike_idx.append(n_cols)
        return spike_idx

    @property
    def is_binary(self):
        """
        Checks and returns `True` if given input is a binary input.
        Beware, that the function does not know if the input is binary
        because e.g `to_bool_array()` was used before or if the input is just
        sparse (i.e. only one spike per bin at maximum).
        Returns
        -------
        bool
            True for binary input, False otherwise.
        """

        return is_binary(self.lst_input)

    def to_bool_array(self):
        """
        Returns a matrix, in which the rows correspond to the spike trains and
        the columns correspond to the bins in the `BinnedSpikeTrain`.
        `True` indicates a spike in given bin of given spike train and
        `False` indicates lack of spikes.
        Returns
        -------
        numpy.ndarray
            Returns a dense matrix representation of the sparse matrix,
            with `True` indicating a spike and `False` indicating a no-spike.
            The columns represent the index position of the bins and rows
            represent the number of spike trains.
        See also
        --------
        scipy.sparse.csr_matrix
        scipy.sparse.csr_matrix.toarray
        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> import quantities as pq
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
        ...                  t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(a, num_bins=10, binsize=1 * pq.s,
        ...                           t_start=0 * pq.s)
        >>> print(x.to_bool_array())
        [[ True  True False  True  True  True  True False False False]]
        """
        return self.to_array().astype(bool)

    def to_array(self, store_array=False):
        """
        Returns a dense matrix, calculated from the sparse matrix, with counted
        time points of spikes. The rows correspond to spike trains and the
        columns correspond to bins in a `BinnedSpikeTrain`.
        Entries contain the count of spikes that occurred in the given bin of
        the given spike train.
        If the boolean :attr:`store_array` is set to `True`, the matrix
        will be stored in memory.
        Returns
        -------
        matrix : np.ndarray
            Matrix with spike counts. Columns represent the index positions of
            the binned spikes and rows represent the spike trains.
        Examples
        --------
        >>> import elephant.conversion as conv
        >>> import neo as n
        >>> a = n.SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
        ...                  t_stop=10.0 * pq.s)
        >>> x = conv.BinnedSpikeTrain(a, num_bins=10, binsize=1 * pq.s,
        ...                           t_start=0 * pq.s)
        >>> print(x.to_array())
        [[2 1 0 1 1 1 1 0 0 0]]
        See also
        --------
        scipy.sparse.csr_matrix
        scipy.sparse.csr_matrix.toarray
        """
        if self._mat_u is not None:
            return self._mat_u
        if store_array:
            self._store_array()
            return self._mat_u
        # Matrix on demand
        else:
            return self._sparse_mat_u.toarray()

    def _store_array(self):
        """
        Stores the matrix with counted time points in memory.
        """
        if self._mat_u is None:
            self._mat_u = self._sparse_mat_u.toarray()

    def remove_stored_array(self):
        """
        Unlinks the matrix with counted time points from memory.
        """
        self._mat_u = None

    def binarize(self, copy=True):
        """
        Clip the internal array (no. of spikes in a bin) to `0` (no spikes) or
        `1` (at least one spike) values only.
        Parameters
        ----------
        copy : bool
            Perform the clipping in-place (False) or on a copy (True).
            Default: True.
        Returns
        -------
        bst : BinnedSpikeTrain
            `BinnedSpikeTrain` with both sparse and dense (if present) array
            representation clipped to `0` (no spike) or `1` (at least one
            spike) entries.
        """
        if copy:
            bst = deepcopy(self)
        else:
            bst = self
        bst._sparse_mat_u.data.clip(max=1, out=bst._sparse_mat_u.data)
        if bst._mat_u is not None:
            bst._mat_u.clip(max=1, out=bst._mat_u)
        return bst

    @property
    def sparsity(self):
        """
        Returns
        -------
        float
            Matrix sparsity defined as no. of nonzero elements divided by
            the matrix size
        """
        num_nonzero = self._sparse_mat_u.data.shape[0]
        return num_nonzero / np.prod(self._sparse_mat_u.shape)

    def _convert_to_binned(self, spiketrains):
        """
        Converts `neo.SpikeTrain` objects to a sparse matrix
        (`scipy.sparse.csr_matrix`), which contains the binned spike times, and
        stores it in :attr:`_sparse_mat_u`.
        Parameters
        ----------
        spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
            Spike trains to bin.
        """
        if not self.is_spiketrain:
            self._sparse_mat_u = sps.csr_matrix(spiketrains, dtype=int)
            return

        row_ids, column_ids = [], []
        # data
        counts = []
        for idx, elem in enumerate(spiketrains):
            ev = elem.view(pq.Quantity)
            scale = np.array(((ev - self.t_start).rescale(
                self.binsize.units) / self.binsize).magnitude, dtype=int)
            la = np.logical_and(ev >= self.t_start.rescale(self.binsize.units),
                                ev <= self.t_stop.rescale(self.binsize.units))
            filled_tmp = scale[la]
            filled_tmp = filled_tmp[filled_tmp < self.num_bins]
            f, c = np.unique(filled_tmp, return_counts=True)
            column_ids.extend(f)
            counts.extend(c)
            row_ids.extend([idx] * len(f))
        csr_matrix = sps.csr_matrix((counts, (row_ids, column_ids)),
                                    shape=(len(spiketrains),
                                           self.num_bins),
                                    dtype=int)
        self._sparse_mat_u = csr_matrix


def _check_neo_spiketrain(matrix):
    """
    Checks if given input contains neo.SpikeTrain objects
    Parameters
    ----------
    matrix
        Object to test for `neo.SpikeTrain`s
    Returns
    -------
    bool
        True if `matrix` is a neo.SpikeTrain or a list or tuple thereof,
        otherwise False.
    """
    # Check for single spike train
    if isinstance(matrix, neo.SpikeTrain):
        return True
    # Check for list or tuple
    if isinstance(matrix, (list, tuple)):
        return all(map(_check_neo_spiketrain, matrix))
    return False


def _check_binned_array(matrix):
    """
    Checks if given input is a binned array
    Parameters
    ----------
    matrix
        Object to test
    Returns
    -------
    bool
        True if `matrix` is an 2D array-like object,
        otherwise False.
    Raises
    ------
    TypeError
        If `matrix` is not 2-dimensional.
    """
    matrix = np.asarray(matrix)
    # Check for proper dimension MxN
    if matrix.ndim == 2:
        return True
    elif matrix.dtype == np.dtype('O'):
        raise TypeError('Please check the dimensions of the input, '
                        'it should be an MxN array, '
                        'the input has the shape: {}'.format(matrix.shape))
    else:
        # Otherwise not supported
        raise TypeError('Input not supported. Please check again')
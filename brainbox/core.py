"""
Creates core data types and functions which support all of brainbox.
"""
from pathlib import Path
import numpy as np
import pandas as pd


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())

    def to_df(self):
        """Attempts to returns a pandas.DataFrame if all elements are arrays of the same length
        Returns the original bunch if it can't"""
        try:
            return pd.DataFrame.from_dict(self)
        except ValueError:
            return self

    def save(self, npz_file, compress=False):
        """
        Saves a npz file containing the arrays of the bunch.

        :param npz_file: output file
        :param compress: bool (False) use compression
        :return: None
        """
        if compress:
            np.savez_compressed(npz_file, **self)
        else:
            np.savez(npz_file, **self)

    @staticmethod
    def load(npz_file):
        """
        Loads a npz file containing the arrays of the bunch.

        :param npz_file: output file
        :return: Bunch
        """
        if not Path(npz_file).exists():
            raise FileNotFoundError(f"{npz_file}")
        return Bunch(np.load(npz_file))


class TimeSeries(dict):
    """A subclass of dict with dot syntax, enforcement of time stamping"""

    def __init__(self, times, values, columns=None, *args, **kwargs):
        """TimeSeries objects are explicity for storing time series data in which entry (row) has
        a time stamp associated. TS objects have obligatory 'times' and 'values' entries which
        must be passed at construction, the length of both of which must match. TimeSeries takes an
        optional 'columns' argument, which defaults to None, that is a set of labels for the
        columns in 'values'. These are also exposed via the dot syntax as pointers to the specific
        columns which they reference.

        :param times: an ordered object containing a list of timestamps for the time series data
        :param values: an ordered object containing the associated measurements for each time stamp
        :param columns: a tuple or list of column labels, defaults to none. Each column name will
            be exposed as ts.colname in the TimeSeries object unless colnames are not strings.

        Also can take any additional kwargs beyond times, values, and columns for additional data
        storage like session date, experimenter notes, etc.

        Example:
        timestamps, mousepos = load_my_data()  # in which mouspos is T x 2 array of x,y coordinates
        positions = TimeSeries(times=timestamps, values=mousepos, columns=('x', 'y'),
                               analyst='John Cleese', petshop=True,
                               notes=("Look, matey, I know a dead mouse when I see one, "
                                      'and I'm looking at one right now."))
        """
        super(TimeSeries, self).__init__(times=np.array(times), values=np.array(values),
                                         columns=columns, *args, **kwargs)
        self.__dict__ = self
        self.columns = columns
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)

        # Enforce times dict key which contains a list or array of timestamps
        if len(self.times) != len(values):
            raise ValueError('Time and values must be of the same length')

        # If column labels are passed ensure same number of labels as columns, then expose
        # each column label using the dot syntax of a Bunch
        if isinstance(self.values, np.ndarray) and columns is not None:
            if self.values.shape[1] != len(columns):
                raise ValueError('Number of column labels must equal number of columns in values')
            self.update({col: self.values[:, i] for i, col in enumerate(columns)})

    def copy(self):
        """Return a new TimeSeries instance which is a copy of the current TimeSeries instance."""
        return TimeSeries(super(TimeSeries, self).copy())

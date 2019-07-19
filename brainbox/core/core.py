'''
Core data types and functions which support all of brainbox.
'''
import numpy as np


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())


class TimeSeries(dict):
    """A subclass of dict with dot syntax, enforcement of time stamping"""
    def __init__(self, times, values, columns=None, *args, **kwargs):
        super(TimeSeries, self).__init__(times=np.array(times), values=np.array(values),
                                         columns=columns, *args, **kwargs)
        self.__dict__ = self
        self.columns = columns
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)

        # Enforce times dict key which contains a list or array of timestamps
        if len(self.times) != len(values):
            raise ValueError('Time and values must be of the same length')

        # If column labels are passed ensure same number of labels as columns.
        if isinstance(self.values, np.ndarray) and columns is not None:
            if self.values.shape[1] != len(columns):
                raise ValueError('Number of column labels must equal number of columns in values')

    def copy(self):
        """Return a new TimeSeries instance which is a copy of the current TimeSeries instance."""
        return TimeSeries(super(TimeSeries, self).copy())

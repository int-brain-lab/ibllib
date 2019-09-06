import numpy as np

def gen_x(intervals, signals, clusters=None):
    """
    Generates X matrix for regression
    :param intervals: Event intervals, must all be the same length
    :param signals: Smoothed spike signals
    :param clusters: Optional, clusters to restrict analysis to. Will use all clusters by default
    :return: X matrix for regression
    """
    if clusters is not None:
        signals = signals[clusters, :]
    window_length = intervals[1][0] - intervals[0][0]

    X = np.zeros(intervals.shape[1], signals.shape[0] * window_length)

    for i in intervals.shape[1]:
        for j in signals.shape[0]:
            X[i, j * window_length: (j+1) * window_length] \
                += signals[j, intervals[0][i]: intervals[1][i]]

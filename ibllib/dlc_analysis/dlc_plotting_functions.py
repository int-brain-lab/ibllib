# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:55:33 2019

List of plotting functions for DLC data

@author: Guido, Kelly
"""

import pandas as pd
import numpy as np
import seaborn as sns


def peri_plot(dlc_traces, timestamps, events, ax, time_win=[-1, 3]):
    """
    Plot a peri-plot centered around a behavioral event
    :dlc_traces:     list with arrays of DLC traces
    :timestamps:     array with timestamps in seconds
    :events: array   with event times to center around
    :time_window:    time window in seconds
    """

    # Transform time window into samples
    sampling_rate = 1/np.mean(np.diff(timestamps))
    sample_win = [np.int(np.round(time_win[0] * sampling_rate)),
                  np.int(np.round(time_win[1] * sampling_rate))]
    time_trace = np.arange(time_win[0], time_win[1], 1/np.round(sampling_rate))

    # Create dataframe for line plot
    peri_df = pd.DataFrame(columns=['event_nr', 'timepoint', 'trace'])
    for i in np.arange(np.size(events)):
        if np.argmin(np.abs(timestamps-events[i]))+sample_win[0] > 0:
            this_trace = dlc_traces[np.argmin(np.abs(timestamps-events[i]))+sample_win[0]:
                                    np.argmin(np.abs(timestamps-events[i]))+sample_win[1]]
            this_df = pd.DataFrame(data={'event_nr': np.ones(np.size(this_trace),
                                                             dtype=int)*(i+1),
                                         'timepoint': time_trace, 'trace': this_trace})
            peri_df = pd.concat([peri_df, this_df], ignore_index=True)

    # Plot
    sns.lineplot(x='timepoint', y='trace', data=peri_df, ax=ax)

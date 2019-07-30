from pathlib import Path

import numpy as np

from brainbox.core import Bunch
from ibllib.io.raw_data_loaders import load_data


def qc_behaviour_bpod_session(ses_path):
    ses_path = Path(ses_path)
    raw_trials = load_data(ses_path)
    if not raw_trials:
        return
    n_trials = len(raw_trials)

    # init the QC dictionary: one row per trial
    qc_trials = Bunch({
        'port1': np.zeros(n_trials) * np.nan,  # port1 out - in lower than 10 ms
        'n_bnc1_high': np.zeros(n_trials) * np.nan,  # number of bnc1 fronts
        'n_bnc2_high': np.zeros(n_trials) * np.nan,  # number of bnc2 fronts
        'n_bnc1_low': np.zeros(n_trials) * np.nan,  # number of bnc1 fronts
        'n_bnc2_low': np.zeros(n_trials) * np.nan,  # number of bnc2 fronts
    })

    for m, raw_trial in enumerate(raw_trials):
        timestamps = raw_trial['behavior_data']['Events timestamps']
        try:
            qc_trials.port1[m] = np.all(np.abs(np.array(timestamps['Port1Out']) -
                                               np.array(timestamps['Port1In']) - 0.01) < 1e3)
        except ValueError:
            pass
        qc_trials.n_bnc1_high[m] = len(timestamps['BNC1High'])
        qc_trials.n_bnc2_high[m] = len(timestamps['BNC2High'])
        qc_trials.n_bnc1_low[m] = len(timestamps['BNC1Low'])
        qc_trials.n_bnc2_low[m] = len(timestamps['BNC2Low'])


if __name__ == "__main__":
    ses_path = '/datadisk/Data/Subjects/ZM_1374/2019-03-24/001'
    qc_behaviour_bpod_session(ses_path)

"""
Use the framework developed by Niccolo B. to review the quality
of the task control.
Load in data saved locally and process.
"""
# Authors: Gaelle

import numpy as np
from ibllib.io import params
from pathlib import Path

rig_location = '_iblrig_mrsicflogel_ephys_0'

one_params = params.read('one_params')
outdir = Path(one_params.CACHE_DIR).joinpath('BPODQC', rig_location)
outfile = Path.joinpath(outdir, 'bpodqc_frame_pass.npz')  # TODO change naming


# Load
varload = np.load(outfile, allow_pickle=True)
l_rig = varload['l_rig']


# --- plotting TODO
# from ibllib.qc.qcplots import rearrange_metrics
# eid, det = random_ephys_session(lab)  # or any one.search that returns details also
# bpod_frame = qcmetrics.get_qcmetrics_frame(eid, data=data)
# df = rearrange_metrics(bpod_frame)
# plot_metrics(df, det, save_path=None)

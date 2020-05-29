"""
STEP2 - Save pandas data frame locally.
A frame contains all data for a rig.
"""
# Author : Gaelle C.
from oneibl.one import ONE
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

one = ONE()
# Get list of all locations (some are labs, some are rigs)
locations = one.alyx.rest('locations', 'list')
# Filter to get only names containing _iblrig_
iblrig = [s['name'] for s in locations if "_iblrig_" in s['name']]
# Filter to get only names containing _ephys_
ephys_rig = [s for s in iblrig if "_ephys_" in s]

# Saving path
cachepath = Path(one._par.CACHE_DIR)


#  for 1 rig at a time

for i_ephysrig in range(0, len(ephys_rig)):
    rig_location = ephys_rig[i_ephysrig]
    datadir = cachepath.joinpath('BPODQC_V2', rig_location)

    # Get session data
    datafiles = os.listdir(datadir)

    # Init save directory
    outdir = cachepath.joinpath('BPODQC_Table', rig_location)
    # Create target Directory if don't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outname = f'{rig_location}__dataframe.npz'
    outfile = Path.joinpath(outdir, outname)

    # Init var
    bm_app = pd.DataFrame()

    for i_file in range(0, len(datafiles)):
        # Load data
        datafile = Path.joinpath(datadir, datafiles[i_file])
        varload = np.load(datafile, allow_pickle=True)
        data = {key: varload[key].item() for key in varload}

        # Concatenate data across sessions for metrics

        dict_met_key = dict()
        for key, value in data['dataqc']['bpod_metrics'].items():
            key_add = f'_metric__{key}'
            dict_met_key[key_add] = value
        bpod_metrics = pd.DataFrame.from_dict(dict_met_key)

        bpod_metrics['eid'] = data['dataqc']['eid']
        bpod_metrics['session'] = i_file
        bpod_metrics['rig_location'] = rig_location

        # Take in value from session details
        for key, value in data['dataqc']['ses_det'].items():
            key_add = f'_sd__{key}'
            bpod_metrics[key_add] = value

        # Take in value from pass
        dict_pass_key = dict()
        for key, value in data['dataqc']['bpod_pass'].items():
            key_add = f'_pass__{key}'
            dict_pass_key[key_add] = value
        bpod_pass = pd.DataFrame.from_dict(dict_pass_key)

        # Append
        var_app = pd.concat([bpod_metrics, bpod_pass], axis=1).copy()
        bm_app = pd.concat([bm_app, var_app], axis=0).copy()

    bm_app.reset_index(inplace=True)

    # Append and save table
    app_token = {
        'dataframe': bm_app
    }
    np.savez(outfile, dataqc=app_token)

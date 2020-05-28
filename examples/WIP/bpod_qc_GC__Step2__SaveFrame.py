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


# Plots for 1 rig at a time

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
    bp_app = pd.DataFrame()

    for i_file in range(0, len(datafiles)):
        # Load data
        datafile = Path.joinpath(datadir, datafiles[i_file])
        varload = np.load(datafile, allow_pickle=True)
        data = {key: varload[key].item() for key in varload}

        # Concatenate data across sessions for metrics
        bpod_metrics = pd.DataFrame.from_dict(data['dataqc']['bpod_metrics'])
        bpod_metrics['eid'] = data['dataqc']['eid']
        bpod_metrics['session'] = i_file
        # Take in value from session details
        for key, value in data['dataqc']['ses_det'].items():
            bpod_metrics[key] = value

        bm_app = pd.concat([bm_app, bpod_metrics]).copy()
        # Do similar for metric pass information
        bpod_pass = pd.DataFrame.from_dict(data['dataqc']['bpod_pass'])
        bp_app = pd.concat([bp_app, bpod_pass]).copy()

    bm_app.reset_index(inplace=True)
    bp_app.reset_index(inplace=True)

    # Append and save table
    app_token = {
        'bm_app': bm_app,
        'bp_app': bp_app
    }
    np.savez(outfile, dataqc=app_token)

    # Plot session metrics
    # i_key = 0  # init iteration value
    # for key, value in bpod_metrics.items():
    #     plt.figure(i_key)  # TODO hacky way to plot on same figure
    #     plt.plot(i_file*np.ones(shape=np.size(value)), value,
    #              '.', alpha=.1, color='b')
    #
    #
    #     plt.title(f'Rig {rig_location}, QC {key}', fontsize=9)
    #     i_key = i_key+1



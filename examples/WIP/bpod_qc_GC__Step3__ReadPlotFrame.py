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
    datadir = cachepath.joinpath('BPODQC_Table', rig_location)

    # Get and load dataframe
    datafileos = os.listdir(datadir)
    datafile = Path.joinpath(datadir, datafileos[0])
    varload = np.load(datafile, allow_pickle=True)
    data = {key: varload[key].item() for key in varload}

    data_bm = pd.DataFrame.from_dict(data['dataqc']['bm_app'])
    data_bp = pd.DataFrame.from_dict(data['dataqc']['bp_app'])

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



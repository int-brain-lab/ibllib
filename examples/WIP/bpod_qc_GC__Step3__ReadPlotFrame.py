"""
STEP3 - Load frames, concatenate across rigs, plot.
"""
# Author : Gaelle C.
from oneibl.one import ONE
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

one = ONE()
# Get list of all locations (some are labs, some are rigs)
locations = one.alyx.rest('locations', 'list')
# Filter to get only names containing _iblrig_
iblrig = [s['name'] for s in locations if "_iblrig_" in s['name']]
# Filter to get only names containing _ephys_
ephys_rig = [s for s in iblrig if "_ephys_" in s]

# Saving path
cachepath = Path(one._par.CACHE_DIR)
outdir = cachepath.joinpath('BPOD_Plot')
# Create target Directory if don't exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Make giant dataframe with data from all rigs
# Init var
bm_app = pd.DataFrame()

for i_ephysrig in range(0, len(ephys_rig)):
    rig_location = ephys_rig[i_ephysrig]
    datadir = cachepath.joinpath('BPODQC_Table', rig_location)

    # Get and load dataframe
    datafileos = os.listdir(datadir)
    datafile = Path.joinpath(datadir, datafileos[0])
    varload = np.load(datafile, allow_pickle=True)
    data = {key: varload[key].item() for key in varload}

    dataframe = pd.DataFrame.from_dict(data['dataqc']['dataframe'])

    # Append
    bm_app = pd.concat([bm_app, dataframe], axis=0).copy()

bm_app.reset_index(inplace=True)

# Find metric key in dataframe
metric_name = [key for key in dataframe.keys() if '_metric__' in key.lower()]

for i_metric in range(0, len(metric_name)):
    # Plot session metrics
    fig, axes = plt.subplots(1, 2)
    name_pass = f'_pass__{metric_name[i_metric][9:]}'

    sns.scatterplot(x=metric_name[i_metric],
                    y="eid",
                    hue=bm_app["rig_location"].astype('category'),
                    data=bm_app,
                    marker=".", alpha=0.3, edgecolor="none",
                    ax=axes[0])


    sns.scatterplot(x=metric_name[i_metric],
                    y="eid",
                    hue=bm_app[name_pass].astype('category'),
                    data=bm_app,
                    marker=".", alpha=0.3, edgecolor="none",
                    ax=axes[1])

    axes[1].set_yticks([])
    axes[1].xaxis.set_ticklabels([])


    # Save fig
    outname = f'{metric_name[i_metric]}.png'
    outfile = Path.joinpath(outdir, outname)
    plt.savefig(outfile)

    # Close fig
    plt.close(fig)

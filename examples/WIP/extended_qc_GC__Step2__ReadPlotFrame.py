"""
STEP2 - Load frames, concatenate across rigs, plot.
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

# Saving path
cachepath = Path(one._par.CACHE_DIR)
outdir = cachepath.joinpath('EXT_Plot_V1')
# Create target Directory if don't exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Make giant dataframe with data from all rigs
# Init var
bm_app = pd.DataFrame()
datadir = cachepath.joinpath('EXT_V1')
datafileos = os.listdir(datadir)
all_dataframe = pd.DataFrame()

for i_sess in range(0, len(datafileos)):
    # Get and load dataframe
    datafile = Path.joinpath(datadir, datafileos[i_sess])
    # Check for extension
    name, ext = os.path.splitext(datafile)
    if ext == '.npz':
        try:
            varload = np.load(datafile, allow_pickle=True)
            data = {key: varload[key].item() for key in varload}

            sess_details = pd.DataFrame.from_dict(data['dataqc']['sess_details'])

            test_details = data['dataqc']['test_details']  # Still dict
            d2 = {}
            for t in test_details:
                for k in test_details[t]:
                    d2[k] = t
            test_details_df = pd.DataFrame(data=d2, index=[0])

            test_details_df.insert(0, "eid", sess_details['eid'])
            test_details_df.insert(0, "rig_location", sess_details['rig_location'])

            all_dataframe = pd.concat([all_dataframe, test_details_df], axis=0).copy()
        except Exception:
            print(f'{datafile} failed, i_sess = [{i_sess}]')

# Plot
metric_name = [key for key in all_dataframe.keys() if '_bpod_' in key.lower()]

for i_metric in range(0, len(metric_name)):
    metric = metric_name[i_metric]
    chart = sns.countplot(x="rig_location", hue=metric,
                          data=all_dataframe, palette=sns.color_palette("husl", 8),
                          hue_order=["CRITICAL", "ERROR", "WARNING", "PASS"])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()

    # Save fig
    outname = f'{metric}.png'
    outfile = Path.joinpath(outdir, outname)
    plt.savefig(outfile)

    # Close fig
    plt.close()

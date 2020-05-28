"""
Use the framework developed by Niccolo B. to review the quality
of the task control.

Download the raw data locally, and perform computations to
retrieve the QC data frame (values of metrics, and pass/fail).
Some sessions do not work with this scheme (exceptions), and
are listed to be skipped from computations.

Save data locally per EID as computations take time.
"""
# Author : Gaelle C.
from oneibl.one import ONE
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
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


# Plots for 1 rig at a time

for i_ephysrig in range(0, len(ephys_rig)):
    rig_location = ephys_rig[i_ephysrig]
    datadir = cachepath.joinpath('BPODQC_V2', rig_location)

    # Save plot folder
    outdir = cachepath.joinpath('BPODQC_PLOT', rig_location)
    # Create target Directory if don't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get session data
    datafiles = os.listdir(datadir)

    #init var
    ei_app = list()
    bm_app = list()
    bp_app = list()
    se_app = list()

    for i_file in range(0, len(datafiles)):
        # Load data
        datafile = Path.joinpath(datadir, datafiles[i_file])
        varload = np.load(datafile, allow_pickle=True)
        data = {key: varload[key].item() for key in varload}

        bpod_metrics = data['dataqc']['bpod_metrics']
        bpod_pass = data['dataqc']['bpod_pass']

        # Plot session metrics
        i_key = 0  # init iteration value
        for key, value in bpod_metrics.items():
            plt.figure(i_key)  # TODO hacky way to plot on same figure
            plt.plot(i_file*np.ones(shape=np.size(value)), value,
                     '.', alpha=.1, color='b')
            

            plt.title(f'Rig {rig_location}, QC {key}', fontsize=9)
            i_key = i_key+1


    #     ei_app.append(data['dataqc']['eid'])
    #     bm_app.append(bpod_metrics)
    #     bp_app.append(data['dataqc']['bpod_pass'])
    #     se_app.append(data['dataqc']['ses_det'])
    #
    # date = [p['start_time'] for p in se_app]
    # i_key = 0  # init iteration value
    # for key, value in bm_app[0].items():
    #     metric_array = [p[key] for p in bm_app]
    #     plt.figure(i_key)
    #     plt.plot(metric_array, '.', alpha=.5)
    #     plt.title(f'Rig {rig_location}, QC {key}', fontsize=9)
    #     i_key = i_key+1


    # Save figure
    for i_fig in range(0, i_key):
        plt.figure(i_fig)
        fname = f'QC{i_fig}_{rig_location}.png'
        plt.savefig(Path.joinpath(outdir, fname))
        plt.close(i_fig)
    # outname = f'{eid}__dataqc.npz'
    # outfile = Path.joinpath(outdir, outname)

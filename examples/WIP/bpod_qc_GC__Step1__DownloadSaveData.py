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
import numpy as np
from pathlib import Path
import os
import pandas as pd
from ibllib.qc import BpodQC
from ibllib.qc import oneutils

one = ONE()
# Get list of all locations (some are labs, some are rigs)
locations = one.alyx.rest('locations', 'list')
# Filter to get only names containing _iblrig_
iblrig = [s['name'] for s in locations if "_iblrig_" in s['name']]
# Filter to get only names containing _ephys_
ephys_rig = [s for s in iblrig if "_ephys_" in s]

# -- Var init
# dtypes = ['ephysData.raw.lf', 'ephysData.raw.meta', 'ephysData.raw.ch']
dtypes = ['_iblrig_taskData.raw']

# Saving path
cachepath = Path(one._par.CACHE_DIR)

# Load list of reject eids
list_file = cachepath.joinpath('eid_reject_list.npz')
if list_file not in os.listdir(cachepath):
    varload = np.load(list_file, allow_pickle=True)
    list_eid_reject = list(varload['list_eid'])
else:
    list_eid_reject = list()

# Plots for 1 rig at a time
for i_ephysrig in range(0, len(ephys_rig)):
    rig_location = ephys_rig[i_ephysrig]

    # Save folder
    outdir = cachepath.joinpath('BPODQC_V2', rig_location)
    # Create target Directory if don't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # list files in save folder
    files_exist = os.listdir(outdir)

    # Get session eIDs, for 1 rig
    eIDs, ses_det = one.search(
        location=rig_location,
        dataset_types=dtypes,
        task_protocol='_iblrig_tasks_ephysChoiceWorld',
        details=True)

    # Download all session and save data frame

    for i_eid in range(0, len(eIDs)):

        eid = eIDs[i_eid]
        outname = f'{eid}__dataqc.npz'
        outfile = Path.joinpath(outdir, outname)

        if (eid not in list_eid_reject) and \
                (outname not in os.listdir(outdir)):
            # Show session number and start compute time counter for session
            print(f'Rig {i_ephysrig + 1} / {len(ephys_rig)} : {rig_location}'
                  f' -- Sessions remaining: {len(eIDs)-len(os.listdir(outdir))-1}'
                  f' -- {eid}')

            # -- Dowload data if necessary
            oneutils.download_bpodqc_raw_data(eid, one=one)

            # -- Start compute, run over exceptions
            try:
                bpodqc = BpodQC(eid, one=one, ensure_data=True, lazy=False)
                passed_df = pd.DataFrame.from_dict(bpodqc.passed)
                metrics_df = pd.DataFrame.from_dict(bpodqc.metrics)

                # -- Append and save variables
                app_token = {
                    'eid': eid,
                    'bpod_metrics': metrics_df,
                    'bpod_pass': passed_df,
                    'ses_det': ses_det[i_eid]
                }
                np.savez(outfile, dataqc=app_token)  # overwrite any existing file
            except Exception:
                list_eid_reject.append(eid)
                np.savez(list_file, list_eid=list_eid_reject)
                pass

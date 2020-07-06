"""
Compute extended QC to get session status.
"""
# Author : Gaelle C.
from ibllib.qc.extended_qc import compute_session_status
from ibllib.qc import ExtendedQC
from oneibl.one import ONE
import pandas as pd
from pathlib import Path
import os
import numpy as np

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
list_file = cachepath.joinpath('EXTQC_eid_reject_list.npz')
if list_file in os.listdir(cachepath):
    varload = np.load(list_file, allow_pickle=True)
    list_eid_reject = list(varload['list_eid'])
else:
    list_eid_reject = list()

# Save folder
outdir = cachepath.joinpath('EXT_V1')
# Create target Directory if don't exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

all_dataframe = pd.DataFrame()
for i_ephysrig in range(0, len(ephys_rig)):
    rig_location = ephys_rig[i_ephysrig]

    # Get session eIDs, for 1 rig
    eIDs, ses_det = one.search(
        location=rig_location,
        dataset_types=dtypes,
        task_protocol='_iblrig_tasks_ephysChoiceWorld',
        details=True)

    for i_eid in range(0, len(eIDs)):
        eid = eIDs[i_eid]
        outname = f'{eid}__extqc.npz'
        outfile = Path.joinpath(outdir, outname)

        if (eid not in list_eid_reject) and \
                (outname not in os.listdir(outdir)):
            # Show session number and start compute time counter for session
            print(f'Rig {i_ephysrig + 1} / {len(ephys_rig)} : {rig_location}'
                  f' -- Sessions remaining: {len(eIDs)-len(os.listdir(outdir))-1}'
                  f' -- {eid}')

            try:
                ext = ExtendedQC(eid=eid, one=one, lazy=False)
                criteria, out_var_test_status, out_var_sess_status = \
                    compute_session_status(ext.frame)

                d = {'sess_status': out_var_sess_status,
                     'eid': eid,
                     'rig_location': rig_location
                     }
                sess_dataframe = pd.DataFrame(data=d, index=[0])

                # Append and save table
                app_token = {
                    'sess_details': sess_dataframe,
                    'test_details': out_var_test_status,
                    'dataframe': ext.frame
                }
                np.savez(Path.joinpath(outdir, f'{eid}__sess_det.npz'), dataqc=app_token)

                all_dataframe = pd.concat([all_dataframe, sess_dataframe], axis=0).copy()

            except Exception:
                list_eid_reject.append(eid)
                np.savez(list_file, list_eid=list_eid_reject)
                pass

# Save folder
outdir = cachepath.joinpath('EXT_All')
# Create target Directory if don't exist
if not os.path.exists(outdir):
    os.makedirs(outdir)
outname = 'Ext_QC_All.npz'
outfile = Path.joinpath(outdir, outname)

if outfile in os.listdir(outdir):
    varload = np.load(outfile, allow_pickle=True)
    data = {key: varload[key].item() for key in varload}
    dataframe = pd.DataFrame.from_dict(data['dataqc']['dataframe'])
    all_dataframe = pd.concat([all_dataframe, dataframe], axis=0).copy()

# Append and save table
app_token = {
    'dataframe': all_dataframe
}
np.savez(outfile, dataqc=app_token)

# # Plot
# import seaborn as sns
# # Fig 1 (status overall)
# ax = sns.countplot(x="sess_status", data=all_dataframe,
#                    palette=sns.color_palette("husl", 8))
# # Fig 2 (status per rig)
# chart = sns.countplot(x="rig_location", hue="sess_status",
#                       data=all_dataframe, palette=sns.color_palette("husl", 8),
#                       hue_order=["CRITICAL", "ERROR", "WARNING", "PASS"])
# chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
#

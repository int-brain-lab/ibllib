"""
Compute extended QC to get session status.
"""
# Author : Gaelle C.
from ibllib.qc.extended_qc import compute_session_status
from ibllib.qc import ExtendedQC

from oneibl.one import ONE
import pandas as pd

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

# Plots for 1 rig at a time
for i_ephysrig in range(0, len(ephys_rig)):
    rig_location = ephys_rig[i_ephysrig]

    # Get session eIDs, for 1 rig
    eIDs, ses_det = one.search(
        location=rig_location,
        dataset_types=dtypes,
        task_protocol='_iblrig_tasks_ephysChoiceWorld',
        details=True)

    rig_dataframe = pd.DataFrame()
    for i_eid in range(0, len(eIDs)):
        eid = eIDs[i_eid]
        ext = ExtendedQC(eid=eid, one=one, lazy=False)
        criteria, out_var_test_status, out_var_sess_status = compute_session_status(ext.frame)

        d = {'sess_status': out_var_sess_status,
             'eid': eid,
             'rig_location': rig_location
             }
        sess_dataframe = pd.DataFrame(data=d, index=[0])

        rig_dataframe = pd.concat([rig_dataframe, sess_dataframe], axis=0).copy()

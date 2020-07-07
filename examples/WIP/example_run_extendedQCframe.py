"""
For a given session, download data and compute ExtendedQC,
then compute session/test metrics
"""
# Author: Gaelle Chapuis
from ibllib.qc import oneutils
from oneibl.one import ONE
from ibllib.qc.extended_qc import compute_session_status
from ibllib.qc import ExtendedQC

one = ONE()

# Select session
eIDs, ses_det = one.search(
    subjects='CSHL046',
    date_range='2020-06-18',
    task_protocol='_iblrig_tasks_ephysChoiceWorld',
    details=True)
eid = eIDs[0]

# Download data
oneutils.download_bpodqc_raw_data(eid, one=one)

# Compute QC frame
ext = ExtendedQC(eid=eid, one=one, lazy=False)

# Compute session/test metrics
criteria, out_var_test_status, out_var_sess_status = \
    compute_session_status(ext.frame)

print(f'Session status: {out_var_sess_status}')

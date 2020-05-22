from ibllib.qc import BpodQC, BpodQCExtractor, ONEQC, ExtendedQC
from ibllib.qc import oneutils
from oneibl.one import ONE
import ibllib.qc.qcplots as qcplots
import pandas as pd


# Get one
one = ONE()
# Find eid of session
eid, det = oneutils.random_ephys_session()
# Alternatively use a session path
session_path = one.path_from_eid(eid)
# Make sure raw data is available locally for computing BpodQC
oneutils.download_bpodqc_raw_data(eid, one=one)

# Call the object with either a path or an eid
# if you don't pass ONE it will create it's own instance pass ONE obj to be
# sure you know which connection you are using
# ensure_data will try to download the data for you if no data is on server will return an error
# If lazy is tru it won't load, extract or run the qc
bpodqc = BpodQC(eid or session_path, one=one, ensure_data=True, lazy=False)
# QC metrics and passed dictionaries are stored here, only computes once
bpodqc.metrics
bpodqc.passed

# If you want to plot metrics and the passed bpod frame you'll need to convert these to pandas df
passed_df = pd.DataFrame.from_dict(bpodqc.passed)
metrics_df = pd.DataFrame.from_dict(bpodqc.metrics)
# get a title for the plot
session_name = "/".join(bpodqc.session_path.parts[-3:])
qcplots.boxplot_metrics(metrics_df, ax=None, describe=False, title=session_name,
                        xlabel='Seconds (s)', xscale='symlog')
qcplots.barplot_passed(passed_df, ax=None, describe=False, title=session_name,
                       xlabel='Proportion of trials that pass criteria', save_path=None)


# If you whish to get the pre generated frame from alyx of the passed metrics:
extended_qc = ExtendedQC(one=one, eid=eid, lazy=True)

alyx_extended_qc_frame = extended_qc.read_extended_qc()

alyx_eqc = pd.DataFrame(alyx_extended_qc_frame, index=[0])
qcplots.barplot_passed(alyx_eqc, title=session_name)
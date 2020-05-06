import ibllib.qc.bpodqc_extractors as bpodqc
import ibllib.ephys.ephysqc as ephysqc
from oneibl.one import ONE

one = ONE()

eid = ''

# get number of files extracted from eid
...
# Get bpod and ephys qc frames
fpgaqc_frame = ephysqc._qc_from_path(session_path, display=False)
bpodqc_frame = bpodqc.get_bpodqc_frame(session_path)
# aggregate them

# ad-hoc aggregation for some variables
extended_qc = {
    'feedback': np.sum(qc_frame.n_feedback != 0) / ntrials,
}
# here we average all bools
for k in qc_frame.keys():
    if qc_frame[k].dtype == np.dtype('bool'):
        extended_qc[k] = np.mean(qc_frame[k])
    elif k.endswith('_times'):
        # Todo remove intervals_start from there if it's an absolute time
        # std + mean ?
        pass
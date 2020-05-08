import numpy as np

import ibllib.qc.bpodqc_metrics as bpodqc
import ibllib.qc.oneqc_metrics as oneqc
from ibllib.qc.bpodqc_extractors import load_bpod_data
from ibllib.qc.oneutils import random_ephys_session
from oneibl.one import ONE

one = ONE()

eid, det = random_ephys_session()


def build_extended_qc_frame(eid, data=None):
    if data is None:
        data = load_bpod_data(eid)

    # Get bpod and one qc frames
    extended_qc = {}
    print("Running QC on ONE DatasetTypes...")
    one_frame = oneqc.get_oneqc_metrics_frame(eid, data=data, apply_criteria=True)
    print("Running QC on Bpod data...")
    bpod_frame = bpodqc.get_bpodqc_metrics_frame(eid, data=data, apply_criteria=True)
    # Make average bool pass
    # def average_frame(frame):
    #     return {k: np.nanmean(v) for k, v in frame.items()}
    average_bpod_frame = (lambda frame: {k: np.nanmean(v) for k, v in frame.items()})(bpod_frame)
    # aggregate them
    extended_qc.update(one_frame)
    extended_qc.update(average_bpod_frame)
    return extended_qc


# ad-hoc aggregation for some variables
extended_qc = {
    "feedback": np.sum(qc_frame.n_feedback != 0) / ntrials,
}
# here we average all bools
for k in qc_frame.keys():
    if qc_frame[k].dtype == np.dtype("bool"):
        extended_qc[k] = np.mean(qc_frame[k])
    elif k.endswith("_times"):
        # Todo remove intervals_start from there if it's an absolute time
        # std + mean ?
        pass


if __name__ == "__main__":
    extended_qc = {
        "_one_nDatasetTypes": None,
        "_one_intervals_length": None,
        "_one_intervals_count": None,
        "_one_stimOnTrigger_times_length": None,
        "_one_stimOnTrigger_times_count": None,
        "_one_stimOn_times_length": None,
        "_one_stimOn_times_count": None,
        "_one_goCueTrigger_times_length": None,
        "_one_goCueTrigger_times_count": None,
        "_one_goCue_times_length": None,
        "_one_goCue_times_count": None,
        "_one_response_times_length": None,
        "_one_response_times_count": None,
        "_one_feedback_times_length": None,
        "_one_feedback_times_count": None,
        "_one_goCueTriggeer_times_length": None,
        "_one_goCueTriggeer_times_count": None,
        "_bpod_goCue_delays": None,
        "_bpod_errorCue_delays": None,
        "_bpod_stimOn_delays": None,
        "_bpod_stimOff_delays": None,
        "_bpod_stimFreeze_delays": None,
        "_bpod_stimOn_goCue_delays": None,
        "_bpod_response_feedback_delays": None,
        "_bpod_response_stimFreeze_delays": None,
        "_bpod_stimOff_itiIn_delays": None,
        "_bpod_wheel_freeze_during_quiescence": None,
        "_bpod_wheel_move_before_feedback": None,
        "_bpod_wheel_move_during_closed_loop": None,
        "_bpod_stimulus_move_before_goCue": None,
        "_bpod_positive_feedback_stimOff_delays": None,
        "_bpod_negative_feedback_stimOff_delays": None,
        "_bpod_valve_pre_trial": None,
        "_bpod_audio_pre_trial": None,
        "_bpod_error_trial_event_sequence": None,
        "_bpod_correct_trial_event_sequence": None,
        "_bpod_trial_length": None,
    }


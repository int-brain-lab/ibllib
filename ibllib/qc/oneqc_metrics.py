import matplotlib.pyplot as plt
import numpy as np

import ibllib.qc.bpodqc_extractors as bpodqc
from ibllib.qc.bpodqc_extractors import bpod_data_loader
from ibllib.qc.oneutils import random_ephys_session
from oneibl.one import ONE

plt.ion()


one = ONE()


def get_oneqc_metrics_frame(eid, data=None, apply_criteria=False):
    """Full extended_qc_frame
    (one value per metric as proportion of trial level criteria that passed)"""
    qcmetrics_frame = {
        "_one_nDatasetTypes": None,  # (Point 17)
        "_one_intervals_length": None,  # (Point 18)
        "_one_intervals_count": None,
        "_one_stimOnTrigger_times_length": None,  # (Point 19)
        "_one_stimOnTrigger_times_count": None,
        "_one_stimOn_times_length": None,  # (Point 20)
        "_one_stimOn_times_count": None,
        "_one_goCueTrigger_times_length": None,  # (Point 21)
        "_one_goCueTrigger_times_count": None,
        "_one_goCue_times_length": None,  # (Point 22)
        "_one_goCue_times_count": None,
        "_one_response_times_length": None,  # (Point 23)
        "_one_response_times_count": None,
        "_one_feedback_times_length": None,  # (Point 24)
        "_one_feedback_times_count": None,
    }

    dstype_names = [
        "trials.intervals",
        "trials.stimOnTrigger_times",
        "trials.stimOn_times",
        "trials.goCueTrigger_times",
        "trials.goCue_times",
        "trials.response_times",
        "trials.feedback_times",
    ]
    for name in dstype_names:
        qcmetrics_frame.update(
            load_dstype_qc_metrics(eid, name, data=data, apply_criteria=apply_criteria)
        )
    return qcmetrics_frame


# ---------------------------------------------------------------------------- #
# ONE qc is atm just counting nans (*_count) or comparing the dims to the bpod "ground truth" data
#
#  bpod_ntrials = len(raw.load_data(one.path_from_eid(eid)))
def load_nDatasetTypes(eid, data=None, apply_criteria=False):
    """ 17. Proportion of datasetTypes extracted
    Variable name: nDatasetTypes
    Metric: len(one.load(eid, offline=True, download_only=True)) / nExpetedDatasetTypes
    (hardcoded per task?)
    """
    return


def load_dstype_qc_metrics(
    eid: str, dstype_name: str, data: dict = None, apply_criteria: bool = False
) -> dict:
    """Returns dict to update to metrics or criteria frame
    Metrics:
        _length = number of trials in ONE dstype
        _count = number of nans in dstype
    Criteria:
        length / bpod number of trials
        count / length of dstype
    NB: Makes sense for dstypes that should have one value per trial and where nans
    are informative of failures
    Other dstypes will have nans because of the contingency of the trial,
    e.g. if contralstLeft has a nan it means the contrast was on the right.
    """
    name = dstype_name.replace("trials.", "")
    # Add namespace and termination strings
    names = [f"_one_{name}_length", f"_one_{name}_count"]
    # Create output dict
    out = dict.fromkeys(names)
    # Load dset data from ONE
    dset = one.load(eid, dataset_types=dstype_name)[0]
    if dset is None:
        return out
    # Define length and count as metric
    _length = len(dset)
    _count = np.sum(np.isnan(dset))
    # if criteria is applies output normalized len and count
    if apply_criteria:
        _length = _length / len(data["intervals_0"])
        _count = 1 - _count / len(dset)
    # Add the values to output dict
    for k, v in zip(names, (_length, _count)):
        out[k] = v

    return out


if __name__ == "__main__":
    eid, det = random_ephys_session()
    data = bpodqc.load_bpod_data(eid, fpga_time=False)
    metrics = get_oneqc_metrics_frame(eid, data=data, apply_criteria=False)
    criteria = get_oneqc_metrics_frame(eid, data=data, apply_criteria=True)

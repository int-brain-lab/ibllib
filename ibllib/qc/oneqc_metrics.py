import logging
import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.qc.oneutils import random_ephys_session
from oneibl.one import ONE

log = logging.getLogger("ibllib")


class ONEQC(object):
    def __init__(self, eid, one=None, bpod_ntrials=None, lazy=True):
        self.eid = eid
        self.one = one or ONE()
        self.bpod_ntrials = bpod_ntrials or np.nan

        self.metrics = None
        self.passed = None

        if not lazy:
            self.compute()

    def compute(self):
        log.info(f"Session {self.eid}: Running QC on ONE DatasetTypes...")
        self.metrics, self.passed = get_oneqc_metrics_frame(
            self.eid, self.bpod_ntrials, one=self.one
        )


def get_oneqc_metrics_frame(eid, bpod_ntrials, one=None):
    one = one or ONE(printout=False)
    """(one value per metric as proportion of trial level criteria that passed)"""
    qcmetrics_frame = {}
    qcmetrics_frame.update(load_nDatasetTypes(eid))
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
        qcmetrics_frame.update(load_dstype_qc_metrics(eid, name, bpod_ntrials, one))

    # Split metrics and passed frames
    metrics = {}
    passed = {}
    for k in qcmetrics_frame:
        metrics[k], passed[k] = qcmetrics_frame[k]

    return (metrics, passed)


# ---------------------------------------------------------------------------- #
# ONE qc is atm just counting nans (*_count) or
# comparing the dims to the bpod "ground truth" data (*_length)
#
#  bpod_ntrials = len(raw.load_data(one.path_from_eid(eid)))
def load_nDatasetTypes(eid):
    """ 17. Proportion of datasetTypes extracted
    Variable name: nDatasetTypes
    Metric: len(one.load(eid, offline=True, download_only=True)) / nExpetedDatasetTypes
    (hardcoded per task?)
    """
    out = {"_one_nDatasetTypes": (None, None)}
    return out


def load_dstype_qc_metrics(
    eid: str, dstype_name: str, bpod_ntrials: int = None, one: object = None,
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
    out = {k: (None, None) for k in out}
    # Load dset data from ONE
    dset = one.load(eid, dataset_types=dstype_name)[0]
    if dset is None:
        return out
    # Define length and count as metric
    # Define criteria is applies output normalized len and count
    _count = np.sum(np.isnan(dset))
    _length = len(dset)
    _count = (_count, 1 - _count / _length)  # len(dset)
    _length = (_length, _length / bpod_ntrials)

    # Add the values to output dict
    for k, v in zip(names, (_length, _count)):
        out[k] = v

    return out


if __name__ == "__main__":
    one = ONE()
    eid, det = random_ephys_session()
    session_path = one.path_from_eid(eid)
    data = raw.load_data(session_path)
    # metrics, passed = get_oneqc_metrics_frame(eid, len(data))
    # metrics, passed = get_oneqc_metrics_frame(eid, np.inf)
    met_obj = ONEQC(eid, one=one, bpod_ntrials=len(data), lazy=False)
    ""

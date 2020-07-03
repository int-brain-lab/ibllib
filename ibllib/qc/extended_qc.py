import logging

import numpy as np

from alf.io import is_uuid_string
from ibllib.qc import ONEQC, BpodQC
from oneibl.one import ONE

log = logging.getLogger("ibllib")

# one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")

# eid, det = random_ephys_session()

# eid = "4153bd83-2168-4bd4-a15c-f7e82f3f73fb"
# det = one.get_details(eid)
# details = one.get_details(eid, full=True)


def compute_session_status(frame, criteria=None):
    """
    :param frame: dictionary of QC test, with 1 value per test reflecting a probability
    :param criteria: dictionary defining lower bounds (values) of categories (keys),
    written in ascending orders of importance (most critical first). Max bound: <=1.
    e.g. ERROR is >= 0.75, <0.95 in default:
    criteria = {"CRITICAL": 0,
                "ERROR": 0.75,
                "WARNING": 0.95,
                "PASS": 0.99 }
    :return: criteria, out_var_test_status, out_var_sess_status
    """
    # Set default value
    MAX_BOUND = 1
    MIN_BOUND = 0
    CRITERIA = {"CRITICAL": 0,
                "ERROR": 0.75,
                "WARNING": 0.95,
                "PASS": 0.99
                }

    out_var_sess_status = []

    criteria = criteria or CRITERIA

    keys_crit = list(criteria.keys())
    # Prepare out variable with same format as criteria
    out_var_test_status = dict.fromkeys(keys_crit)

    # Get values and key to compute test / session status

    values_f = np.array(list(frame.values()), dtype=np.float)
    # Remove Nans
    indx_remove_test = np.where(np.isnan(values_f))
    values_f = np.delete(values_f, indx_remove_test)

    keys_f = np.array(list(frame.keys()))
    keys_f = np.delete(keys_f, indx_remove_test)

    # Check range of values
    if np.logical_or(
            np.any(values_f > MAX_BOUND),
            np.any(values_f < MIN_BOUND)):
        raise ValueError("Values out of bound")



    # Find tests that fall under a given criterion
    for i_key in range(0, len(keys_crit)):
        # Get lower threshold
        threshold_lower = criteria[keys_crit[i_key]]
        # Get upper threshold and find index of corresponding values
        if i_key == len(keys_crit) - 1:
            threshold_upper = 1
            indx = np.where(np.logical_and(
                values_f >= threshold_lower, values_f <= threshold_upper))[0]
        else:
            threshold_upper = criteria[keys_crit[i_key + 1]]
            indx = np.where(np.logical_and(
                values_f >= threshold_lower, values_f < threshold_upper))[0]
        # Get name of test corresponding to index and save in out var
        out_var_test_status[keys_crit[i_key]] = keys_f[indx]

        # Assign session status
        # Note: The order of criteria importance has to be respected for this to run correctly
        if len(out_var_sess_status) == 0 and len(out_var_test_status[keys_crit[i_key]]) != 0:
            out_var_sess_status = keys_crit[i_key]

    # Check each test is assigned only to one status
    assigned = np.concatenate(list(out_var_test_status.values()), axis=0)
    # Compare with keys_f, removing the Nan type first
    if not np.array_equal(np.sort(assigned), np.sort(keys_f)):
        raise ValueError("One test has to be assigned to one status - this is not the case.")

    return criteria, out_var_test_status, out_var_sess_status


class ExtendedQC(object):
    def __init__(self, one=None, eid=None, lazy=True):
        self.one = one or ONE()
        self.eid = eid if is_uuid_string(eid) else None

        self.bpodqc = None
        self.oneqc = None
        self.frame = None

        if not lazy:
            self.compute_all_qc()
            self.build_extended_qc_frame()

    def compute_all_qc(self):
        self.bpodqc = BpodQC(self.eid, one=self.one, lazy=False)
        self.oneqc = ONEQC(
            self.eid, one=self.one, bpod_ntrials=self.bpodqc.bpod_ntrials, lazy=False
        )

    def build_extended_qc_frame(self):
        if self.bpodqc is None:
            self.compute_all_qc()
        # Get bpod and one qc frames
        extended_qc = {}
        # Make average bool pass for bpodqc.metrics frame
        # def average_frame(frame):
        #     return {k: np.nanmean(v) for k, v in frame.items()}
        average_bpod_frame = (lambda frame: {k: np.nanmean(v) for k, v in frame.items()})(
            self.bpodqc.passed
        )
        # aggregate them
        extended_qc.update(self.oneqc.passed)
        extended_qc.update(average_bpod_frame)
        # Ensure None instead of NaNs
        for k, v in extended_qc.items():
            if v is not None and np.isnan(v):
                extended_qc[k] = None

        self.frame = extended_qc

    def read_extended_qc(self):
        return self.one.alyx.rest("sessions", "read", id=self.eid)["extended_qc"]

    def update_extended_qc(self):
        if self.frame is None:
            log.warning("ExtendedQC frame is not built yet, nothing to update")
            return

        out = self.one.alyx.json_field_update(
            endpoint="sessions", uuid=self.eid, field_name="extended_qc", data=self.frame
        )
        return out

    def compute_session_status(self, crit):
        return compute_session_status(self.frame)

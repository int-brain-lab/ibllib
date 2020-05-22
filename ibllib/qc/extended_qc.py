import logging

import numpy as np

import ibllib.qc.bpodqc_metrics as bpodqc
import ibllib.qc.oneqc_metrics as oneqc
from ibllib.qc.oneutils import random_ephys_session
from oneibl.one import ONE
from alf.io import is_uuid_string

from ibllib.qc import BpodQC
from ibllib.qc import ONEQC

log = logging.getLogger("ibllib")

# one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")

# eid, det = random_ephys_session()

# eid = "4153bd83-2168-4bd4-a15c-f7e82f3f73fb"
# det = one.get_details(eid)
# details = one.get_details(eid, full=True)


class ExtendedQC(object):
    def __init__(self, one=None, eid=None, lazy=True):
        self.one = one or ONE(printout=False)
        self.eid = eid if is_uuid_string(eid) else None

        self.bpodqc = BpodQC(eid, one=self.one, lazy=lazy)
        self.oneqc = ONEQC(
            self.eid, one=self.one, bpod_ntrials=self.bpodqc.bpod_ntrials, lazy=lazy
        )
        self.frame = None

        if not lazy:
            self.build_extended_qc_frame()

    def compute_qc(self):
        self.bpodqc.compute()
        self.oneqc.compute()

    def build_extended_qc_frame(self):
        if self.bpodqc.passed is None:
            self.compute_qc()
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

        self.frame = extended_qc

    def read_extended_qc(self):
        return self.one.alyx.rest("sessions", "read", id=self.eid)["extended_qc"]

    def update_extended_qc(self):
        return self.one.alyx.json_field_update(
            endpoint="sessions", uuid=self.eid, field_name="extended_qc", data=self.frame
        )


if __name__ == "__main__":
    from pyinstrument import Profiler

    eid, det = random_ephys_session("churchlandlab")
    # trial_data = bpodqc.extract_bpod_trial_table(eid, fpga_time=False)

    profiler = Profiler()
    profiler.start()

    # code you want to profile
    # metrics = get_bpodqc_metrics_frame(eid, trial_data=trial_data)
    # criteria = get_bpodqc_metrics_frame(eid, trial_data=trial_data, apply_criteria=True)
    # mean_criteria = {k: np.nanmean(v) for k, v in criteria.items()}
    extended_qc = ExtendedQC(one=None, eid=eid, lazy=True)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))

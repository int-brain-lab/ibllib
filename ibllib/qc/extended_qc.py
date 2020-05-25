import logging

import numpy as np

from oneibl.one import ONE
from alf.io import is_uuid_string

from ibllib.qc import BpodQC, ONEQC

log = logging.getLogger("ibllib")

# one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")

# eid, det = random_ephys_session()

# eid = "4153bd83-2168-4bd4-a15c-f7e82f3f73fb"
# det = one.get_details(eid)
# details = one.get_details(eid, full=True)


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

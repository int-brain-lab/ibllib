import numpy as np

from . import base
from alf.io import is_uuid_string


class ONEQC(base.QC):
    def __init__(self, eid, one=None, bpod_ntrials=None, lazy=False):
        super().__init__(eid, one)
        self.details = self.one.get_details(self.eid, full=True)
        self.bpod_ntrials = bpod_ntrials or self.details["n_trials"]

        if not lazy:
            self.compute()

    def _set_eid_or_path(self, session_path_or_eid):
        """Overloading base: session path not supported"""
        if is_uuid_string(str(session_path_or_eid)):
            self.eid = session_path_or_eid
            # Try to set session_path if data is found locally
            self.session_path = self.one.path_from_eid(self.eid)
        else:
            self.log.error("Cannot run ONE QC: an experiment uuid requried")
            raise ValueError("'session' must be a valid session uuid")

    def compute(self):
        self.log.info(f"Session {self.eid}: Running QC on ONE DatasetTypes...")
        """(one value per metric as proportion of trial level criteria that passed)"""
        qcmetrics_frame = {}
        qcmetrics_frame.update(self.load_nDatasetTypes())
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
            qcmetrics_frame.update(self.load_dstype_qc_metrics(name))

        # Split metrics and passed frames
        metrics = {}
        passed = {}
        for k in qcmetrics_frame:
            metrics[k], passed[k] = qcmetrics_frame[k]

        self.metrics = metrics
        self.passed = passed
        return

    # ---------------------------------------------------------------------------- #
    # ONE qc is atm just counting nans (*_count) or
    # comparing the dims to the bpod "ground truth" data (*_length)
    #
    #  bpod_ntrials = len(raw.load_data(one.path_from_eid(eid)))

    def load_nDatasetTypes(self):
        """ 17. Proportion of datasetTypes extracted
        Variable name: nDatasetTypes
        Metric: len(one.load(eid, offline=True, download_only=True)) / nExpetedDatasetTypes
        (hardcoded per task?)
        """
        self.log.warning("QC test not implemented: nDatasetTypes")
        out = {"_one_nDatasetTypes": (None, None)}
        return out

    def load_dstype_qc_metrics(self, dstype_name: str) -> dict:
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
        dset = self.one.load(self.eid, dataset_types=dstype_name)[0]
        if dset is None:
            self.log.warning(f"ONE datasetType not found: {dstype_name}")
            return out
        # Define length and count as metric
        # Define criteria is applies output normalized len and count
        _count = np.sum(np.isnan(dset))
        _length = len(dset)
        _count = (_count, 1 - _count / _length)  # len(dset)
        _length = (_length, _length / self.bpod_ntrials)

        # Add the values to output dict
        for k, v in zip(names, (_length, _count)):
            out[k] = v

        return out

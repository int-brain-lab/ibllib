import json
import logging

import numpy as np

import ibllib.qc.bpodqc_metrics as bpodqc
import ibllib.qc.oneqc_metrics as oneqc
from ibllib.qc.bpodqc_extractors import load_bpod_data
from oneibl.one import ONE

log = logging.getLogger('ibllib')

# one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")
one = ONE()

# eid, det = random_ephys_session()

# eid = "4153bd83-2168-4bd4-a15c-f7e82f3f73fb"
# det = one.get_details(eid)
# details = one.get_details(eid, full=True)


def build_extended_qc_frame(eid, data=None):
    if data is None:
        data = load_bpod_data(eid)

    # Get bpod and one qc frames
    extended_qc = {}
    log.info(f"Session {eid}: Running QC on ONE DatasetTypes...")
    one_frame = oneqc.get_oneqc_metrics_frame(eid, data=data, apply_criteria=True)
    log.info(f"Session {eid}: Running QC on Bpod data...")
    bpod_frame = bpodqc.get_bpodqc_metrics_frame(eid, data=data, apply_criteria=True)
    # Make average bool pass
    # def average_frame(frame):
    #     return {k: np.nanmean(v) for k, v in frame.items()}
    average_bpod_frame = (lambda frame: {k: np.nanmean(v) for k, v in frame.items()})(bpod_frame)
    # aggregate them
    extended_qc.update(one_frame)
    extended_qc.update(average_bpod_frame)
    return extended_qc


def write_extended_qc(eid: str, eqc_data: dict) -> dict:
    """write_extended_qc
    Write data to extended_qc WILL NOT CHECK IF DATA EXISTS
    NOTE: Destructive write!

    :param eid: Valid uuid sting for a session
    :type eid: str
    :param eqc_data: dict to upload
    :type eqc_data: dict
    :return: uploaded dict
    :rtype: dict
    """
    # Prepare data to patch
    patch_dict = {"extended_qc": eqc_data}
    # Upload new extended_qc to session
    session_details = one.alyx.rest("sessions", "partial_update", id=eid, data=patch_dict)
    log.info(f"Written to extended_qc json field in session {eid}")
    return session_details["extended_qc"]


def update_extended_qc(eid: str, eqc_data: dict) -> dict:
    """update_extended_qc_on_session
    Non destructive update of extended_qc field for session "eid"
    Will update the extended_qc field of the session with the eqc_data dict inputted
    If eqc_data has fields with the same name of existing fields it will squash the old
    values (uses the dict.update() method)

    :param eid: Valid uuid sting for a session
    :type eid: str
    :param data: dict with etended_qc frame to be uploaded
    :type data: dict
    :return: new patched extended_qc dict
    :rtype: dict
    """
    # Load current extended_qc field
    current = read_extended_qc(eid)
    # Patch current dict of extended_qc with new data
    if current is not None:
        # NOTE: Check here if we want to enforce a specific structure
        current.update(eqc_data)
    extended_qc = write_extended_qc(eid, eqc_data)
    return extended_qc


def read_extended_qc(eid: str) -> dict:
    """ Query the extended_qc field of session eid"""
    eqc = one.alyx.rest("sessions", "read", eid)["extended_qc"]
    return eqc


def delete_extended_qc(eid: str) -> None:
    _ = one.alyx.rest("sessions", "partial_update", id=eid, data={"extended_qc": None})


def remove_extended_qc_key(eid: str, key: str) -> dict:
    current = read_extended_qc(eid)
    if current is None:
        return
    if current.get(key, None) is None:
        log.warning(f"{key}: Key not found in extended_qc session field")
        return current
    log.info(f"Removing {key}")
    current.pop(key)
    extended_qc = write_extended_qc(eid, current)
    return extended_qc


def build_and_upload_extended_qc(eid, data=None):
    if data is None:
        data = load_bpod_data(eid)

    eqc_data = build_extended_qc_frame(eid, data=data)
    new_eqc_data = update_extended_qc(eid, eqc_data)
    return new_eqc_data


class AlyxJsonField(object):
    def __init__(self, eid, endpoint, field):
        self.uuid: str = None
        self.endpoint: str = None
        self.field: str = None
        self.data: dict = None

    def _validate_attribs(self):
        is_uuid_string(self.eid)
        self.endpoint in one.alyx.rest()

    def write(self, data) -> dict:
        """write_extended_qc
        Write data to WILL NOT CHECK IF DATA EXISTS
        NOTE: Destructive write!

        :param eid: Valid uuid sting for a given endpoint
        :type eid: str
        :param eqc_data: dict to upload
        :type eqc_data: dict
        :return: uploaded dict
        :rtype: dict
        """
        # Prepare data to patch
        patch_dict = {self.field: data}
        # Upload new extended_qc to session
        ret = one.alyx.rest(self.endpoint, "partial_update", id=self.eid, data=patch_dict)
        return ret[self.field]

    def json_field_update(self, endpoint, uuid, filed_name, data) -> dict:
        """update_extended_qc_on_session
        Non destructive update of extended_qc field for session "eid"
        Will update the extended_qc field of the session with the eqc_data dict inputted
        If eqc_data has fields with the same name of existing fields it will squash the old
        values (uses the dict.update() method)

        :param eid: Valid uuid sting for a session
        :type eid: str
        :param data: dict with etended_qc frame to be uploaded
        :type data: dict
        :return: new patched extended_qc dict
        :rtype: dict
        """
        # Load current extended_qc field
        current = self.read()
        # Patch current dict of extended_qc with new data
        if current is not None:
            current.update(self.data)
        written_field = self.write()
        return written_field

    def read(self) -> dict:
        """ Query the extended_qc field of session eid"""
        read_field = one.alyx.rest(self.endpoint, "read", self.eid)[self.field]
        return read_field

    def delete(self) -> None:
        _ = one.alyx.rest(self.endpoint, "partial_update", id=self.eid, data={self.field: None})

    def remove_key_from_field(self, key) -> dict:
        current = self.read()
        if current is None:
            return
        if current.get(key, None) is None:
            print(f"{key}: Key not found in endpoint {self.endpoint} field {self.field}")
            return current
        print(f"Removing {key}")
        current.pop(key)
        written = self.write(current)
        return written


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

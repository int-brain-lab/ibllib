import numpy as np
import uuid


def uuid2np(eids_uuid):
    return np.asfortranarray(
        np.array([np.frombuffer(eid.bytes, dtype=np.int64) for eid in eids_uuid]))


def str2np(eids_str):
    return uuid2np([uuid.UUID(eid) for eid in eids_str])


def np2uuid(eids_np):
    return [uuid.UUID(bytes=npu.tobytes()) for npu in eids_np]


def np2str(eids_np):
    return [str(u) for u in np2uuid(eids_np)]

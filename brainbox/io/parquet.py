import uuid
import numpy as np

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

from brainbox.core import Bunch


def load(file):
    """
    Loads parquet file into pandas dataframe
    :param file:
    :return:
    """
    return pq.read_table(file).to_pandas()


def save(file, table):
    """
    Save pandas dataframe to parquet
    :param file:
    :param table:
    :return:
    """
    pq.write_table(pa.Table.from_pandas(table), file)


def uuid2np(eids_uuid):
    return np.asfortranarray(
        np.array([np.frombuffer(eid.bytes, dtype=np.int64) for eid in eids_uuid]))


def str2np(eids_str):
    if isinstance(eids_str, str):
        eids_str = [eids_str]
    return uuid2np([uuid.UUID(eid) for eid in eids_str])


def np2uuid(eids_np):
    if isinstance(eids_np, pd.DataFrame) | isinstance(eids_np, pd.Series):
        eids_np = eids_np.to_numpy()
    if eids_np.ndim >= 2:
        return [uuid.UUID(bytes=npu.tobytes()) for npu in eids_np]
    else:
        return uuid.UUID(bytes=eids_np.tobytes())


def np2str(eids_np):
    eids = np2uuid(eids_np)
    eids = str(eids) if isinstance(eids, uuid.UUID) else [str(u) for u in np2uuid(eids_np)]
    return eids


def rec2col(rec, join=None, include=None, exclude=None, uuid_fields=None, types=None):
    """
    Change a record list (usually from a REST API endpoint) to a column based dictionary
    (pandas dataframe).
    :param rec (list): list of dictionaries with consistent keys
    :param join (dictionary): dictionary of scalar keys that will be replicated over the full
    array (join operation)
    :param include: list of strings representing dictionary keys: if specified will only include
    the keys specified here
    :param exclude: list of strings representing dictionary keys: if specified will exclude the
    keys specified here
    :param uuid_fields: if the field is an UUID, will split it into 2 distinct int64 columns for
    efficient lookups and intersections
    :param types: for a given key, will force the type; example: types = {'file_size': np.double}
    :return: a Bunch
    """
    if isinstance(rec, dict):
        rec = [rec]
    if len(rec) == 0:
        return Bunch()
    if include is None:
        include = rec[0].keys() if isinstance(rec, list) else rec.keys()
    if exclude is None:
        exclude = []
    if uuid_fields is None:
        uuid_fields = []
    if join is None:
        join = {}

    # first loop over the records and create each columns as a numpy array
    nrecs = len(rec)
    col = {}
    keys = [k for k in rec[0] if k in include and k not in exclude]
    for key in keys:
        if key in uuid_fields:
            npuuid = str2np(np.array([c[key] for c in rec]))
            col[f"{key}_0"] = npuuid[:, 0]
            col[f"{key}_1"] = npuuid[:, 1]
        elif types and key in types:
            col[key] = np.array([c[key] for c in rec]).astype(types[key])
        else:
            col[key] = np.array([c[key] for c in rec])

    # then perform the joins if any
    for key in join:
        if key in uuid_fields:
            npuuid = str2np([join[key]])
            col[f"{key}_0"] = np.tile(npuuid[0, 0], (nrecs,))
            col[f"{key}_1"] = np.tile(npuuid[0, 1], (nrecs,))
        else:
            col[key] = np.tile(np.array(join[key]), (nrecs,))

    return Bunch(col)

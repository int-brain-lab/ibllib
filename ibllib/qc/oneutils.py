from functools import partial, wraps
from pathlib import Path

import numpy as np

from alf.io import is_details_dict, is_session_path, is_uuid_string
from oneibl.one import ONE

one = ONE()


# Decorators
def _dl_raw_behavior(session_path, full=False, dry=False, force=False):
    """ Helper function to download raw behavioral data from session_path based functions
    """
    dsts = [x for x in one.list(None, "dataset_types") if "_iblrig_" in x]
    min_dsts = [
        "_iblrig_taskData.raw",
        "_iblrig_taskSettings.raw",
        "_iblrig_stimPositionScreen.raw",
        "_iblrig_syncSquareUpdate.raw",
        "_iblrig_encoderEvents.raw",
        "_iblrig_encoderPositions.raw",
        "_iblrig_encoderTrialInfo.raw",
        "_iblrig_ambientSensorData.raw",
    ]
    session_path = Path(session_path)
    eid = one.eid_from_path(session_path)

    if eid is None:
        return
    if full:
        one.load(eid, download_only=True, dry_run=dry, dataset_types=dsts, clobber=force)
    elif not full:
        one.load(eid, download_only=True, dry_run=dry, dataset_types=min_dsts, clobber=force)


def uuid_to_path(func=None, dl=False, full=False, dry=False, force=False):
    """ Check if first argument of func is eID, if valid return path with oprional download
    """
    if func is None:
        return partial(uuid_to_path, dl=dl, full=full, dry=dry, force=force)

    @wraps(func)
    def wrapper(eid, *args, **kwargs):
        if eid is None:
            print("Input eid or session_path is None")
            return
        # Check if first arg is path or eid
        if is_uuid_string(str(eid)):
            session_path = one.path_from_eid(eid)
        else:
            session_path = Path(eid)
        if dl:
            _dl_raw_behavior(session_path, full=full, dry=dry, force=force)

        return func(session_path, *args, **kwargs)

    return wrapper


# Other utils
def _one_load_session_delays_between_events(eid, dstype1, dstype2):
    """ Returns difference between times of 2 different dataset types
    Func is called with eid and dstypes in temporal order, returns delay between
    event1 and event 2, i.e. event_time2 - event_time1
    """
    event_times1, event_times2 = one.load(eid, dataset_types=[dstype1, dstype2])
    if all(np.isnan(event_times1)) or all(np.isnan(event_times2)):
        print(
            f"{eid}\nall {dstype1} nan: {all(np.isnan(event_times1))}",
            f"\nall {dstype2} nan: {all(np.isnan(event_times2))}",
        )
        return
    delay_between_events = event_times2 - event_times1
    return delay_between_events


def _to_eid(invar):
    """ get eid from: details, path, or lists of details or paths
    """
    outvar = []
    if isinstance(invar, list) or isinstance(invar, tuple):
        for i in invar:
            outvar.append(_to_eid(i))
        return outvar
    elif isinstance(invar, dict) and is_details_dict(invar):
        return invar["url"][-36:]
    elif isinstance(invar, str) and is_session_path(invar):
        return one.eid_from_path(invar)
    elif isinstance(invar, str) and is_uuid_string(invar):
        return invar
    else:
        print("Unknown input type: please input a valid path or details object")
        return


def search_lab_ephys_sessions(
    lab: str, dstypes: list = [], nlatest: int = 3, det: bool = True, check_download: bool = False
):
    ephys_sessions0, session_details0 = one.search(
        task_protocol="_iblrig_tasks_ephysChoiceWorld6.4.0",
        dataset_types=dstypes,
        limit=1000,
        details=True,
        lab=lab,
    )
    ephys_sessions1, session_details1 = one.search(
        task_protocol="_iblrig_tasks_ephysChoiceWorld6.2.5",
        dataset_types=dstypes,
        limit=1000,
        details=True,
        lab=lab,
    )
    ephys_sessions = list(ephys_sessions0) + list(ephys_sessions1)
    session_details = list(session_details0) + list(session_details1)
    # Check if you found anything
    if ephys_sessions == []:
        print(f"No sessions found for {lab}")
        return
    if not check_download:
        return (
            ephys_sessions[:nlatest],
            session_details[:nlatest] if det else ephys_sessions[:nlatest],
        )
    print(f"Processing {lab}")
    out_sessions = []
    out_details = []
    for esess, edets in zip(ephys_sessions, session_details):
        dstypes_data = one.load(esess, dataset_types=dstypes)
        # Check if dstypes have all NaNs
        skip_esess = False
        for dsname, dsdata in zip(dstypes, dstypes_data):
            if "raw" in dsname:
                continue
            if np.all(np.isnan(dsdata)):
                print(f"Skipping {esess}, one or more dstypes are all NaNs")
                skip_esess = True
        if skip_esess:
            continue
        # Check if all dstypes have the same length
        if not all(len(x) == len(dstypes_data[0]) for x in dstypes_data):
            print(f"Skipping {esess}, one or more dstypes have different lengths")
            continue
        out_sessions.append(esess)
        out_details.append(edets)
        if len(out_details) == nlatest:
            break
    return out_sessions, out_details if det else out_sessions


def random_ephys_session(lab=None, complete=False):
    if lab is None:
        lab = random_lab()
    if complete:
        dstypes = one.list()
    else:
        dstypes = []
    sessions = search_lab_ephys_sessions(lab, dstypes=dstypes, nlatest=None)
    if sessions is None:
        return
    eids, dets = sessions
    out = np.random.choice(dets)
    print(out)
    return _to_eid(out), out


def random_lab():
    labs = one.list(None, "lab")
    return np.random.choice(labs)


def get_details(eid, full=False):
    """ Returns details of eid like from one.search, optional return full
    session details.
    """
    # TODO: integrate into ONE, check for valid eid, make test
    dets = one.alyx.rest("sessions", "read", eid)
    if full:
        return dets
    det_fields = ["subject", "start_time", "number", "lab", "project",
                  "url", "task_protocol", "local_path"]
    return {k: v for k, v in dets.items() if k in det_fields}


if __name__ == "__main__":
    sp = "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2240/2020-01-22/001"
    eid = "259927fd-7563-4b03-bc5d-17b4d0fa7a55"
    det = {
        "subject": "ZM_2240",
        "start_time": "2020-01-22T10:50:59",
        "number": 1,
        "lab": "mainenlab",
        "url": "https://alyx.internationalbrainlab.org/sessions/259927fd-7563-4b03-bc5d-17b4d0fa7a55",
        "task_protocol": "_iblrig_tasks_ephysChoiceWorld6.2.5",
        "local_path": "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2240/2020-01-22/001",
    }
    lab = "mainenlab"
    # Test random dataset
    print(is_uuid_string(random_ephys_session(lab, complete=False)[0]))
    print(is_details_dict(random_ephys_session(lab, complete=False)[1]))
    print(random_ephys_session(lab, complete=True) is None)
    print(random_ephys_session("sakjdhka", complete=False) is None)
    # Test _to_eid
    # All == eid
    print(_to_eid(eid) == eid)
    print(_to_eid([eid, eid]) == [eid, eid])
    print(_to_eid((eid, eid)) == [eid, eid])
    print(_to_eid(sp) == eid)
    print(_to_eid([sp, sp]) == [eid, eid])
    print(_to_eid((sp, sp)) == [eid, eid])
    print(_to_eid(det) == eid)
    print(_to_eid([det, det]) == [eid, eid])
    print(_to_eid((det, det)) == [eid, eid])
    # All None
    print(_to_eid(123) is None)
    print(_to_eid("sda") is None)
    print(_to_eid({1: 2}) is None)
    print(_to_eid([1, 2]) == [None, None])
    print(_to_eid((1, 2)) == [None, None])

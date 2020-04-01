import numpy as np
from oneibl.one import ONE
from alf.io import is_details_dict, is_session_path, is_uuid_string

one = ONE()


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
        return ephys_sessions, session_details if det else ephys_sessions
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


def random_ephys_session(lab, complete=False):
    if complete:
        dstypes = one.list()
    else:
        dstypes = []
    sessions = search_lab_ephys_sessions(lab, dstypes=dstypes)
    if sessions is None:
        return
    eids, dets = sessions
    out = np.random.choice(dets)
    return _to_eid(out), out


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
    print(random_ephys_session(lab, complete=False))
    print(random_ephys_session(lab, complete=True))

    print(_to_eid(eid))
    print(_to_eid([eid, eid]))
    print(_to_eid((eid, eid)))
    print(_to_eid(sp))
    print(_to_eid([sp, sp]))
    print(_to_eid((sp, sp)))
    print(_to_eid(det))
    print(_to_eid([det, det]))
    print(_to_eid((det, det)))
    print(_to_eid(123))
    print(_to_eid('sda'))
    print(_to_eid({1:2}))
    print(_to_eid([1,2]))
    print(_to_eid((1,2)))
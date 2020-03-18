import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ibllib.ephys.bpodqc as bpodqc
from ibllib.ephys.ephysqc import _qc_from_path
from oneibl.one import ONE

# plt.ion()

one = ONE(
    username="niccolo",
    password="ItTakesBrains!",
    base_url="https://alyx.internationalbrainlab.org",
)


def _one_load_session_deleay_between_events(eid, dstype1, dstype2):
    """ Returns difference between times of 2 different dataset types
    Func is called with eid and dstypes in temporal order, returns delay between
    event1 and event 2, i.e. event_time2 - event_time1
    """
    event_times1, event_times2 = one.load(eid, dataset_types=[dstype1, dstype2])
    if all(np.isnan(event_times1)) or all(np.isnan(event_times2)):
        print(
            f'{eid}\nall {dstype1} nan: {all(np.isnan(event_times1))}',
            f'\nall {dstype2} nan: {all(np.isnan(event_times2))}'
        )
        return
    delay_between_events = event_times2 - event_times1
    return delay_between_events


def load_session_stimon_gocue_delays(eid):
    return _one_load_session_deleay_between_events(
        eid, 'trials.stimOn_times', 'trials.goCue_times'
    )


def load_session_response_feddback_delays(eid):
    return _one_load_session_deleay_between_events(
        eid, 'trials.response_times', 'trials.feedback_times'
    )


def load_session_response_stimFreeze_delays(eid):
    response = one.load(eid, dataset_types=['trials.response_times'])[0]
    _, _, stimFreeze = bpodqc.get_stimOnOffFreeze_times_from_BNC1(eid)
    if len(response) != len(stimFreeze):
        session_path = one.path_from_eid(eid)
        response = bpodqc.get_response_times(session_path, save=False)
    assert len(response) == len(stimFreeze)
    return stimFreeze - response


def search_lab_ephys_sessions(lab: str, dstypes: list, nlatest: int = 3, det: bool = True):
    ephys_sessions0, session_details0 = one.search(
        task_protocol='_iblrig_tasks_ephysChoiceWorld6.4.0',
        dataset_types=dstypes,
        limit=1000, details=True, lab=lab
    )
    ephys_sessions1, session_details1 = one.search(
        task_protocol='_iblrig_tasks_ephysChoiceWorld6.2.5',
        dataset_types=dstypes,
        limit=1000, details=True, lab=lab
    )
    ephys_sessions = list(ephys_sessions0) + list(ephys_sessions1)
    session_details = list(session_details0) + list(session_details1)
    print(f"Processing {lab}")
    # Check if you found anything
    if ephys_sessions == []:
        print(f'No sessions found for {lab}')
        return
    out_sessions = []
    out_details = []
    for esess, edets in zip(ephys_sessions, session_details):
        dstypes_data = one.load(esess, dataset_types=dstypes)
        # Check if dstypes have all NaNs
        skip_esess = False
        for dsdata in dstypes_data:
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


def _load_df_from_details(details=None, func=None):
    """
    Applies a session level func(eid) from session details dict from Alyx
    """
    if details or func is None:
        return
    if isinstance(details, dict):
        details = [details]
    data = []
    labels = []
    for i, det in enumerate(details):
        eid = det['url'][-36:]
        data.append(load_session_stimon_gocue_delays(eid))
        labels.append(det['lab'] + str(i))

    df = pd.DataFrame(data).transpose()
    df.columns = labels

    return df


def process_session_stimon_gocue_delays(details):
    if details is None:
        return
    df = _load_df_from_details(
        details, func=load_session_stimon_gocue_delays
    )
    return df


# Plots
def plot_session_stimon_gocue_delays(details: list, ax=None, describe=False):
    if details is None:
        return
    if ax is None:
        f, ax = plt.subplots()

    # Load and process data
    df = process_session_stimon_gocue_delays(details)
    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title("goCue - stimOn")
    p.set_xlabel('Seconds (s)')
    p.set(xscale="symlog")


def process_session_response_feedback_delays(details):
    if details is None:
        return
    df = _load_df_from_details(
        details, func=load_session_response_feddback_delays
    )
    return df


def plot_session_response_feedback_delays(details: list, ax=None, describe=False):
    if details is None:
        return
    if ax is None:
        f, ax = plt.subplots()
    # Load and process data
    df = process_session_stimon_gocue_delays(details)
    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title("feedback - response")
    p.set_xlabel('Seconds (s)')
    p.set(xscale="symlog")


if __name__ == "__main__":
    eid = '0deb75fb-9088-42d9-b744-012fb8fc4afb'
    eid = 'af74b29d-a671-4c22-a5e8-1e3d27e362f3'
    # lab = 'zadorlab'
    # ed = search_lab_ephys_sessions(lab, ['trials.stimOn_times', 'trials.goCue_times'])

    # f, ax = plt.subplots()
    labs = one.list(None, 'lab')
    eids = []
    details = []
    for lab in labs:
        ed = search_lab_ephys_sessions(lab, ['trials.stimOn_times', 'trials.goCue_times'])
        if ed is not None:
            eids.extend(ed[0])
            details.extend(ed[1])
    plot_session_stimon_gocue_delays(details)
    plt.show()
    #  get_session_stimon_gocue_delays(eid)
    # get_response_feddback_delays(eid)
    # get_response_stimFreeze_delays(eid)  # FIXME:Have to fix timescales!!!!
    # bpod = bpodqc.load_bpod_data(eid)
    # response = one.load(eid, dataset_types=['trials.response_times'])[0]
    # plt.plot(response, '-o', label='response_one')
    # plt.plot(bpod['response_times'], '-o', label='response_bpod')
    # plt.show()
    # plt.legend(loc='best')

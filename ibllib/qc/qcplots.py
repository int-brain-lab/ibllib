import json
from functools import wraps
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ibllib.qc.bpodqc_extractors as bpodqc
import ibllib.qc.bpodqc_metrics as qcmetrics
import ibllib.qc.oneutils as oneutils
from oneibl.one import ONE
from alf.io import is_details_dict

plt.ion()


one = ONE(printout=False)


def _load_df_from_details(details=None, func=None):
    """
    Applies a session level loader_func(eid) from session details dict from Alyx
    Example:
    df = _load_df_from_details(details, func=load_stimOff_itiIn_delays)
    """
    if details is None or func is None:
        print("One or more required inputs are None.")
        return
    if is_details_dict(details):
        details = [details]
    data = []
    labels = []
    for i, det in enumerate(details):
        eid = _to_eid(det)
        data.append(func(eid))
        labels.append(det["lab"] + str(i))

    df = pd.DataFrame(data).transpose()
    df.columns = labels

    return df


def plot_random_session_metrics(lab):
    if lab is None:
        print("Please input a lab")
        return
    eid, det = oneutils.random_ephys_session(lab)
    data = bpodqc.extract_bpod_trial_data(eid, fpga_time=False)
    metrics = qcmetrics.get_qcmetrics_frame(eid, data=data)


def get_last_from_all_labs(n=3):
    labs = one.list(None, 'lab')
    eids = []
    details = []
    for lab in labs:
        ed = oneutils.search_lab_ephys_sessions(lab, dstypes=['_iblrig_taskData.raw'], nlatest=n)
        if not ed:
            continue
        eids.extend(ed[0])
        details.extend(ed[1])
    return eids, details


def get_metrics_from_list(eids):
    outdata = []
    outmetrics = []
    outcriteria = []
    for eid in eids:
        data = bpodqc.extract_bpod_trial_data(eid, fpga_time=False)
        metrics = qcmetrics.get_qcmetrics_frame(eid, data=data)
        criteria = qcmetrics.get_qccriteria_frame(eid, data=data)
        outdata.append(data)
        outmetrics.append(metrics)
        outcriteria.append(criteria)
    return (outdata, outmetrics, outcriteria)


def rearrange_metrics(metrics):
    out_dict = {k: [] for k in metrics[0].keys()}
    for k in out_dict:
        for met in metrics:
            out_dict[k].append(met[k])

    return pd.DataFrame.from_dict(out_dict)


def plot_metrics(df, details, save_path=None):
    ylabels = []
    for d in details:
        ylabels.append('/'.join(d['local_path'].split('/')[-3:]))

    for k in df.columns:
        a4_dims = (11.7, 8.27)
        fig, ax = plt.subplots(figsize=a4_dims)
        p = sns.boxplot(ax=ax, data=df[k], orient="h")
        mng = plt.get_current_fig_manager()
        p.set_title(k)
        p.set_xlabel("Seconds (s)")
        p.set(xscale='log')
        p.set_yticks(range(len(ylabels)))
        p.set_ylim((len(ylabels), -1))
        p.set_yticklabels(ylabels)
        mng.window.showMaximized()
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.exists():
                print(f'Folder {save_path} does not exist, not saving...')
                continue
            p.figure.savefig(save_path.joinpath(f"{k}.png"))


def plot_criteria(criteria, details, save_path=None):
    titles = []
    for d in details:
        titles.append('/'.join(d['local_path'].split('/')[-3:]))

    for c, t in zip(criteria, titles):
        dfc = pd.DataFrame(c, index=[0])
        a4_dims = (11.7, 8.27)
        fig, ax = plt.subplots(figsize=a4_dims)
        mng = plt.get_current_fig_manager()
        p = sns.barplot(ax=ax, data=dfc, orient='h')
        p.set_title(t)
        p.set_xlabel("Proportion of trials that pass criteria")
        mng.window.showMaximized()
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.exists():
                print(f'Folder {save_path} does not exist, not saving...')
                continue
            p.figure.savefig(save_path.joinpath(f"{t.replace('/', '-')}.png"))


def boxplots_from_df(
    df, ax=None, describe=False, title="", xlabel="Seconds (s)", xscale="symlog",
):
    if ax is None:
        f, ax = plt.subplots()

    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title(title)
    p.set_xlabel(xlabel)
    p.set(xscale=xscale)


def boxplot_metrics(eid, qcmetrics_frame=None):
    if qcmetrics_frame is None:
        qcmetrics_frame = get_qcmetrics_frame(eid)
    df = pd.DataFrame.from_dict({k: qcmetrics_frame[k] for k in qcmetrics_frame if "delays" in k})
    boxplots_from_df(df, describe=True)




@uuid_to_path(dl=True)
def plot_session_trigger_response_diffs(session_path, ax=None):
    trigger_diffs = get_session_trigger_response_delays(session_path)

    sett = raw.load_settings(session_path)
    eid = one.eid_from_path(session_path)
    if ax is None:
        f, ax = plt.subplots()
    tit = f"{sett['SESSION_NAME']}: {eid}"
    ax.title.set_text(tit)
    ax.hist(trigger_diffs["goCue"], alpha=0.5, bins=50, label="goCue_diff")
    ax.hist(trigger_diffs["errorCue"], alpha=0.5, bins=50, label="errorCue_diff")
    ax.hist(trigger_diffs["stimOn"], alpha=0.5, bins=50, label="stimOn_diff")
    ax.hist(trigger_diffs["stimOff"], alpha=0.5, bins=50, label="stimOff_diff")
    ax.hist(trigger_diffs["stimFreeze"], alpha=0.5, bins=50, label="stimFreeze_diff")
    ax.legend(loc="best")


@uuid_to_path(dl=True)
def get_session_trigger_response_delays(session_path):
    bpod = extract_bpod_trial_table(session_path)
    # get diff from triggers to detected events
    goCue_diff = np.abs(bpod["goCueTrigger_times"] - bpod["goCue_times"])
    errorCue_diff = np.abs(bpod["errorCueTrigger_times"] - bpod["errorCue_times"])
    stimOn_diff = np.abs(bpod["stimOnTrigger_times"] - bpod["stimOn_times"])
    stimOff_diff = np.abs(bpod["stimOffTrigger_times"] - bpod["stimOff_times"])
    stimFreeze_diff = np.abs(bpod["stimFreezeTrigger_times"] - bpod["stimFreeze_times"])

    return {
        "goCue": goCue_diff,
        "errorCue": errorCue_diff,
        "stimOn": stimOn_diff,
        "stimOff": stimOff_diff,
        "stimFreeze": stimFreeze_diff,
    }


def _describe_trigger_diffs(trigger_diffs):
    print(trigger_diffs.describe())
    for k in trigger_diffs:
        print(k, "nancount:", sum(np.isnan(trigger_diffs[k])))

    return trigger_diffs


@uuid_to_path(dl=True)
def describe_sesion_trigger_response_diffs(session_path):
    trigger_diffs = get_session_trigger_response_delays(session_path)
    return _describe_trigger_diffs(trigger_diffs)


def get_trigger_response_diffs(eid_or_path_list):
    trigger_diffs = {
        "goCue": np.array([]),
        "errorCue": np.array([]),
        "stimOn": np.array([]),
        "stimOff": np.array([]),
        "stimFreeze": np.array([]),
    }
    for sess in eid_or_path_list:
        td = get_session_trigger_response_delays(sess)
        for k in trigger_diffs:
            trigger_diffs[k] = np.append(trigger_diffs[k], td[k])

    df = pd.DataFrame.from_dict(trigger_diffs)

    return df


def describe_trigger_response_diffs(eid_or_path_list):
    trigger_diffs = get_trigger_response_diffs(eid_or_path_list)
    return _describe_trigger_diffs(trigger_diffs)


def describe_lab_trigger_response_delays(labname):
    eids, dets = one.search(
        task_protocol="_iblrig_tasks_ephysChoiceWorld6.2.5",
        lab=labname,
        dataset_types=["_iblrig_taskData.raw", "_iblrig_taskSettings.raw"],
        details=True,
    )
    trigger_diffs = get_trigger_response_diffs(eids)
    return _describe_trigger_diffs(trigger_diffs)


if __name__ == "__main__":
    import pickle
    # eids, details = get_last_from_all_labs()
    # data, metrics, criteria = get_metrics_from_list(eids)
    path = '/home/nico/Projects/IBL/scratch/plots/data'
    fpeids = path + '/eids.p'
    fpdetails = path + '/details.p'
    fpdata = path + '/data.p'
    fpmetrics = path + '/metrics.p'
    fpcriteria = path + '/criteria.p'
    # Save
    # pickle.dump(eids, open(fpeids, 'wb'))
    # pickle.dump(details, open(fpdetails, 'wb'))
    # pickle.dump(data, open(fpdata, 'wb'))
    # pickle.dump(metrics, open(fpmetrics, 'wb'))
    # pickle.dump(criteria, open(fpcriteria, 'wb'))

    # Load
    eids = pickle.load(open(fpeids, 'rb'))
    data = pickle.load(open(fpdata, 'rb'))
    details = pickle.load(open(fpdetails, 'rb'))
    metrics = pickle.load(open(fpmetrics, 'rb'))
    criteria = pickle.load(open(fpcriteria, 'rb'))

    df = rearrange_metrics(metrics)
    # plot_metrics(df, details)
    # eid, det = oneutils.random_ephys_session("churchlandlab")
    # data = bpodqc.extract_bpod_trial_table(eid, fpga_time=False)
    # metrics = qcmetrics.get_qcmetrics_frame(eid, data=data)
    # criteria = qcmetrics.get_qccriteria_frame(eid, data=data)
import json
from functools import wraps
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ibllib.ephys.bpodqc as bpodqc
import ibllib.ephys.qcmetrics as qcmetrics
import ibllib.ephys.oneutils as oneutils
from ibllib.ephys.oneutils import search_lab_ephys_sessions, _to_eid, random_ephys_session
from oneibl.one import ONE
from alf.io import is_details_dict

plt.ion()


one = ONE()


def plot_random_session_metrics(lab):
    if lab is None:
        print("Please input a lab")
        return
    eid, det = random_ephys_session(lab)
    data = bpodqc.load_bpod_data(eid, fpga_time=False)
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
        data = bpodqc.load_bpod_data(eid, fpga_time=False)
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
    # eid, det = random_ephys_session("churchlandlab")
    # data = bpodqc.load_bpod_data(eid, fpga_time=False)
    # metrics = qcmetrics.get_qcmetrics_frame(eid, data=data)
    # criteria = qcmetrics.get_qccriteria_frame(eid, data=data)
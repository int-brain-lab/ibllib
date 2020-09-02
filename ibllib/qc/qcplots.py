from collections import Counter
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ibllib.qc.task_metrics import TaskQC
import ibllib.qc.oneutils as oneutils
from oneibl.one import ONE


def plot_results(qc_obj, save_path=None):
    if not isinstance(qc_obj, TaskQC):
        raise ValueError('Input must be TaskQC object')

    if not qc_obj.passed:
        qc_obj.compute()

    outcome, results, outcomes = qc_obj.compute_session_status()

    n_trials = qc_obj.extractor.data['intervals_0'].size
    d = qc_obj.one.get_details(qc_obj.eid)
    ref = f"{datetime.fromisoformat(d['start_time']).date()}_{d['number']:d}_{d['subject']}"

    # Sort into each category
    counts = Counter(outcomes.values())
    fig = plt.Figure()
    plt.bar(range(len(counts)), counts.values(), align='center')
    fig.suptitle = ref

    a4_dims = (11.7, 8.27)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=a4_dims)
    fig.suptitle = ref

    # Plot failed trial level metrics
    def get_trial_level_failed(d):
        new_dict = {k[6:]: v for k, v in d.items() if outcomes[k] == 'FAIL' and len(v) == n_trials}
        return pd.DataFrame.from_dict(new_dict)
    sns.boxplot(data=get_trial_level_failed(qc_obj.metrics), orient='h', ax=ax0)
    ax0.set_yticklabels(ax0.get_yticklabels(), rotation=30)
    ax0.set(xscale='symlog', title='metrics (failed)')

    # Plot failed trial level metrics
    sns.barplot(data=get_trial_level_failed(qc_obj.passed), orient='h', ax=ax1)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=30)
    ax1.set_title('counts')

    if save_path is not None:
        save_path = Path(save_path)

        if save_path.is_dir() and not save_path.exists():
            print(f"Folder {save_path} does not exist, not saving...")
        elif save_path.is_dir():
            fig.savefig(save_path.joinpath(f"{ref}_qc.png"))
        else:
            fig.savefig(save_path)


if __name__ == "__main__":
    one = ONE(printout=False)
    # Load data
    eid, det = oneutils.random_ephys_session()
    # Run QC
    qc = TaskQC(eid, one=one)
    plot_results(qc)
    plt.show()

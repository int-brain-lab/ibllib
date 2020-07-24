import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ibllib.qc.bpodqc_metrics import BpodQC
import ibllib.qc.oneutils as oneutils
from oneibl.one import ONE


def boxplot_metrics(
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


def barplot_passed(
    df,
    ax=None,
    describe=False,
    title=None,
    xlabel="Proportion of trials that pass criteria",
    save_path=None,
):
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    p = sns.barplot(ax=ax, data=df, orient="h")
    p.set_title(title)
    p.set_xlabel(xlabel)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.exists() or title is None:
            print(f"Folder {save_path} does not exist, not saving...")
        p.figure.savefig(save_path.joinpath(f"{title.replace('/', '-')}.png"))


if __name__ == "__main__":
    one = ONE(printout=False)
    # Load data
    eid, det = oneutils.random_ephys_session()
    # Run QC
    bpodqc = BpodQC(eid, one=one, ensure_data=True, lazy=False)

    session_name = "/".join(bpodqc.session_path.parts[-3:])

    df_metric = pd.DataFrame.from_dict(bpodqc.metrics)
    df_passed = pd.DataFrame.from_dict(bpodqc.passed)
    boxplot_metrics(df_metric, title=session_name)
    barplot_passed(df_passed, title=session_name)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, September 7th 2020, 11:51:17 am
import alf.io
from oneibl.one import ONE
from pathlib import Path
import numpy as np
import json
import ibllib.io.extractors.passive as passive
from ibllib.io.extractors import ephys_fpga
import ibllib.io.raw_data_loaders as rawio
from ibllib.qc.oneutils import random_ephys_session

# hardcoded var
FRAME_FS = 60  # Sampling freq of the ipad screen, in Hertz
FS_FPGA = 30000  # Sampling freq of the neural recording system screen, in Hertz
NVALVE = 40  # number of expected valve clicks
NGABOR = 20 + 20 * 4 * 2  # number of expected Gabor patches
NTONES = 40
NNOISES = 40
DEBUG_PLOTS = False

# load data
one = ONE()
dataset_types = [
    "_spikeglx_sync.times",
    "_spikeglx_sync.channels",
    "_spikeglx_sync.polarities",
    "_iblrig_RFMapStim.raw",
    "_iblrig_stimPositionScreen.raw",
    "_iblrig_syncSquareUpdate.raw",
    "ephysData.raw.meta",
    "_iblrig_taskSettings.raw",
    "_iblrig_taskData.raw",
]

eid = one.search(subject="CSH_ZAD_022", date_range="2020-05-24", number=1)[0]
eid = "01864d6f-31e8-49c9-aadd-2e5021ea0ee7"  # not working
eid = "fff7d745-bbce-4756-a690-3431e2f3d108"  # number of expected spacers wrong
eid = "193fe7a8-4eb5-4f3e-815a-0c45864ddd77"  # OK
# eid, det = random_ephys_session()

local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)

session_path = alf.io.get_session_path(local_paths[0])

# load session fixtures
settings = rawio.load_settings(session_path)
ses_nb = settings["SESSION_ORDER"][settings["SESSION_IDX"]]
path_fixtures = Path(ephys_fpga.__file__).parent.joinpath("ephys_sessions")
fixture = {
    "pcs": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_pcs.npy")),
    "delays": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimDelays.npy")),
    "ids": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimIDs.npy")),
}

# load general metadata
with open(path_fixtures.joinpath("passive_stim_meta.json"), "r") as f:
    meta = json.load(f)
t_end_ephys = passive.ephysCW_end(session_path=session_path)
# load stimulus sequence
sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
# fpga_sync = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"])
fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=t_end_ephys)
audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=t_end_ephys)
bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=t_end_ephys)


def get_spacers():
    """
    load and get spacer information, do corr to find spacer timestamps
    returns t_passive_starts, t_starts, t_ends
    """
    spacer_template = (
        np.array(meta["VISUAL_STIM_0"]["ttl_frame_nums"], dtype=np.float32) / FRAME_FS
    )
    jitter = 3 / FRAME_FS  # allow for 3 screen refresh as jitter
    t_quiet = meta["VISUAL_STIM_0"]["delay_around"]
    spacer_times, _ = passive.get_spacer_times(
        spacer_template=spacer_template, jitter=jitter, ttl_signal=fttl["times"], t_quiet=t_quiet
    )

    # Check correct number of spacers found
    n_exp_spacer = np.sum(np.array(meta["STIM_ORDER"]) == 0)  # Hardcoded 0 for spacer
    if n_exp_spacer != np.size(spacer_times) / 2:
        raise ValueError(
            f"The number of expected spacer ({n_exp_spacer}) "
            f"is different than the one found on the raw "
            f"trace ({np.size(spacer_times)/2})"
        )

    spacer_times = np.r_[spacer_times.flatten(), sync["times"][-1]]
    return spacer_times[0], spacer_times[1::2], spacer_times[2::2]


# loop over stimuli , get start/end times and meta dictionary key
t_start_passive, t_starts, t_ends = get_spacers()
tspontaneous = [t_starts[0], t_ends[0]]
trfm = [t_starts[1], t_ends[1]]
treplay = [t_starts[2], t_ends[2]]

# 3/3 Replay of task stimuli
fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=treplay[0])
audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=treplay[0])
bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=treplay[0])

np.diff(fttl['times'])
# get idxs of where the diff is of a gabor presentation
# This will ignore the first puls where the onset is hidden
diff_idxs = np.where(np.diff(fttl['times']) < 0.4)[0]
# get the previous polarity change (which hsould be the end of previous stim presentation)
idx_end_stim = diff_idxs - 1
# append the last stim end diff_idx[-1] + 1
idx_end_stim = np.append(idx_end_stim, diff_idxs[-1] + 1)
assert len(idx_end_stim) == sum(fixture['ids'] == 'G'), "wrong number of GaborEnd times"
np.median(fttl['times'][diff_idxs])


plt.plot(np.diff(fttl['times']), '.')

plt.axhline(0.4)
plt.axvlines(fttl['times'][idx_end_stim])
# Get valve intervals

# Get Tone and Noise cue instervals

# Get Gabor patches intervals








from ibllib.plots import squares, vertical_lines, color_cycle
import matplotlib.pyplot as plt

pl, ax = plt.subplots(1, 1)
for i, lab in enumerate(["frame2ttl", "audio", "bpod"]):
    sy = ephys_fpga._get_sync_fronts(sync, sync_map[lab], tmin=t_start_passive)
    squares(sy["times"], sy["polarities"], yrange=[0.1 + i, 0.9 + i], color="k", ax=ax)

vertical_lines(
    np.r_[t_start_passive, t_starts, t_ends],
    ymin=-1,
    ymax=4,
    color=color_cycle(0),
    ax=ax,
    label="spacers",
)
vertical_lines(
    fttl['times'][idx_end_stim],
    ymin=0,
    ymax=1,
    color=color_cycle(1),
    ax=ax,
    label="Gabor end",
)

ax.legend()

plt.show()
# print(det)
print(eid)

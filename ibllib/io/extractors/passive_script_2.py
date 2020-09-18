#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, September 7th 2020, 11:51:17 am
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import alf.io
import ibllib.io.extractors.passive as passive
import ibllib.io.raw_data_loaders as rawio
from ibllib.io.extractors import ephys_fpga
from ibllib.plots import color_cycle, squares, vertical_lines
from ibllib.qc.oneutils import random_ephys_session
from oneibl.one import ONE

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

eid = '02fbb6da-3034-47d6-a61b-7d06c796a830'  # Wrong num of soundOn_times
# danlab/Subjects/DY_010/2020-01-29/001 BAD SoundCard?

eid = "01864d6f-31e8-49c9-aadd-2e5021ea0ee7"  # not working
# number of expected spacers wrong
eid = "fff7d745-bbce-4756-a690-3431e2f3d108"
eid = "849c9acb-8223-4e09-8cb1-95004b452baf"
eid = "d1442e39-68de-41d0-9449-35e5cfe5a94f"
eid = "e6adaabd-2bd8-4956-9c4d-53cf02d1c0e7"
eid = "a9272cce-6914-4b45-a05f-9e925b4c472a"

# AssertionError: multiple object sync with the same attribute in probe01, restrict parts/namespace
eid = "c7a9f89c-2c1d-4302-94b8-effcbe4a85b3"
eid = "c7a9f89c-2c1d-4302-94b8-effcbe4a85b3"

# Spikeglx.py error (pathlib None)
eid = "4ddb8a95-788b-48d0-8a0a-66c7c796da96"
eid = "c1fc4aac-4123-49e4-a05c-ee06deac7b5d"
eid = "f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1"


# OK
# Gabor on HIGH
eid = "193fe7a8-4eb5-4f3e-815a-0c45864ddd77"
eid = "8435e122-c0a4-4bea-a322-e08e8038478f"
# eid = one.search(subject="CSH_ZAD_022", date_range="2020-05-24", number=1)[0]
# Gabor on LOW
eid = "a82800ce-f4e3-4464-9b80-4c3d6fade333"
eid = "03cf52f6-fba6-4743-a42e-dd1ac3072343"
eid = "a8a8af78-16de-4841-ab07-fde4b5281a03"
eid = "db4df448-e449-4a6f-a0e7-288711e7a75a"

eid, det = random_ephys_session()
eid = 'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b'  # wrong number of GaborEnd times


local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)

session_path = alf.io.get_session_path(local_paths[0])


# load session fixtures
def load_passive_session_fixtures(session_path):
    settings = rawio.load_settings(session_path)
    ses_nb = settings["SESSION_ORDER"][settings["SESSION_IDX"]]
    path_fixtures = Path(ephys_fpga.__file__).parent.joinpath("ephys_sessions")

    fixture = {
        "pcs": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_pcs.npy")),
        "delays": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimDelays.npy")),
        "ids": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimIDs.npy")),
    }

    return fixture


def load_passive_stim_meta():
    path_fixtures = Path(ephys_fpga.__file__).parent.joinpath("ephys_sessions")
    with open(path_fixtures.joinpath("passive_stim_meta.json"), "r") as f:
        meta = json.load(f)

    return meta


# load general metadata

# fpga_sync = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"])


def get_passive_spacers(session_path, sync=None, sync_map=None):
    """
    load and get spacer information, do corr to find spacer timestamps
    returns t_passive_starts, t_starts, t_ends
    """
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)

    meta = load_passive_stim_meta()
    t_end_ephys = passive.ephysCW_end(session_path=session_path)
    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=t_end_ephys)
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


# Load sessions sync channels, map adnd fixtures
sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
fixture = load_passive_session_fixtures(session_path)

pl, ax = plt.subplots(1, 1)
for i, device in enumerate(["frame2ttl", "audio", "bpod"]):
    sy = ephys_fpga._get_sync_fronts(sync, sync_map[device], tmin=t_start_passive)
    squares(sy["times"], sy["polarities"], yrange=[0.1 + i, 0.9 + i], color="k", ax=ax)




# Define start and end times of the 3 passive periodes
t_start_passive, t_starts, t_ends = get_passive_spacers(session_path, sync=sync, sync_map=sync_map)
tspontaneous = [t_starts[0], t_ends[0]]
trfm = [t_starts[1], t_ends[1]]
treplay = [t_starts[2], t_ends[2]]
# TODO export this to a dstype

vertical_lines(
    np.r_[t_start_passive, t_starts, t_ends],
    ymin=-1,
    ymax=4,
    color=color_cycle(0),
    ax=ax,
    label="spacers",
)

# 3/3 Replay of task stimuli
fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=treplay[0])

# get idxs of where the diff is of a gabor presentation.
# This will get the start of a gabor patch presentation
# and ignore the first pulse where the onset is hidden
# 0.3 is the expected gabor length and 0.5 isthe expected delay length.
# We use 0.4 to split the difference and to allow for maximum drift
# At this stage we want to define what pulses are and not quality control them.
#FIXME: Like this v
# find highest value of lower diffs
# Find lower value of higher diffs
# set threshold in the middle!!
# every other where max < 1
diff_idxs = np.where(np.diff(fttl["times"]) <= 0.4)[0]
# move one change back, i.e. get the OFFset of the previous stimulus
# get the previous polarity change (which should be the end of previous stim presentation)
idx_end_stim = diff_idxs - 1
# We've now lost the last stim presentation so get the last onset and move to it's offset
# append it to the end indexes diff_idx[-1] + 1
idx_end_stim = np.append(idx_end_stim, diff_idxs[-1] + 1)
assert len(idx_end_stim) == sum(fixture["ids"] == "G"), "wrong number of GaborEnd times"
# np.median(np.diff(fttl['times'])[diff_idxs])

# Get the start times from the end times
# Check if first end stim detected is the first sample


start_times = fttl["times"][idx_end_stim - 1]
# If first stimOff detected is first sample first stimON is wrong
# If first stimOff detected minus the fist stimON detected is > 0.3 (expected stim length)
# first stimON is also wrong
# in any of these cases extrapolate based upon the stim expected length
if (idx_end_stim[0] == 0) or (
    fttl["times"][idx_end_stim[0]] - fttl["times"][idx_end_stim[0] - 1] > 0.3
):
    start_times[0] = fttl["times"][idx_end_stim[0]] - 0.3
# Move the end times to a var
end_times = fttl["times"][idx_end_stim]

# TODO export this to a dstype
passiveGabor_properties = fixture["pcs"]
passiveGabor_properties_metadata = ["position, contrast, phase"]
# intervals dstype requires reshaping of start and end times
passiveGabor_intervals = np.array([(x, y) for x, y in zip(start_times, end_times)])

# Check length of presentation of stim is  within 100msof expected
assert np.allclose(
    [y - x for x, y in passiveGabor_intervals], 0.3, atol=0.1
), "Stim length seems wrong"

vertical_lines(
    start_times,
    ymin=0,
    ymax=1,
    color=color_cycle(1),
    ax=ax,
    label="GaborOn_times",
)
vertical_lines(
    end_times,
    ymin=0,
    ymax=1,
    color=color_cycle(2),
    ax=ax,
    label="GaborOff_times",
)
# passiveValve.intervals
# Get valve intervals from bpod channel
bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=treplay[0])
# bpod channel should only contain valve output for passiveCW protocol
# All high fronts == valve open times and low fronts == valve close times
valveOn_times = bpod["times"][bpod["polarities"] > 0]
valveOff_times = bpod["times"][bpod["polarities"] < 0]
# TODO export this to a dstype

assert len(valveOn_times) == NVALVE, "Wrong number of valve ONSET times"
assert len(valveOff_times) == NVALVE, "Wrong number of valve OFFSET times"
assert len(bpod["times"]) == NVALVE * 2, "Wrong number of valve FRONTS detected"  # (40 * 2)

# check all values are within bpod tolerance of 100µs
assert np.allclose(
    valveOff_times - valveOn_times, valveOff_times[0] - valveOn_times[0], atol=0.0001
), "Some valve outputs are longer or shorter than others"


# Get Tone and Noise cue intervals
audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=treplay[0])

# Get all sound onsets and offsets
soundOn_times = audio["times"][audio["polarities"] > 0]
soundOff_times = audio["times"][audio["polarities"] < 0]
# Check they are the correct number
assert len(soundOn_times) == NTONES + NNOISES, "Wrong number of sound ONSETS"
assert len(soundOff_times) == NTONES + NNOISES, "Wrong number of sound OFFSETS"

diff = soundOff_times - soundOn_times
# Tone is ~100ms so check if diff < 0.3
toneOn_times = soundOn_times[diff < 0.3]
toneOff_times = soundOff_times[diff < 0.3]
# Noise is ~500ms so check if diff > 0.3
noiseOn_times = soundOn_times[diff > 0.3]
noiseOff_times = soundOff_times[diff > 0.3]
# TODO export this to a dstype

assert len(toneOn_times) == NTONES
assert len(toneOff_times) == NTONES
assert len(noiseOn_times) == NNOISES
assert len(noiseOff_times) == NNOISES

# Fixed delays from soundcard ~500µs
np.allclose(toneOff_times - toneOn_times, 0.1, atol=0.0006)
np.allclose(noiseOff_times - noiseOn_times, 0.5, atol=0.0005)


# Look at it
vertical_lines(
    valveOn_times,
    ymin=2,
    ymax=3,
    color=color_cycle(3),
    ax=ax,
    label="ValveOn_times",
)
vertical_lines(
    valveOff_times,
    ymin=2,
    ymax=3,
    color=color_cycle(4),
    ax=ax,
    label="ValveOff_times",
)
vertical_lines(
    toneOn_times,
    ymin=1,
    ymax=2,
    color=color_cycle(5),
    ax=ax,
    label="toneOn_times",
)
vertical_lines(
    toneOff_times,
    ymin=1,
    ymax=2,
    color=color_cycle(6),
    ax=ax,
    label="toneOff_times",
)
vertical_lines(
    noiseOn_times,
    ymin=1,
    ymax=2,
    color=color_cycle(7),
    ax=ax,
    label="noiseOn_times",
)
vertical_lines(
    noiseOff_times,
    ymin=1,
    ymax=2,
    color=color_cycle(8),
    ax=ax,
    label="noiseOff_times",
)

ax.legend()
# plt.show()
# %gui qt
# print(det)
print(eid)

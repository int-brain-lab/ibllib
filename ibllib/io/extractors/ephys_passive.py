#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, September 7th 2020, 11:51:17 am
import json
import random
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import axvline

import alf.folders as folders
import alf.io
import ibllib.io.extractors.passive as passive
import ibllib.io.raw_data_loaders as rawio
from ibllib.io.extractors import ephys_fpga
from ibllib.plots import color_cycle, squares, vertical_lines
from oneibl.one import ONE
import logging

log = logging.getLogger("ibllib")

# hardcoded var
FRAME_FS = 60  # Sampling freq of the ipad screen, in Hertz
FS_FPGA = 30000  # Sampling freq of the neural recording system screen, in Hertz
NVALVE = 40  # number of expected valve clicks
NGABOR = 20 + 20 * 4 * 2  # number of expected Gabor patches
NTONES = 40
NNOISES = 40
DEBUG_PLOTS = False

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

min_dataset_types = [
    "_spikeglx_sync.times",
    "_spikeglx_sync.channels",
    "_spikeglx_sync.polarities",
    "_iblrig_RFMapStim.raw",
    "ephysData.raw.meta",
    "_iblrig_taskSettings.raw",
    "_iblrig_taskData.raw",
]


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


# 1/3 Define start and end times of the 3 passive periods
def extract_passive_periods(session_path, sync=None, sync_map=None):
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)

    t_start_passive, t_starts, t_ends = get_passive_spacers(
        session_path, sync=sync, sync_map=sync_map
    )
    tspontaneous = [t_starts[0], t_ends[0]]
    trfm = [t_starts[1], t_ends[1]]
    treplay = [t_starts[2], t_ends[2]]
    # TODO export this to a dstype
    return t_start_passive, tspontaneous, trfm, treplay


# 2/3 RFMapping stimuli
def extract_rfmapping(session_path, sync=None, sync_map=None):
    meta = load_passive_stim_meta()
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)

    t_start_passive, tspontaneous, trfm, treplay = extract_passive_periods(
        session_path, sync=sync, sync_map=sync_map
    )

    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=trfm[0], tmax=trfm[1])

    RF_file = Path.joinpath(session_path, "raw_passive_data", "_iblrig_RFMapStim.raw.bin")
    RF_frames, RF_ttl_trace = passive.reshape_RF(RF_file=RF_file, meta_stim=meta[s["mkey"]])
    rf_id_up, rf_id_dw, RF_n_ttl_expected = passive.get_id_raisefall_from_analogttl(RF_ttl_trace)
    meta[s["mkey"]]["ttl_num"] = RF_n_ttl_expected
    RF_times = passive.check_n_ttl_between(
        n_exp=meta[s["mkey"]]["ttl_num"],
        key_stim=s["mkey"],
        t_start_search=s["start"] + 0.2,
        t_end_search=s["end"] - 0.2,
        ttl=fttl,
    )
    RF_times_1 = RF_times[0::2]
    # Interpolate times for RF before outputting dataset
    times_interp_RF = passive.interpolate_rf_mapping_stimulus(
        idxs_up=rf_id_up,
        idxs_dn=rf_id_dw,
        times=RF_times_1,
        Xq=np.arange(RF_frames.shape[0]),
        t_bin=1 / FRAME_FS,
    )

    return RF_frames, times_interp_RF


# 3/3 Replay of task stimuli
def _extract_passiveGabor_df(fttl, session_path):
    # At this stage we want to define what pulses are and not quality control them.
    # Pulses are stricty altternating with intevals
    # find min max lengths for both (we don'tknow which are pulses and which are intervals yet)
    # remove first and last pulse
    diff0 = (np.min(np.diff(fttl["times"])[2:-2:2]), np.max(np.diff(fttl["times"])[2:-1:2]))
    diff1 = (np.min(np.diff(fttl["times"])[3:-2:2]), np.max(np.diff(fttl["times"])[3:-1:2]))
    # Highest max is of the intervals
    if max(diff0 + diff1) in diff0:
        thresh = diff0[0]
    elif max(diff0 + diff1) in diff1:
        thresh = diff1[0]
    # Anything lower than the min length of intervals is a pulse
    idx_start_stims = np.where((np.diff(fttl["times"]) < thresh) & (np.diff(fttl["times"]) > 0.1))[
        0
    ]
    # Check if any pulse has been missed
    # i.e. expected lenght (without first puls) and that it's alternating
    if len(idx_start_stims) < NGABOR - 1 and np.any(np.diff(idx_start_stims) > 2):
        log.warning("Looks like one or more pulses were not detected, trying to extrapolate...")
        missing_where = np.where(np.diff(idx_start_stims) > 2)[0]
        insert_where = missing_where + 1
        missing_value = idx_start_stims[missing_where] + 2
        idx_start_stims = np.insert(idx_start_stims, insert_where, missing_value)

    idx_end_stims = idx_start_stims + 1

    start_times = fttl["times"][idx_start_stims]
    end_times = fttl["times"][idx_end_stims]
    # missing the first stim
    first_stim_off_idx = idx_start_stims[0] - 1
    first_stim_on_idx = first_stim_off_idx - 1
    if first_stim_on_idx <= 0:
        end_times = np.insert(end_times, 0, fttl["times"][first_stim_off_idx])
        start_times = np.insert(start_times, 0, end_times[0] - 0.3)

    # intervals dstype requires reshaping of start and end times
    passiveGabor_intervals = np.array([(x, y) for x, y in zip(start_times, end_times)])

    # Check length of presentation of stim is  within 150msof expected
    if not np.allclose([y - x for x, y in passiveGabor_intervals], 0.3, atol=0.15):
        log.warning("Some Gabor presentation lengths seem wrong.")

    assert (
        len(passiveGabor_intervals) == NGABOR
    ), f"Wrong number of Gabor stimuli detected: {len(passiveGabor_intervals)} / {NGABOR}"
    fixture = load_passive_session_fixtures(session_path)
    passiveGabor_properties = fixture["pcs"]
    passiveGabor_table = np.append(passiveGabor_intervals, passiveGabor_properties, axis=1)
    columns = ["start", "stop", "position", "contrast", "phase"]
    passiveGabor_df = pd.DataFrame(passiveGabor_table, columns=columns)
    # TODO save dstype
    return passiveGabor_df


def _extract_passiveValve_intervals(bpod):
    # passiveValve.intervals
    # Get valve intervals from bpod channel
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

    return np.array([(x, y) for x, y in zip(valveOn_times, valveOff_times)])


def _extract_passiveAudio_intervals(audio):
    # Get Tone and Noise cue intervals

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
    np.allclose(noiseOff_times - noiseOn_times, 0.5, atol=0.0006)

    passiveTone_intervals = np.append(
        toneOn_times.reshape((len(toneOn_times), 1)),
        toneOff_times.reshape((len(toneOff_times), 1)),
        axis=1,
    )
    passiveNoise_intervals = np.append(
        noiseOn_times.reshape((len(noiseOn_times), 1)),
        noiseOff_times.reshape((len(noiseOff_times), 1)),
        axis=1,
    )
    return passiveTone_intervals, passiveNoise_intervals


def extract_replay(session_path, sync=None, sync_map=None):
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)

    t_start_passive, tspontaneous, trfm, treplay = extract_passive_periods(
        session_path, sync=sync, sync_map=sync_map
    )

    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=treplay[0])
    passiveGabor_df = _extract_passiveGabor_df(fttl, session_path)

    bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=treplay[0])
    passiveValve_intervals = _extract_passiveValve_intervals(bpod)

    audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=treplay[0])
    passiveTone_intervals, passiveNoise_intervals = _extract_passiveAudio_intervals(audio)

    return passiveGabor_df, passiveValve_intervals, passiveTone_intervals, passiveNoise_intervals


def extract_replay_plot(session_path):
    # Load sessions sync channels, map
    sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)

    f, ax = plt.subplots(1, 1)
    f.suptitle("/".join(str(session_path).split("/")[-5:]))
    plot_sync_channels(sync=sync, sync_map=sync_map, ax=ax)

    t_start_passive, t_starts, t_ends = get_passive_spacers(
        session_path, sync=sync, sync_map=sync_map
    )
    t_start_passive, tspontaneous, trfm, treplay = extract_passive_periods(
        session_path, sync=sync, sync_map=sync_map
    )
    plot_passive_periods(t_start_passive, t_starts, t_ends, ax=ax)

    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=treplay[0])
    passiveGabor_df = _extract_passiveGabor_df(fttl)
    plot_gabor_times(passiveGabor_df, ax=ax)

    bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=treplay[0])
    passiveValve_intervals = _extract_passiveValve_intervals(bpod)
    plot_valve_times(passiveValve_intervals, ax=ax)

    audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=treplay[0])
    passiveTone_intervals, passiveNoise_intervals = _extract_passiveAudio_intervals(audio)
    plot_audio_times(passiveTone_intervals, passiveNoise_intervals, ax=ax)


# Mian passiveCWe xtractor, calls all others
# TODO: try catch so it always extracts what it can + add option to plot stuff
def extract_passive_choice_world(session_path):
    sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)

    t_start_passive, tspontaneous, trfm, treplay = extract_passive_periods(
        session_path, sync=sync, sync_map=sync_map
    )

    RF_frames, times_interp_RF = extract_rfmapping(session_path, sync=sync, sync_map=sync_map)

    (
        passiveGabor_df,
        passiveValve_intervals,
        passiveTone_intervals,
        passiveNoise_intervals,
    ) = extract_replay(session_path, sync=sync, sync_map=sync_map)

    return  # TODO: return something


# PLOTTING
def plot_sync_channels(sync, sync_map, ax=None):
    # Plot all sync pulses
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for i, device in enumerate(["frame2ttl", "audio", "bpod"]):
        sy = ephys_fpga._get_sync_fronts(sync, sync_map[device])  # , tmin=t_start_passive)
        squares(sy["times"], sy["polarities"], yrange=[0.1 + i, 0.9 + i], color="k", ax=ax)


def plot_passive_periods(t_start_passive, t_starts, t_ends, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    # Update plot
    vertical_lines(
        np.r_[t_start_passive, t_starts, t_ends],
        ymin=-1,
        ymax=4,
        color=color_cycle(0),
        ax=ax,
        label="spacers",
    )
    ax.legend()


def plot_gabor_times(passiveGabor_df, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    # Update plot
    vertical_lines(
        passiveGabor_df["start"].values,
        ymin=0,
        ymax=1,
        color=color_cycle(1),
        ax=ax,
        label="GaborOn_times",
    )
    vertical_lines(
        passiveGabor_df["stop"].values,
        ymin=0,
        ymax=1,
        color=color_cycle(2),
        ax=ax,
        label="GaborOff_times",
    )
    ax.legend()


def plot_valve_times(passiveValve_intervals, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    # Update the plot
    vertical_lines(
        passiveValve_intervals[:, 0],
        ymin=2,
        ymax=3,
        color=color_cycle(3),
        ax=ax,
        label="ValveOn_times",
    )
    vertical_lines(
        passiveValve_intervals[:, 1],
        ymin=2,
        ymax=3,
        color=color_cycle(4),
        ax=ax,
        label="ValveOff_times",
    )
    ax.legend()


def plot_audio_times(passiveTone_intervals, passiveNoise_intervals, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    # Look at it
    vertical_lines(
        passiveTone_intervals[:, 0],
        ymin=1,
        ymax=2,
        color=color_cycle(5),
        ax=ax,
        label="toneOn_times",
    )
    vertical_lines(
        passiveTone_intervals[:, 1],
        ymin=1,
        ymax=2,
        color=color_cycle(6),
        ax=ax,
        label="toneOff_times",
    )
    vertical_lines(
        passiveNoise_intervals[:, 0],
        ymin=1,
        ymax=2,
        color=color_cycle(7),
        ax=ax,
        label="noiseOn_times",
    )
    vertical_lines(
        passiveNoise_intervals[:, 1],
        ymin=1,
        ymax=2,
        color=color_cycle(8),
        ax=ax,
        label="noiseOff_times",
    )

    ax.legend()
    # plt.show()


if __name__ == "__main__":
    # load data
    one = ONE()
    # eids = one.search(dataset_types=min_dataset_types)
    # session_paths = []
    # for i, eid in enumerate(eids):
    #     try:
    #         local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)
    #         session_paths.append(alf.io.get_session_path(local_paths[0]))
    #     except BaseException as e:
    #         print(f"{i+1}/{len(eids)} - Failed session: {eid}\n\n{e}")
    #     print(f"\n\n{i+1}/{len(eids)}\n\n")

    # failed = []
    # for i, sp in enumerate(session_paths):
    #     try:
    #         extract_replay(sp)
    #     except BaseException as e:
    #         failed.append((sp, e))
    #     print(f"\n{i+1}/{len(session_paths)}")
    # print(f"nfailed = {len(failed)} / {len(session_paths)}")

    some_eids = [
        "c6db3304-c906-400c-aa0f-45dd3945b2ea",
        "88d24c31-52e4-49cc-9f32-6adbeb9eba87",
        "6fb1e12c-883b-46d1-a745-473cde3232c8",
        "83769b74-6e3b-422a-8648-50c1a32c5dd4",
        "695a6073-eae0-49e0-bb0f-e9e57a9275b9",
        "9cb1fb73-ab3f-488b-bcab-474994de38a8",
        "d86f3850-5183-4329-80ea-6902b9eb0e13",
        "f3ce3197-d534-4618-bf81-b687555d1883",
        "251ece37-7798-477c-8a06-2845d4aa270c",
        "aa20388b-9ea3-4506-92f1-3c2be84b85db",
        "c3d9b6fb-7fa9-4413-a364-92a54df0fc5d",
        "d23a44ef-1402-4ed7-97f5-47e9a7a504d9",
        "89e258e9-cbca-4eca-bac4-13a2388b5113",
        "5339812f-8b91-40ba-9d8f-a559563cc46b",
        "768a371d-7e88-47f8-bf21-4a6a6570dd6e",
        "eba834fd-74d7-4453-a11d-3ef4c7e55fe1",
        "62a5c50d-eaf4-4d48-b37d-b84cab0e90b3",
        "f49d972a-cf76-40c1-bf28-b83470ad6443",
        "d5a57a4c-d28b-4079-a549-abda2b9a00db",
        "0d8a7628-6c04-4d4b-bd99-95f2bda3e700",
        "308274fc-28e8-4bfd-a4e3-3903b7b48c28",
        "b3e335a4-3fe4-43cc-beb1-d3d3a802b03c",
        "6b82f9ef-bf10-42a8-b891-ef0d1fcc1593",
        "ebe2efe3-e8a1-451a-8947-76ef42427cc9",
        "edd22318-216c-44ff-bc24-49ce8be78374",
        "71e55bfe-5a3a-4cba-bdc7-f085140d798e",
        "49e0ab27-827a-4c91-bcaa-97eea27a1b8d",
        "5adab0b7-dfd0-467d-b09d-43cb7ca5d59c",
        "6527e2f1-8b2b-4b9b-a9dd-2a0206603ad8",
        "7f6b86f9-879a-4ea2-8531-294a221af5d0",
        "8c33abef-3d3e-4d42-9f27-445e9def08f9",
        "7b04526b-f0f7-41b9-93fd-d88d96c889b0",
        "f833e88a-fc3c-4cf5-80bb-ad0b41cc9053",
        "61e11a11-ab65-48fb-ae08-3cb80662e5d6",
        "eae2a292-d065-48f2-aca4-4da6832ea78f",
        "c7248e09-8c0d-40f2-9eb4-700a8973d8c8",
        "03063955-2523-47bd-ae57-f7489dd40f15",
        "1e45d992-c356-40e1-9be1-a506d944896f",
        "4d8c7767-981c-4347-8e5e-5d5fffe38534",
        "85d2b9d0-010b-470e-a00c-e13c5ca12fda",
        "41dfdc2a-987a-402a-99ae-779d5f569566",
        "252893e5-29fa-488e-b090-a48a4402fada",
        "e71bd25a-8c1e-4751-8985-4463a91c1b66",
        "fe1fd79f-b051-411f-a0a9-2530a02cc78d",
        "0fe99726-9982-4c41-a07c-2cd7af6a6733",
        "934dd7a4-fbdc-459c-8830-04fe9033bc28",
        "b39752db-abdb-47ab-ae78-e8608bbf50ed",
        "ee8b36de-779f-4dea-901f-e0141c95722b",
        "f9860a11-24d3-452e-ab95-39e199f20a93",
        "bd456d8f-d36e-434a-8051-ff3997253802",
        "b658bc7d-07cd-4203-8a25-7b16b549851b",
        "862ade13-53cd-4221-a3fa-dda8643641f2",
        "7622da34-51b6-4661-98ae-a57d40806008",
        "4720c98a-a305-4fba-affb-bbfa00a724a4",
        "66d98e6e-bcd9-4e78-8fbb-636f7e808b29",
        "f25642c6-27a5-4a97-9ea0-06652db79fbd",
        "b9003c82-6178-47ed-9ff2-75087a49c2f7",
        "77d2dfe0-a48d-4a46-8c13-2b9399c46ad3",
        "708e797c-5496-4742-9f00-1d6346768d7a",
        "8207abc6-6b23-4762-92b4-82e05bed5143",
        "ae384fe2-fa03-4366-a107-69b341b7368a",
        "1eed29f4-3311-4f54-8a6a-c2efa0675e4a",
        "14af98b4-f72b-46bf-ac06-97f61adc62cc",
        "7c0df410-ca82-4464-9de7-6e200b3a1d05",
        "8b1ad76f-7f0a-44fa-89f7-060be21c202e",
        "90e74228-fd1a-482f-bd56-05dbad132861",
        "6a601cc5-7b79-4c75-b0e8-552246532f82",
        "a82800ce-f4e3-4464-9b80-4c3d6fade333",
        "a66f1593-dafd-4982-9b66-f9554b6c86b5",
        "d855576e-5b34-41bf-8e3b-2bea0cae1380",
        "41431f53-69fd-4e3b-80ce-ea62e03bf9c7",
        "1eac875c-feaa-4a30-b148-059b954b11d8",
        "8db36de1-8f17-4446-b527-b5d91909b45a",
        "da188f2c-553c-4e04-879b-c9ea2d1b9a93",
        "03cf52f6-fba6-4743-a42e-dd1ac3072343",
        "cccee322-3163-44bb-9b4b-22083e96e09f",
        "fee6dd62-01d9-42d2-bcc1-5a7e27244edc",
        "37602f31-1b09-4fc9-87b8-cc13b922d0e6",
        "c90cdfa0-2945-4f68-8351-cb964c258725",
        "7be8fec4-406b-4e74-8548-d2885dcc3d5e",
        "ded7c877-49cf-46ad-b726-741f1cf34cef",
        "6364ff7f-6471-415a-ab9e-632a12052690",
        "af74b29d-a671-4c22-a5e8-1e3d27e362f3",
        "56d38157-bb5a-4561-ab5c-3df05a5d6e28",
        "9931191e-8056-4adc-a410-a4a93487423f",
        "e535fb62-e245-4a48-b119-88ce62a6fe67",
        "f10efe41-0dc0-44d0-8f26-5ff68dca23e9",
        "1191f865-b10a-45c8-9c48-24a980fd9402",
        "765ba913-eb0e-4f7d-be7e-4964abe8b27b",
        "6668c4a0-70a4-4012-a7da-709660971d7a",
        "37e96d0b-5b4b-4c6e-9b29-7edbdc94bbd0",
        "d16a9a8d-5f42-4b49-ba58-1746f807fcc1",
        "9a629642-3a9c-42ed-b70a-532db0e86199",
        "e5c772cd-9c92-47ab-9525-d618b66a9b5d",
        "dda5fc59-f09a-4256-9fb5-66c67667a466",
        "57b5ae8f-d446-4161-b439-b191c5e3e77b",
        "720a3fe6-5dfc-4a23-84f0-2f0b08e10ec2",
        "d2f5a130-b981-4546-8858-c94ae1da75ff",
        "7939711b-8b4d-4251-b698-b97c1eaa846e",
    ]
    some_session_paths = [
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-20/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-19/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-18/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/CSP016/2020-09-18/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-17/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/CSP016/2020-09-17/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/CSP016/2020-09-16/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-14/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-13/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-12/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL024/2020-09-12/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-11/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_039/2020-09-10/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-28/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-27/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-26/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-25/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL027/2020-08-20/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-20/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-18/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-16/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-14/005",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-14/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-13/002",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-12/006",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-11/002",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-10/002",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3001/2020-08-07/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3001/2020-08-05/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_038/2020-08-01/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_038/2020-07-31/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_038/2020-07-30/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-30/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-29/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-29/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-28/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-28/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-27/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-27/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-19/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-18/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_042/2020-07-17/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-17/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-16/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_042/2020-07-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-14/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_042/2020-07-13/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-13/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_021/2020-06-16/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_024/2020-06-11/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_024/2020-06-09/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_024/2020-06-08/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-07/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-05/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-04/002",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-03/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-28/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-26/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-25/002",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-24/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-23/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-22/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-21/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-20/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-19/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_011/2020-03-23/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_011/2020-03-22/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-08/2020-03-20/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-08/2020-03-19/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-08/2020-03-18/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_018/2020-03-16/002",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-15/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-14/002",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-14/003",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-13/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-13/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-12/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-12/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-12/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-11/004",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-10/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-09/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-08/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-07/002",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-07/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-07/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-06/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-05/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-04/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-03/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-28/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-21/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-19/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-19/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-19/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-18/001",
    ]
    some_failing_errors = [
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL027/2020-08-20/001",
            ValueError(
                "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
            ),
        ),
        (
            "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-18/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-14/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-12/006",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3001/2020-08-05/001",
            ValueError(
                "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
            ),
        ),
        (
            "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-30/001",
            ValueError(
                "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
            ),
        ),
        (
            "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-29/001",
            ValueError("first array argument cannot be empty"),
        ),
        (
            "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-28/001",
            ValueError(
                "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
            ),
        ),
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-27/001",
            AssertionError("Wrong number of valve ONSET times"),
        ),
        (
            "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-27/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-13/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-07/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-05/001",
            IndexError("index 0 is out of bounds for axis 0 with size 0"),
        ),
        (
            "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-28/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-26/001",
            ValueError(
                "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
            ),
        ),
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-14/003",
            ValueError("zero-size array to reduction operation minimum which has no identity"),
        ),
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-12/001",
            AssertionError("Wrong number of valve ONSET times"),
        ),
        (
            "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-12/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-10/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-07/001",
            AssertionError("Wrong number of valve ONSET times"),
        ),
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-05/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-04/001",
            ValueError(
                "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
            ),
        ),
        (
            "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-28/001",
            ValueError(
                "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
            ),
        ),
        (
            "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-19/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
        (
            "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-18/001",
            AssertionError("Wrong number of Gabor stimuli detected"),
        ),
    ]

    all_eids = [
        "4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a",
        "c6db3304-c906-400c-aa0f-45dd3945b2ea",
        "88d24c31-52e4-49cc-9f32-6adbeb9eba87",
        "6fb1e12c-883b-46d1-a745-473cde3232c8",
        "83769b74-6e3b-422a-8648-50c1a32c5dd4",
        "695a6073-eae0-49e0-bb0f-e9e57a9275b9",
        "9cb1fb73-ab3f-488b-bcab-474994de38a8",
        "d86f3850-5183-4329-80ea-6902b9eb0e13",
        "f3ce3197-d534-4618-bf81-b687555d1883",
        "251ece37-7798-477c-8a06-2845d4aa270c",
        "aa20388b-9ea3-4506-92f1-3c2be84b85db",
        "c3d9b6fb-7fa9-4413-a364-92a54df0fc5d",
        "d23a44ef-1402-4ed7-97f5-47e9a7a504d9",
        "89e258e9-cbca-4eca-bac4-13a2388b5113",
        "5339812f-8b91-40ba-9d8f-a559563cc46b",
        "768a371d-7e88-47f8-bf21-4a6a6570dd6e",
        "eba834fd-74d7-4453-a11d-3ef4c7e55fe1",
        "62a5c50d-eaf4-4d48-b37d-b84cab0e90b3",
        "f49d972a-cf76-40c1-bf28-b83470ad6443",
        "d5a57a4c-d28b-4079-a549-abda2b9a00db",
        "0d8a7628-6c04-4d4b-bd99-95f2bda3e700",
        "308274fc-28e8-4bfd-a4e3-3903b7b48c28",
        "b3e335a4-3fe4-43cc-beb1-d3d3a802b03c",
        "6b82f9ef-bf10-42a8-b891-ef0d1fcc1593",
        "ebe2efe3-e8a1-451a-8947-76ef42427cc9",
        "edd22318-216c-44ff-bc24-49ce8be78374",
        "71e55bfe-5a3a-4cba-bdc7-f085140d798e",
        "49e0ab27-827a-4c91-bcaa-97eea27a1b8d",
        "5adab0b7-dfd0-467d-b09d-43cb7ca5d59c",
        "6527e2f1-8b2b-4b9b-a9dd-2a0206603ad8",
        "7f6b86f9-879a-4ea2-8531-294a221af5d0",
        "8c33abef-3d3e-4d42-9f27-445e9def08f9",
        "7b04526b-f0f7-41b9-93fd-d88d96c889b0",
        "f833e88a-fc3c-4cf5-80bb-ad0b41cc9053",
        "61e11a11-ab65-48fb-ae08-3cb80662e5d6",
        "eae2a292-d065-48f2-aca4-4da6832ea78f",
        "c7248e09-8c0d-40f2-9eb4-700a8973d8c8",
        "03063955-2523-47bd-ae57-f7489dd40f15",
        "1e45d992-c356-40e1-9be1-a506d944896f",
        "4d8c7767-981c-4347-8e5e-5d5fffe38534",
        "85d2b9d0-010b-470e-a00c-e13c5ca12fda",
        "41dfdc2a-987a-402a-99ae-779d5f569566",
        "252893e5-29fa-488e-b090-a48a4402fada",
        "e71bd25a-8c1e-4751-8985-4463a91c1b66",
        "fe1fd79f-b051-411f-a0a9-2530a02cc78d",
        "0fe99726-9982-4c41-a07c-2cd7af6a6733",
        "934dd7a4-fbdc-459c-8830-04fe9033bc28",
        "b39752db-abdb-47ab-ae78-e8608bbf50ed",
        "ee8b36de-779f-4dea-901f-e0141c95722b",
        "f9860a11-24d3-452e-ab95-39e199f20a93",
        "bd456d8f-d36e-434a-8051-ff3997253802",
        "b658bc7d-07cd-4203-8a25-7b16b549851b",
        "862ade13-53cd-4221-a3fa-dda8643641f2",
        "7622da34-51b6-4661-98ae-a57d40806008",
        "4720c98a-a305-4fba-affb-bbfa00a724a4",
        "66d98e6e-bcd9-4e78-8fbb-636f7e808b29",
        "f25642c6-27a5-4a97-9ea0-06652db79fbd",
        "b9003c82-6178-47ed-9ff2-75087a49c2f7",
        "77d2dfe0-a48d-4a46-8c13-2b9399c46ad3",
        "708e797c-5496-4742-9f00-1d6346768d7a",
        "8207abc6-6b23-4762-92b4-82e05bed5143",
        "ae384fe2-fa03-4366-a107-69b341b7368a",
        "1eed29f4-3311-4f54-8a6a-c2efa0675e4a",
        "14af98b4-f72b-46bf-ac06-97f61adc62cc",
        "7c0df410-ca82-4464-9de7-6e200b3a1d05",
        "8b1ad76f-7f0a-44fa-89f7-060be21c202e",
        "90e74228-fd1a-482f-bd56-05dbad132861",
        "6a601cc5-7b79-4c75-b0e8-552246532f82",
        "a82800ce-f4e3-4464-9b80-4c3d6fade333",
        "a66f1593-dafd-4982-9b66-f9554b6c86b5",
        "d855576e-5b34-41bf-8e3b-2bea0cae1380",
        "41431f53-69fd-4e3b-80ce-ea62e03bf9c7",
        "1eac875c-feaa-4a30-b148-059b954b11d8",
        "8db36de1-8f17-4446-b527-b5d91909b45a",
        "da188f2c-553c-4e04-879b-c9ea2d1b9a93",
        "03cf52f6-fba6-4743-a42e-dd1ac3072343",
        "cccee322-3163-44bb-9b4b-22083e96e09f",
        "fee6dd62-01d9-42d2-bcc1-5a7e27244edc",
        "37602f31-1b09-4fc9-87b8-cc13b922d0e6",
        "c90cdfa0-2945-4f68-8351-cb964c258725",
        "7be8fec4-406b-4e74-8548-d2885dcc3d5e",
        "ded7c877-49cf-46ad-b726-741f1cf34cef",
        "6364ff7f-6471-415a-ab9e-632a12052690",
        "af74b29d-a671-4c22-a5e8-1e3d27e362f3",
        "56d38157-bb5a-4561-ab5c-3df05a5d6e28",
        "9931191e-8056-4adc-a410-a4a93487423f",
        "e535fb62-e245-4a48-b119-88ce62a6fe67",
        "f10efe41-0dc0-44d0-8f26-5ff68dca23e9",
        "1191f865-b10a-45c8-9c48-24a980fd9402",
        "765ba913-eb0e-4f7d-be7e-4964abe8b27b",
        "6668c4a0-70a4-4012-a7da-709660971d7a",
        "37e96d0b-5b4b-4c6e-9b29-7edbdc94bbd0",
        "d16a9a8d-5f42-4b49-ba58-1746f807fcc1",
        "9a629642-3a9c-42ed-b70a-532db0e86199",
        "e5c772cd-9c92-47ab-9525-d618b66a9b5d",
        "dda5fc59-f09a-4256-9fb5-66c67667a466",
        "2f63c555-eb74-4d8d-ada5-5c3ecf3b46be",
        "a19c7a3a-7261-42ce-95d5-1f4ca46007ed",
        "57b5ae8f-d446-4161-b439-b191c5e3e77b",
        "f4a4143d-d378-48a3-aed2-fa7958648c24",
        "413a6825-2144-4a50-b3fc-cf38ddd6fd1a",
        "ee13c19e-2790-4418-97ca-48f02e8013bb",
        "30e5937e-e86a-47e6-93ae-d2ae3877ff8e",
        "720a3fe6-5dfc-4a23-84f0-2f0b08e10ec2",
        "0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9",
        "d2f5a130-b981-4546-8858-c94ae1da75ff",
        "158d5d35-a2ab-4a76-87b0-51048c5d5283",
        "3c851386-e92d-4533-8d55-89a46f0e7384",
        "3dd347df-f14e-40d5-9ff2-9c49f84d2157",
        "81958127-baf9-4e68-95a0-9c1ddfb61667",
        "7939711b-8b4d-4251-b698-b97c1eaa846e",
        "ed997f98-424b-4f1d-a736-2d1eb0f35dbb",
        "db4df448-e449-4a6f-a0e7-288711e7a75a",
        "fa704052-147e-46f6-b190-a65b837e605e",
        "2fe8aa16-faab-49e3-8e13-aceb0b095a30",
        "fac00dc6-9c61-4de2-b456-a9eb21b60318",
        "dfd8e7df-dc51-4589-b6ca-7baccfeb94b4",
        "034e726f-b35f-41e0-8d6c-a22cc32391fb",
        "56956777-dca5-468c-87cb-78150432cc57",
        "4b00df29-3769-43be-bb40-128b1cba6d35",
        "266a0360-ea0a-4580-8f6a-fe5bad9ed17c",
        "83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4",
        "5386aba9-9b97-4557-abcd-abc2da66b863",
        "dd0faa76-4f49-428c-9507-6de7382a5d9e",
        "6713a4a7-faed-4df2-acab-ee4e63326f8d",
        "85dc2ebd-8aaf-46b0-9284-a197aee8b16f",
        "fb7b21c9-b50e-4145-9254-a91a50d656ca",
        "3663d82b-f197-4e8b-b299-7b803a155b84",
        "01864d6f-31e8-49c9-aadd-2e5021ea0ee7",
        "0cbeae00-e229-4b7d-bdcc-1b0569d7e0c3",
        "57fd2325-67f4-4d45-9907-29e77d3043d7",
        "a71175be-d1fd-47a3-aa93-b830ea3634a1",
        "79de526f-aed6-4106-8c26-5dfdfa50ce86",
        "cf43dbb1-6992-40ec-a5f9-e8e838d0f643",
        "8658a5ad-58ce-4626-8d46-be68cd33581b",
        "741979ce-3f10-443a-8526-2275620c8473",
        "d42bb88e-add2-414d-a60a-a3efd66acd2a",
        "ccffff9f-c432-4377-b228-e2710bc109b6",
        "ab583ab8-08bd-4ebc-a0b8-5d40af551068",
        "ecb5520d-1358-434c-95ec-93687ecd1396",
        "d9bcf951-067e-41c0-93a2-14818adf88fe",
        "202128f9-02af-4c6c-b6ce-25740e6ba8cd",
        "74bae29c-f614-4abe-8066-c4d83d7da143",
        "1c213d82-32c3-49f7-92ca-06e28907e1b4",
        "810b1e07-009e-4ebe-930a-915e4cd8ece4",
        "36280321-555b-446d-9b7d-c2e17991e090",
        "115d264b-1939-4b8e-9d17-4ed8dfc4fadd",
        "eef82e27-c20e-48da-b4b7-c443031649e3",
        "7bee9f09-a238-42cf-b499-f51f765c6ded",
        "f8d5c8b0-b931-4151-b86c-c471e2e80e5d",
        "c8e60637-de79-4334-8daf-d35f18070c29",
        "b9cc9f7b-f689-41d3-ab83-5ba5520e5dca",
        "097afc11-4214-4879-bd7a-643a4d16396e",
        "ee40aece-cffd-4edb-a4b6-155f158c666a",
        "2199306e-488a-40ab-93cb-2d2264775578",
        "0deb75fb-9088-42d9-b744-012fb8fc4afb",
        "097ba865-f424-49a3-96fb-863506fac3e0",
        "12dc8b34-b18e-4cdd-90a9-da134a9be79c",
        "e49d8ee7-24b9-416a-9d04-9be33b655f40",
        "02fbb6da-3034-47d6-a61b-7d06c796a830",
        "83121823-33aa-4b49-9e65-34a7c202a8b5",
        "3ce452b3-57b4-40c9-885d-1b814036e936",
        "465c44bd-2e67-4112-977b-36e1ac7e3f8c",
        "931a70ae-90ee-448e-bedb-9d41f3eda647",
        "ff4187b5-4176-4e39-8894-53a24b7cf36b",
        "1538493d-226a-46f7-b428-59ce5f43f0f9",
        "b03fbc44-3d8e-4a6c-8a50-5ea3498568e0",
        "1c27fd32-e872-4284-b9a5-7079453f4cbc",
        "3d6f6788-0b99-410f-9703-c43ca3e42a21",
        "bb6a5aae-2431-401d-8f6a-9fdd6de655a9",
        "193fe7a8-4eb5-4f3e-815a-0c45864ddd77",
        "510b1a50-825d-44ce-86f6-9678f5396e02",
        "032ffcdf-7692-40b3-b9ff-8def1fc18b2e",
        "90d1e82c-c96f-496c-ad4e-ee3f02067f25",
        "a8a8af78-16de-4841-ab07-fde4b5281a03",
        "3d5996a0-13bc-47ac-baae-e551f106bddc",
        "259927fd-7563-4b03-bc5d-17b4d0fa7a55",
        "2d5f6d81-38c4-4bdc-ac3c-302ea4d5f46e",
        "4fa70097-8101-4f10-b585-db39429c5ed0",
        "cb2ad999-a6cb-42ff-bf71-1774c57e5308",
        "b52182e7-39f6-4914-9717-136db589706e",
        "d33baf74-263c-4b37-a0d0-b79dcb80a764",
        "4364a246-f8d7-4ce7-ba23-a098104b96e4",
        "e80b92b7-6aab-4220-b44d-eaf1a52f3be5",
        "ff2e3f6c-a338-4c59-829d-0b225055b2df",
        "c99d53e6-c317-4c53-99ba-070b26673ac4",
        "53738f95-bd08-4d9d-9133-483fdb19e8da",
        "21e16736-fd59-44c7-b938-9b1333d25da8",
        "e5fae088-ed96-4d9b-82f9-dfd13c259d52",
        "266c32c3-4f75-4d44-9337-ef12f2980ecc",
        "40e2f1bd-6910-4635-b9a7-1e76771a422e",
        "1d364d2b-e02b-4b5d-869c-11c1a0c8cafc",
        "17231390-9b95-4ec6-806d-b3aae8af76ac",
        "f354dc45-caef-4e3e-bd42-2c19a5425114",
    ]
    all_session_paths = [
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-21/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-20/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-19/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-18/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/CSP016/2020-09-18/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-17/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/CSP016/2020-09-17/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/CSP016/2020-09-16/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_043/2020-09-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-14/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-13/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-12/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL024/2020-09-12/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_016/2020-09-11/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_039/2020-09-10/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-28/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-27/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-26/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-25/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-21/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL027/2020-08-20/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-20/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-19/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-19/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-19/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-18/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-18/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-16/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-14/005",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-14/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-13/002",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-12/006",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-11/002",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-10/002",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3001/2020-08-07/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3001/2020-08-05/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_038/2020-08-01/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_038/2020-07-31/001",
        "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_038/2020-07-30/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-30/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-29/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-29/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-28/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-28/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-27/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-27/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-19/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-18/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_042/2020-07-17/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-17/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-16/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_042/2020-07-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-15/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-14/001",
        "/home/nico/Downloads/FlatIron/hoferlab/Subjects/SWC_042/2020-07-13/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-13/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_021/2020-06-16/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_024/2020-06-11/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_024/2020-06-09/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_024/2020-06-08/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-07/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-05/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-04/002",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-03/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-28/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-26/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-25/002",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-24/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-23/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-22/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-21/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-20/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-19/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_011/2020-03-23/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_011/2020-03-22/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-08/2020-03-20/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-08/2020-03-19/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-08/2020-03-18/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_018/2020-03-16/002",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-15/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-14/002",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-14/003",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-13/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-13/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-12/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-12/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-12/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-11/004",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-10/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-09/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-08/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-07/002",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-07/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-07/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-06/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-06/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-06/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-05/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_008/2020-03-05/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-05/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_008/2020-03-04/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-04/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-04/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_008/2020-03-03/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-03/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-03/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-02/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-02-29/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH006/2020-02-28/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-28/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH006/2020-02-27/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-02-27/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-26/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH006/2020-02-26/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH006/2020-02-25/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-25/002",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-24/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-11/2020-02-21/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL052/2020-02-21/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-11/2020-02-20/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL052/2020-02-20/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL052/2020-02-19/002",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-11/2020-02-19/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-11/2020-02-18/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL052/2020-02-18/002",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL055/2020-02-18/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL052/2020-02-17/003",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-11/2020-02-17/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL055/2020-02-17/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL054/2020-02-12/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL054/2020-02-11/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-10/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-02-08/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH011/2020-02-07/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-02-07/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-07/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH011/2020-02-06/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-06/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-05/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-02-05/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-02-04/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-02-03/003",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-02-03/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-02-02/002",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-02-01/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-02-01/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-02-01/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-01-31/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-31/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-01-31/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH012/2020-01-31/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-31/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-30/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-01-30/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-01-30/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-30/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-01-29/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-29/008",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-29/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-01-28/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-28/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-28/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-28/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-27/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-27/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-27/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-26/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-12/2020-01-24/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-24/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2240/2020-01-24/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2240/2020-01-23/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-12/2020-01-23/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-23/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-12/2020-01-22/001",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-22/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2240/2020-01-22/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-22/002",
        "/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-21/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2245/2020-01-21/002",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-21/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2240/2020-01-21/001",
        "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-12/2020-01-20/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_1896/2020-01-17/001",
        "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_001/2020-01-15/001",
        "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL049/2020-01-09/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_1898/2019-12-11/002",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_1898/2019-12-10/007",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_1897/2019-12-06/001",
        "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_1897/2019-12-05/001",
        "/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-29/001",
        "/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-27/001",
        "/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-26/001",
        "/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-25/001",
    ]
    all_failing_errors = []
    error_types = {
        "gabor_stim": [
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-19/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-26/2020-08-18/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-18/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_019/2020-08-14/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-21/2020-08-12/006"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-27/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_014/2020-07-13/001"),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-07/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-28/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-12/001"),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-10/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-05/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-03/001"),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH006/2020-02-27/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-02-04/001"),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-02-03/003"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-31/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-27/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-12/2020-01-24/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2240/2020-01-23/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-12/2020-01-23/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_1897/2019-12-06/001"
                ),
                AssertionError("Wrong number of Gabor stimuli detected"),
            ),
        ],
        "spacer": [
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL027/2020-08-20/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3001/2020-08-05/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-28/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-01-29/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_2241/2020-01-29/008"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-30/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-26/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-03-06/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL059/2020-03-04/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_009/2020-02-29/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-28/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/NYU-11/2020-02-21/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-10/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (2.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-02-07/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-06/001"
                ),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-27/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-23/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (4.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-29/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (7.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-27/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (6.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-26/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (7.0)"
                ),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/cortexlab/Subjects/KS004/2019-09-25/001"),
                ValueError(
                    "The number of expected spacer (3) is different than the one found on the raw trace (7.0)"
                ),
            ),
        ],
        "sound_onsets": [
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-07/001"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL051/2020-02-05/001"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-02-03/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-02-02/002"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-02-01/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-02-01/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-02-01/001"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-01-31/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-01-31/001"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-01-30/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL053/2020-01-30/001"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-30/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-29/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-27/001"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-26/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-24/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-22/002"
                ),
                AssertionError("Wrong number of sound ONSETS"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-21/001"),
                AssertionError("Wrong number of sound ONSETS"),
            ),
        ],
        "valve_onsets": [
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL029/2020-07-27/001"
                ),
                AssertionError("Wrong number of valve ONSET times"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL_019/2020-03-12/001"
                ),
                AssertionError("Wrong number of valve ONSET times"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_013/2020-03-07/001"),
                AssertionError("Wrong number of valve ONSET times"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH011/2020-02-06/001"
                ),
                AssertionError("Wrong number of valve ONSET times"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/angelakilab/Subjects/SH012/2020-01-31/001"
                ),
                AssertionError("Wrong number of valve ONSET times"),
            ),
        ],
        "misc": [
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/zadorlab/Subjects/CSH_ZAD_017/2020-06-05/001"
                ),
                IndexError("index 0 is out of bounds for axis 0 with size 0"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_010/2020-01-22/001"),
                IndexError("index 0 is out of bounds for axis 0 with size 0"),
            ),
            (
                PosixPath("/home/nico/Downloads/FlatIron/danlab/Subjects/DY_011/2020-01-28/001"),
                ValueError("first array argument cannot be empty"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/mainenlab/Subjects/ZM_3003/2020-07-29/001"
                ),
                ValueError("first array argument cannot be empty"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-14/003"
                ),
                ValueError("zero-size array to reduction operation minimum which has no identity"),
            ),
            (
                PosixPath(
                    "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL047/2020-01-21/001"
                ),
                AssertionError("Some valve outputs are longer or shorter than others"),
            ),
        ],
    }
    print([(k, len(v)) for k, v in error_types.items()])
    for s in error_types["gabor_stim"]:
        try:
            extract_replay_plot(s[0])
        except:
            continue
    # eid = eids[random.randint(0, len(eids))]
    # print(eid)
    # session_paths = []
    # for i, eid in enumerate(eids):
    #     try:
    #         local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)
    #         session_paths.append(alf.io.get_session_path(local_paths[0]))
    #         print(f"{i+1}/{len(eids)}")
    #     except BaseException as e:
    #         print(f"{i+1}/{len(eids)} - Failed session: {eid}")

    # session_path = "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-25/001"

    # session_path = session_paths[random.randint(0, len(eids))]
    # session_path = "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_045/2020-08-25/001"
    # extract_replay_plot(session_path)

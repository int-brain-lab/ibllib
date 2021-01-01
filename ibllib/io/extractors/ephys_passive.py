#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, September 7th 2020, 11:51:17 am
import json
import logging
from pathlib import Path
from typing import Tuple

import ibllib.io.raw_data_loaders as rawio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ibllib.io.extractors import ephys_fpga
from ibllib.io.extractors.base import BaseExtractor
from ibllib.io.extractors.passive_plotting import (
    plot_audio_times,
    plot_gabor_times,
    plot_passive_periods,
    plot_rfmapping,
    plot_stims_times,
    plot_sync_channels,
    plot_valve_times,
)

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
def _load_passive_session_fixtures(session_path: str) -> dict:
    """load_passive_session_fixtures Loads corresponding ephys session fixtures

    :param session_path: the path to a session
    :type session_path: str
    :return: position contrast phase delays and stim id's
    :rtype: dict
    """
    settings = rawio.load_settings(session_path)
    ses_nb = settings["SESSION_ORDER"][settings["SESSION_IDX"]]
    path_fixtures = Path(ephys_fpga.__file__).parent.joinpath("ephys_sessions")

    fixture = {
        "pcs": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_pcs.npy")),
        "delays": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimDelays.npy")),
        "ids": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimIDs.npy")),
    }

    return fixture


def _load_passive_stim_meta() -> dict:
    """load_passive_stim_meta Loads the passive protocol metadata

    :return: metadata about passive protocol stimulus presentation
    :rtype: dict
    """
    path_fixtures = Path(ephys_fpga.__file__).parent.joinpath("ephys_sessions")
    with open(path_fixtures.joinpath("passive_stim_meta.json"), "r") as f:
        meta = json.load(f)

    return meta


# 1/3 Define start and end times of the 3 passive periods
def _get_spacer_times(spacer_template, jitter, ttl_signal, t_quiet):
    """
    Find timestamps of spacer signal.
    :param spacer_template: list of indices where ttl signal changes
    :type spacer_template: array-like
    :param jitter: jitter (in seconds) for matching ttl_signal with spacer_template
    :type jitter: float
    :param ttl_signal:
    :type ttl_signal: array-like
    :param t_quiet: seconds between spacer and next stim
    :type t_quiet: float
    :return: times of spacer onset/offset
    :rtype: n_spacer x 2 np.ndarray; first col onset times, second col offset
    """
    diff_spacer_template = np.diff(spacer_template)
    # add jitter;
    # remove extreme values
    spacer_model = jitter + diff_spacer_template[2:-2]
    # diff ttl signal to compare to spacer_model
    dttl = np.diff(ttl_signal)
    # remove diffs larger than max diff in model to clean up signal
    dttl[dttl > np.max(spacer_model)] = 0
    # convolve cleaned diff ttl signal w/ spacer model
    conv_dttl = np.correlate(dttl, spacer_model, mode="full")
    # find spacer location
    thresh = 3.0
    idxs_spacer_middle = np.where(
        (conv_dttl[1:-2] < thresh) & (conv_dttl[2:-1] > thresh) & (conv_dttl[3:] < thresh)
    )[0]
    # adjust indices for
    # - `np.where` call above
    # - length of spacer_model
    idxs_spacer_middle += 2 - int((np.floor(len(spacer_model) / 2)))
    # pull out spacer times (middle)
    ts_spacer_middle = ttl_signal[idxs_spacer_middle]
    # put beginning/end of spacer times into an array
    spacer_length = np.max(spacer_template)
    spacer_times = np.zeros(shape=(ts_spacer_middle.shape[0], 2))
    for i, t in enumerate(ts_spacer_middle):
        spacer_times[i, 0] = t - (spacer_length / 2) - t_quiet
        spacer_times[i, 1] = t + (spacer_length / 2) + t_quiet
    return spacer_times, conv_dttl


def _get_passive_spacers(session_path, sync=None, sync_map=None):
    """
    load and get spacer information, do corr to find spacer timestamps
    returns t_passive_starts, t_starts, t_ends
    """
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga.get_main_probe_sync(session_path, bin_exists=False)
    meta = _load_passive_stim_meta()
    # t_end_ephys = passive.ephysCW_end(session_path=session_path)
    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=None)
    spacer_template = (
        np.array(meta["VISUAL_STIM_0"]["ttl_frame_nums"], dtype=np.float32) / FRAME_FS
    )
    jitter = 3 / FRAME_FS  # allow for 3 screen refresh as jitter
    t_quiet = meta["VISUAL_STIM_0"]["delay_around"]
    spacer_times, _ = _get_spacer_times(
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


# 2/3 RFMapping stimuli
def _interpolate_rf_mapping_stimulus(idxs_up, idxs_dn, times, Xq, t_bin):
    """
    Interpolate stimulus presentation times to screen refresh rate to match `frames`
    :param ttl_01:
    :type ttl_01: array-like
    :param times: array of stimulus switch times
    :type times: array-like
    :param Xq: number of times (found in frames)
    :type frames: array-like
    :param t_bin: period of screen refresh rate
    :type t_bin: float
    :return: tuple of (stim_times, stim_frames)
    """

    beg_extrap_val = -10001
    end_extrap_val = -10000

    X = np.sort(np.concatenate([idxs_up, idxs_dn]))
    # make left and right extrapolations distinctive to easily find later
    Tq = np.interp(Xq, X, times, left=beg_extrap_val, right=end_extrap_val)
    # uniform spacing outside boundaries of ttl signal
    # first values
    n_beg = len(np.where(Tq == beg_extrap_val)[0])
    if 0 < n_beg < Tq.shape[0]:
        Tq[:n_beg] = times[0] - np.arange(n_beg, 0, -1) * t_bin
    # end values
    n_end = len(np.where(Tq == end_extrap_val)[0])
    if 0 < n_end < Tq.shape[0]:
        Tq[-n_end:] = times[-1] + np.arange(1, n_end + 1) * t_bin
    return Tq


def _get_id_raisefall_from_analogttl(ttl_01):
    """
    Get index of raise/fall from analog continuous TTL signal  (0-1 values)
    :param ttl_01: analog continuous TTL signal  (0-1 values)
    :return: index up (0>1), index down (1>0), number of ttl transition
    """
    # Check values are 0, 1, -1
    if not np.all(np.isin(np.unique(ttl_01), [-1, 0, 1])):
        raise ValueError("Values in input must be 0, 1, -1")
    else:
        # Find number of passage from [0 1] and [0 -1]
        d_ttl_01 = np.diff(ttl_01)
        id_up = np.where(np.logical_and(ttl_01 == 0, np.append(d_ttl_01, 0) == 1))[0]
        id_dw = np.where(np.logical_and(ttl_01 == 0, np.append(d_ttl_01, 0) == -1))[0]
        n_ttl_expected = 2 * (len(id_up) + len(id_dw))  # *2 for rise/fall of ttl pulse
        return id_up, id_dw, n_ttl_expected


def _reshape_RF(RF_file, meta_stim):
    """
    Reshape Receptive Field (RF) matrix. Take data associated to corner
    where frame2ttl placed to create TTL trace.
    :param RF_file: vector to be reshaped, containing RF info
    :param meta_stim: variable containing metadata information on RF
    :return: frames (reshaped RF), analog trace (0-1 values)
    """
    frame_array = np.fromfile(RF_file, dtype="uint8")
    y_pix, x_pix, _ = meta_stim["stim_file_shape"]
    frames = np.transpose(np.reshape(frame_array, [y_pix, x_pix, -1], order="F"), [2, 1, 0])
    ttl_trace = frames[:, 0, 0]
    # Convert values to 0,1,-1 for simplicity
    ttl_analogtrace_01 = np.zeros(np.size(ttl_trace))
    ttl_analogtrace_01[np.where(ttl_trace == 0)] = -1
    ttl_analogtrace_01[np.where(ttl_trace == 255)] = 1
    return frames, ttl_analogtrace_01


# 3/3 Replay of task stimuli
def _extract_passiveGabor_df(fttl: dict, session_path: str) -> pd.DataFrame:
    # At this stage we want to define what pulses are and not quality control them.
    # Pulses are stricty altternating with intevals
    # find min max lengths for both (we don'tknow which are pulses and which are intervals yet)
    # trim edges of pulses
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
    # Check if we missed the first stim
    if len(start_times) < NGABOR:
        first_stim_off_idx = idx_start_stims[0] - 1
        # first_stim_on_idx = first_stim_off_idx - 1
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
    fixture = _load_passive_session_fixtures(session_path)
    passiveGabor_properties = fixture["pcs"]
    passiveGabor_table = np.append(passiveGabor_intervals, passiveGabor_properties, axis=1)
    columns = ["start", "stop", "position", "contrast", "phase"]
    passiveGabor_df = pd.DataFrame(passiveGabor_table, columns=columns)
    return passiveGabor_df


def _extract_passiveValve_intervals(bpod: dict) -> np.array:
    # passiveValve.intervals
    # Get valve intervals from bpod channel
    # bpod channel should only contain valve output for passiveCW protocol
    # All high fronts == valve open times and low fronts == valve close times
    valveOn_times = bpod["times"][bpod["polarities"] > 0]
    valveOff_times = bpod["times"][bpod["polarities"] < 0]

    assert len(valveOn_times) == NVALVE, "Wrong number of valve ONSET times"
    assert len(valveOff_times) == NVALVE, "Wrong number of valve OFFSET times"
    assert len(bpod["times"]) == NVALVE * 2, "Wrong number of valve FRONTS detected"  # (40 * 2)

    # check all values are within bpod tolerance of 100µs
    assert np.allclose(
        valveOff_times - valveOn_times, valveOff_times[0] - valveOn_times[0], atol=0.0001
    ), "Some valve outputs are longer or shorter than others"

    return np.array([(x, y) for x, y in zip(valveOn_times, valveOff_times)])


def _extract_passiveAudio_intervals(audio: dict) -> Tuple[np.array, np.array]:
    # Get Tone and Noise cue intervals

    # Get all sound onsets and offsets
    soundOn_times = audio["times"][audio["polarities"] > 0]
    soundOff_times = audio["times"][audio["polarities"] < 0]
    # Check they are the correct number
    assert len(soundOn_times) == NTONES + NNOISES, "Wrong number of sound ONSETS"
    assert len(soundOff_times) == NTONES + NNOISES, "Wrong number of sound OFFSETS"

    diff = soundOff_times - soundOn_times
    # Tone is ~100ms so check if diff < 0.3
    toneOn_times = soundOn_times[diff <= 0.3]
    toneOff_times = soundOff_times[diff <= 0.3]
    # Noise is ~500ms so check if diff > 0.3
    noiseOn_times = soundOn_times[diff > 0.3]
    noiseOff_times = soundOff_times[diff > 0.3]

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


# ------------------------------------------------------------------
def extract_passive_periods(
    session_path: str, sync: dict = None, sync_map: dict = None
) -> pd.DataFrame:
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga.get_main_probe_sync(session_path, bin_exists=False)

    t_start_passive, t_starts, t_ends = _get_passive_spacers(
        session_path, sync=sync, sync_map=sync_map
    )
    t_starts_col = np.insert(t_starts, 0, t_start_passive)
    t_ends_col = np.insert(t_ends, 0, t_ends[-1])
    # tpassive_protocol = [t_start_passive, t_ends[-1]]
    # tspontaneous = [t_starts[0], t_ends[0]]
    # trfm = [t_starts[1], t_ends[1]]
    # treplay = [t_starts[2], t_ends[2]]
    passivePeriods_df = pd.DataFrame(
        [t_starts_col, t_ends_col],
        index=["start", "stop"],
        columns=["passiveProtocol", "spontaneousActivity", "RFM", "taskReplay"],
    )
    return passivePeriods_df  # _ibl_passivePeriods.intervalsTable.csv


def extract_rfmapping(
    session_path: str, sync: dict = None, sync_map: dict = None, trfm: np.array = None
) -> Tuple[np.array, np.array]:
    meta = _load_passive_stim_meta()
    mkey = (
        "VISUAL_STIM_"
        + {v: k for k, v in meta["VISUAL_STIMULI"].items()}["receptive_field_mapping"]
    )
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga.get_main_probe_sync(session_path, bin_exists=False)
    if trfm is None:
        passivePeriods_df = extract_passive_periods(session_path, sync=sync, sync_map=sync_map)
        trfm = passivePeriods_df.RFM.values

    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=trfm[0], tmax=trfm[1])

    RF_file = Path().joinpath(session_path, "raw_passive_data", "_iblrig_RFMapStim.raw.bin")
    passiveRFM_frames, RF_ttl_trace = _reshape_RF(RF_file=RF_file, meta_stim=meta[mkey])
    rf_id_up, rf_id_dw, RF_n_ttl_expected = _get_id_raisefall_from_analogttl(RF_ttl_trace)
    meta[mkey]["ttl_num"] = RF_n_ttl_expected
    rf_times_on_idx = np.where(np.diff(fttl["times"]) < 1)[0]
    rf_times_off_idx = rf_times_on_idx + 1
    RF_times = fttl["times"][np.sort(np.concatenate([rf_times_on_idx, rf_times_off_idx]))]
    RF_times_1 = RF_times[0::2]
    # Interpolate times for RF before outputting dataset
    passiveRFM_times = _interpolate_rf_mapping_stimulus(
        idxs_up=rf_id_up,
        idxs_dn=rf_id_dw,
        times=RF_times_1,
        Xq=np.arange(passiveRFM_frames.shape[0]),
        t_bin=1 / FRAME_FS,
    )

    return passiveRFM_times  # _ibl_passiveRFM.times.npy


def extract_task_replay(
    session_path: str, sync: dict = None, sync_map: dict = None, treplay: np.array = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga.get_main_probe_sync(session_path, bin_exists=False)

    if treplay is None:
        passivePeriods_df = extract_passive_periods(session_path, sync=sync, sync_map=sync_map)
        treplay = passivePeriods_df.taskReplay.values

    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=treplay[0])
    passiveGabor_df = _extract_passiveGabor_df(fttl, session_path)

    bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=treplay[0])
    passiveValve_intervals = _extract_passiveValve_intervals(bpod)

    audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=treplay[0])
    passiveTone_intervals, passiveNoise_intervals = _extract_passiveAudio_intervals(audio)

    passiveStims_df = np.concatenate(
        [passiveValve_intervals, passiveTone_intervals, passiveNoise_intervals], axis=1
    )
    columns = ["valveOn", "valveOff", "toneOn", "toneOff", "noiseOn", "noiseOff"]
    passiveStims_df = pd.DataFrame(passiveStims_df, columns=columns)
    return (
        passiveGabor_df,
        passiveStims_df,
    )  # _ibl_passiveGabor.table.csv, _ibl_passiveStims.times_table.csv


def extract_replay_debug(
    session_path: str,
    sync: dict = None,
    sync_map: dict = None,
    treplay: np.array = None,
    ax: plt.axes = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load sessions sync channels, map
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga.get_main_probe_sync(session_path, bin_exists=False)

    if treplay is None:
        passivePeriods_df = extract_passive_periods(session_path, sync=sync, sync_map=sync_map)
        treplay = passivePeriods_df.taskReplay.values

    if ax is None:
        f, ax = plt.subplots(1, 1)

    f = ax.figure
    f.suptitle("/".join(str(session_path).split("/")[-5:]))
    plot_sync_channels(sync=sync, sync_map=sync_map, ax=ax)

    passivePeriods_df = extract_passive_periods(session_path, sync=sync, sync_map=sync_map)
    treplay = passivePeriods_df.taskReplay.values

    plot_passive_periods(passivePeriods_df, ax=ax)

    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=treplay[0])
    passiveGabor_df = _extract_passiveGabor_df(fttl, session_path)
    plot_gabor_times(passiveGabor_df, ax=ax)

    bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=treplay[0])
    passiveValve_intervals = _extract_passiveValve_intervals(bpod)
    plot_valve_times(passiveValve_intervals, ax=ax)

    audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=treplay[0])
    passiveTone_intervals, passiveNoise_intervals = _extract_passiveAudio_intervals(audio)
    plot_audio_times(passiveTone_intervals, passiveNoise_intervals, ax=ax)

    passiveStims_df = np.concatenate(
        [passiveValve_intervals, passiveTone_intervals, passiveNoise_intervals], axis=1
    )
    columns = ["valveOn", "valveOff", "toneOn", "toneOff", "noiseOn", "noiseOff"]
    passiveStims_df = pd.DataFrame(passiveStims_df, columns=columns)

    return (
        passiveGabor_df,
        passiveStims_df,
    )  # _ibl_passiveGabor.table.csv, _ibl_passiveStims.table.csv


# Maan passiveCWe xtractor, calls all others
class PassiveChoiceWorld(BaseExtractor):
    save_names = (
        "_ibl_passivePeriods.intervalsTable.csv",
        "_ibl_passiveRFM.times.npy",
        "_ibl_passiveGabor.table.csv",
        "_ibl_passiveStims.table.csv",
    )
    var_names = (
        "passivePeriods_df",
        "passiveRFM_times",
        "passiveGabor_df",
        "passiveStims_df",
    )

    def _extract(
        self, sync: dict = None, sync_map: dict = None, plot: bool = False, **kwargs
    ) -> tuple:
        if sync is None or sync_map is None:
            sync, sync_map = ephys_fpga.get_main_probe_sync(self.session_path, bin_exists=False)

        try:
            # Passive periods
            passivePeriods_df = extract_passive_periods(
                self.session_path, sync=sync, sync_map=sync_map
            )
            trfm = passivePeriods_df.RFM.values
            treplay = passivePeriods_df.taskReplay.values

        except BaseException as e:
            log.error(f"Failed to extract passive periods: {e}")
            passivePeriods_df = None
            trfm = None
            treplay = None
            return (None, None, None, None)

        try:
            # RFMapping
            passiveRFM_times = extract_rfmapping(
                self.session_path, sync=sync, sync_map=sync_map, trfm=trfm
            )
        except Exception as e:
            log.error(f"Failed to extract RFMapping datasets: {e}")
            passiveRFM_times = None

        try:
            (passiveGabor_df, passiveStims_df,) = extract_task_replay(
                self.session_path, sync=sync, sync_map=sync_map, treplay=treplay
            )
        except Exception as e:
            log.error(f"Failed to extract task replay stimuli: {e}")
            (passiveGabor_df, passiveStims_df,) = (
                None,
                None,
            )
        if plot:
            f, ax = plt.subplots(1, 1)
            f.suptitle("/".join(str(self.session_path).split("/")[-5:]))
            plot_sync_channels(sync=sync, sync_map=sync_map, ax=ax)
            plot_passive_periods(passivePeriods_df, ax=ax)
            plot_rfmapping(passiveRFM_times, ax=ax)
            plot_gabor_times(passiveGabor_df, ax=ax)
            plot_stims_times(passiveStims_df, ax=ax)
            plt.show()

        return (
            passivePeriods_df,  # _ibl_passivePeriods.intervalsTable.csv
            passiveRFM_times,  # _ibl_passiveRFM.times.npy
            passiveGabor_df,  # _ibl_passiveGabor.table.csv,
            passiveStims_df,  # _ibl_passiveStims.table.csv
        )


if __name__ == "__main__":
    # Working session
    session_path = "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_054/2020-10-10/001"
    # # Broken session
    #     session_path = "/home/nico/Downloads/FlatIron/integration/ephys/\
    # choice_world/KS022/2019-12-10/001"
    pcw = PassiveChoiceWorld(session_path)
    data, paths = pcw.extract(save=False)
    (
        passivePeriods_df,
        passiveRFM_times,
        passiveGabor_df,
        passiveStims_df
    ) = data
    # sp = '/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_029/2020-10-07/001'
    # extract_passive_choice_world(sp)
    # extract_passive_choice_world(sp, plot=True)
    # from oneibl.one import ONE
    # import alf.io

    # one = ONE()

    # eids = one.search(dataset_types=min_dataset_types)
    # session_paths = []
    # for i, eid in enumerate(eids):
    #     try:
    #         local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)
    #         session_paths.append(alf.io.get_session_path(local_paths[0]))
    #     except BaseException as e:
    #         print(f"{i+1}/{len(eids)} - Failed to DL session: {eid}\n\n{e}")
    #     print(f"\n\n{i+1}/{len(eids)}\n\n")

    # failed = []
    # for i, sp in enumerate(session_paths):
    #     try:
    #         extract_passive_choice_world(sp)
    #     except BaseException as e:
    #         failed.append((sp, e))
    #     print(f"\n{i+1}/{len(session_paths)}")
    # print(f"nfailed = {len(failed)} / {len(session_paths)}")

    # for s in all_session_paths[:20]:
    #     try:
    #         extract_passive_choice_world_plot(s)
    #     except BaseException as e:
    #         print(e)
    #         continue
    print(".")

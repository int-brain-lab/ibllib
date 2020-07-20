#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, July 16th 2018, 1:28:46 pm
"""
Raw Data Loader functions for PyBpod rig

Module contains one loader function per raw datafile
"""
import json
import logging
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alf.io import get_session_path
from ibllib.io import jsonable
from ibllib.misc import version

_logger = logging.getLogger('ibllib')


def trial_times_to_times(raw_trial):
    """
    Parse and convert all trial timestamps to "absolute" time.
    Float64 seconds from session start.

    0---BpodStart---TrialStart0---------TrialEnd0-----TrialStart1---TrialEnd1...0---ts0---ts1---
    tsN...absTS = tsN + TrialStartN - BpodStart

    Bpod timestamps are in microseconds (µs)
    PyBpod timestamps are is seconds (s)

    :param raw_trial: raw tiral data
    :type raw_trial: dict
    :return: trial data with modified timestamps
    :rtype: dict
    """
    ts_bs = raw_trial['behavior_data']['Bpod start timestamp']
    ts_ts = raw_trial['behavior_data']['Trial start timestamp']
    # ts_te = raw_trial['behavior_data']['Trial end timestamp']

    def convert(ts):
        return ts + ts_ts - ts_bs

    converted_events = {}
    for k, v in raw_trial['behavior_data']['Events timestamps'].items():
        converted_events.update({k: [convert(i) for i in v]})
    raw_trial['behavior_data']['Events timestamps'] = converted_events

    converted_states = {}
    for k, v in raw_trial['behavior_data']['States timestamps'].items():
        converted_states.update({k: [[convert(i) for i in x] for x in v]})
    raw_trial['behavior_data']['States timestamps'] = converted_states

    shift = raw_trial['behavior_data']['Bpod start timestamp']
    raw_trial['behavior_data']['Bpod start timestamp'] -= shift
    raw_trial['behavior_data']['Trial start timestamp'] -= shift
    raw_trial['behavior_data']['Trial end timestamp'] -= shift
    assert(raw_trial['behavior_data']['Bpod start timestamp'] == 0)
    return raw_trial


def load_bpod(session_path):
    """
    Load both settings and data from bpod (.json and .jsonable)

    :param session_path: Absolute path of session folder
    :return: dict settings and list of dicts data
    """
    return load_settings(session_path), load_data(session_path)


def load_data(session_path, time='absolute'):
    """
    Load PyBpod data files (.jsonable).

    Bpod timestamps are in microseconds (µs)
    PyBpod timestamps are is seconds (s)

    :param session_path: Absolute path of session folder
    :type session_path: str
    :return: A list of len ntrials each trial being a dictionary
    :rtype: list of dicts
    """
    if session_path is None:
        _logger.warning("No data loaded: session_path is None")
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_taskData.raw*.jsonable"), None)
    if not path:
        _logger.warning("No data loaded: could not find raw data file")
        return None
    data = jsonable.read(path)
    if time == 'absolute':
        data = [trial_times_to_times(t) for t in data]
    return data


def load_settings(session_path):
    """
    Load PyBpod Settings files (.json).

    [description]

    :param session_path: Absolute path of session folder
    :type session_path: str
    :return: Settings dictionary
    :rtype: dict
    """
    if session_path is None:
        _logger.warning("No data loaded: session_path is None")
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_taskSettings.raw*.json"), None)
    if not path:
        _logger.warning("No data loaded: could not find raw settings file")
        return None
    with open(path, 'r') as f:
        settings = json.load(f)
    if 'IBLRIG_VERSION_TAG' not in settings.keys():
        settings['IBLRIG_VERSION_TAG'] = ''
    return settings


def load_stim_position_screen(session_path):
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_stimPositionScreen.raw*.csv"), None)

    data = pd.read_csv(path, sep=',', header=None, error_bad_lines=False)
    data.columns = ['contrast', 'position', 'bns_ts']
    data[2] = pd.to_datetime(data[2])
    return data


def load_encoder_events(session_path, settings=False):
    """
    Load Rotary Encoder (RE) events raw data file.

    Assumes that a folder called "raw_behavior_data" exists in folder.

    Events number correspond to following bpod states:
    1: correct / hide_stim
    2: stim_on
    3: closed_loop
    4: freeze_error / freeze_correct

    >>> data.columns
    >>> ['re_ts',   # Rotary Encoder Timestamp (ms) 'numpy.int64'
         'sm_ev',   # State Machine Event           'numpy.int64'
         'bns_ts']  # Bonsai Timestamp (int)        'pandas.Timestamp'
        # pd.to_datetime(data.bns_ts) to work in datetimes

    :param session_path: [description]
    :type session_path: [type]
    :return: dataframe w/ 3 cols and (ntrials * 3) lines
    :rtype: Pandas.DataFrame
    """
    if session_path is None:
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_encoderEvents.raw*.ssv"), None)
    if not settings:
        settings = load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
        # auto-detect old files when version is not labeled
        with open(path) as fid:
            line = fid.readline()
        if line.startswith('Event') and 'StateMachine' in line:
            settings = {'IBLRIG_VERSION_TAG': '0.0.0'}
    if not path:
        return None
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        return _load_encoder_events_file_ge5(path)
    else:
        return _load_encoder_events_file_lt5(path)


def _load_encoder_ssv_file(file_path, **kwargs):
    file_path = Path(file_path)
    if file_path.stat().st_size == 0:
        _logger.error(f"{file_path.name} is an empty file. ")
        raise ValueError(f"{file_path.name} is an empty file. ABORT EXTRACTION. ")
    return pd.read_csv(file_path, sep=' ', header=None, error_bad_lines=False, **kwargs)


def _load_encoder_positions_file_lt5(file_path):
    """
    File loader without the session overhead
    :param file_path:
    :return: dataframe of encoder events
    """
    data = _load_encoder_ssv_file(file_path,
                                  names=['_', 're_ts', 're_pos', 'bns_ts', '__'],
                                  usecols=['re_ts', 're_pos', 'bns_ts'])
    return _groom_wheel_data_lt5(data, label='_iblrig_encoderPositions.raw.ssv', path=file_path)


def _load_encoder_positions_file_ge5(file_path):
    """
    File loader without the session overhead
    :param file_path:
    :return: dataframe of encoder events
    """
    data = _load_encoder_ssv_file(file_path,
                                  names=['re_ts', 're_pos', '_'],
                                  usecols=['re_ts', 're_pos'])
    return _groom_wheel_data_ge5(data, label='_iblrig_encoderPositions.raw.ssv', path=file_path)


def _load_encoder_events_file_lt5(file_path):
    """
    File loader without the session overhead
    :param file_path:
    :return: dataframe of encoder events
    """
    data = _load_encoder_ssv_file(file_path,
                                  names=['_', 're_ts', '__', 'sm_ev', 'bns_ts', '___'],
                                  usecols=['re_ts', 'sm_ev', 'bns_ts'])
    return _groom_wheel_data_lt5(data, label='_iblrig_encoderEvents.raw.ssv', path=file_path)


def _load_encoder_events_file_ge5(file_path):
    """
    File loader without the session overhead
    :param file_path:
    :return: dataframe of encoder events
    """
    data = _load_encoder_ssv_file(file_path,
                                  names=['re_ts', 'sm_ev', '_'],
                                  usecols=['re_ts', 'sm_ev'])
    return _groom_wheel_data_ge5(data, label='_iblrig_encoderEvents.raw.ssv', path=file_path)


def load_encoder_positions(session_path, settings=False):
    """
    Load Rotary Encoder (RE) positions from raw data file within a session path.

    Assumes that a folder called "raw_behavior_data" exists in folder.
    Positions are RE ticks [-512, 512] == [-180º, 180º]
    0 == trial stim init position
    Positive nums are rightwards movements (mouse) or RE CW (mouse)

    Variable line number, depends on movements.

    Raw datafile Columns:
        Position, RE timestamp, RE Position, Bonsai Timestamp

    Position is always equal to 'Position' so this column was dropped.

    >>> data.columns
    >>> ['re_ts',   # Rotary Encoder Timestamp (ms)     'numpy.int64'
         're_pos',  # Rotary Encoder position (ticks)   'numpy.int64'
         'bns_ts']  # Bonsai Timestamp                  'pandas.Timestamp'
        # pd.to_datetime(data.bns_ts) to work in datetimes

    :param session_path: Absolute path of session folder
    :type session_path: str
    :return: dataframe w/ 3 cols and N positions
    :rtype: Pandas.DataFrame
    """
    if session_path is None:
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
    if not settings:
        settings = load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
        # auto-detect old files when version is not labeled
        with open(path) as fid:
            line = fid.readline()
        if line.startswith('Position'):
            settings = {'IBLRIG_VERSION_TAG': '0.0.0'}
    if not path:
        _logger.warning("No data loaded: could not find raw encoderPositions file")
        return None
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        return _load_encoder_positions_file_ge5(path)
    else:
        return _load_encoder_positions_file_lt5(path)


def load_encoder_trial_info(session_path):
    """
    Load Rotary Encoder trial info from raw data file.

    Assumes that a folder calles "raw_behavior_data" exists in folder.

    NOTE: Last trial probably inexistent data (Trial info is sent on trial start
    and data is only saved on trial exit...) max(trialnum) should be N+1 if N
    is the amount of trial data saved.

    Raw datafile Columns:

    >>> data.columns
    >>> ['trial_num',     # Trial Number                     'numpy.int64'
         'stim_pos_init', # Initial position of visual stim  'numpy.int64'
         'stim_contrast', # Contrast of visual stimulus      'numpy.float64'
         'stim_freq',     # Frequency of gabor patch         'numpy.float64'
         'stim_angle',    # Angle of Gabor 0 = Vertical      'numpy.float64'
         'stim_gain',     # Wheel gain (mm/º of stim)        'numpy.float64'
         'stim_sigma',    # Size of patch                    'numpy.float64'
         'stim_phase',    # Phase of gabor                    'numpy.float64'
         'bns_ts' ]       # Bonsai Timestamp                 'pandas.Timestamp'
        # pd.to_datetime(data.bns_ts) to work in datetimes

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :return: dataframe w/ 9 cols and ntrials lines
    :rtype: Pandas.DataFrame
    """
    if session_path is None:
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_encoderTrialInfo.raw*.ssv"), None)
    if not path:
        return None
    data = pd.read_csv(path, sep=' ', header=None)
    data = data.drop([9], axis=1)
    data.columns = ['trial_num', 'stim_pos_init', 'stim_contrast', 'stim_freq',
                    'stim_angle', 'stim_gain', 'stim_sigma', 'stim_phase', 'bns_ts']
    # return _groom_wheel_data_lt5(data, label='_iblrig_encoderEvents.raw.ssv', path=path)
    return data


def load_ambient_sensor(session_path):
    """
    Load Ambient Sensor data from session.

    Probably could be extracted to DatasetTypes:
    _ibl_trials.temperature_C, _ibl_trials.airPressure_mb,
    _ibl_trials.relativeHumidity
    Returns a list of dicts one dict per trial.
    dict keys are:
    dict_keys(['Temperature_C', 'AirPressure_mb', 'RelativeHumidity'])

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :return: list of dicts
    :rtype: list
    """
    if session_path is None:
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_ambientSensorData.raw*.jsonable"), None)
    if not path:
        return None
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_mic(session_path):
    """
    Load Microphone wav file to np.array of len nSamples

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :return: An array of values of the sound waveform
    :rtype: numpy.array
    """
    if session_path is None:
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_micData.raw*.wav"), None)
    if not path:
        return None
    fp = wave.open(path)
    nchan = fp.getnchannels()
    N = fp.getnframes()
    dstr = fp.readframes(N * nchan)
    data = np.frombuffer(dstr, np.int16)
    data = np.reshape(data, (-1, nchan))
    return data


def _clean_wheel_dataframe(data, label, path):
    if np.any(data.isna()):
        _logger.warning(label + ' has missing/incomplete records \n %s', path)
    # first step is to re-interpret as numeric objects if not already done
    for col in data.columns:
        if data[col].dtype == np.object and col not in ['bns_ts']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    # then drop Nans and duplicates
    data.dropna(inplace=True)
    data.drop_duplicates(keep='first', inplace=True)
    data.reset_index(inplace=True)
    # handle the clock resets when microseconds exceed uint32 max value
    drop_first = False
    data['re_ts'] = data['re_ts'].astype(np.double, copy=False)
    if any(np.diff(data['re_ts']) < 0):
        ind = np.where(np.diff(data['re_ts']) < 0)[0]
        for i in ind:
            # the first sample may be corrupt, in this case throw away
            if i <= 1:
                drop_first = i
                _logger.warning(label + ' rotary encoder positions timestamps'
                                        ' first sample corrupt ' + str(path))
            # if it's an uint32 wraparound, the diff should be close to 2 ** 32
            elif 32 - np.log2(data['re_ts'][i] - data['re_ts'][i + 1]) < 0.2:
                data.loc[i + 1:, 're_ts'] = data.loc[i + 1:, 're_ts'] + 2 ** 32
            # there is also the case where 2 positions are swapped and need to be swapped back

            elif data['re_ts'][i] > data['re_ts'][i + 1] > data['re_ts'][i - 1]:
                _logger.warning(label + ' rotary encoder timestamps swapped at index: ' +
                                str(i) + '  ' + str(path))
                a, b = data.iloc[i].copy(), data.iloc[i + 1].copy()
                data.iloc[i], data.iloc[i + 1] = b, a
            # if none of those 3 cases apply, raise an error
            else:
                _logger.error(label + ' Rotary encoder timestamps are not sorted.' + str(path))
                data.sort_values('re_ts', inplace=True)
                data.reset_index(inplace=True)
    if drop_first is not False:
        data.drop(data.loc[:drop_first].index, inplace=True)
        data = data.reindex()
    return data


def _groom_wheel_data_lt5(data, label='file ', path=''):
    """
    The whole purpose of this function is to account for variability and corruption in
    the wheel position files. There are many possible errors described below, but
    nothing excludes getting new ones.
    """
    data = _clean_wheel_dataframe(data, label, path)
    data.drop(data.loc[data.bns_ts.apply(len) != 33].index, inplace=True)
    # check if the time scale is in ms
    sess_len_sec = (datetime.strptime(data['bns_ts'].iloc[-1][:25], '%Y-%m-%dT%H:%M:%S.%f') -
                    datetime.strptime(data['bns_ts'].iloc[0][:25], '%Y-%m-%dT%H:%M:%S.%f')).seconds
    if data['re_ts'].iloc[-1] / (sess_len_sec + 1e-6) < 1e5:  # should be 1e6 normally
        _logger.warning('Rotary encoder reset logs events in ms instead of us: ' +
                        'RE firmware needs upgrading and wheel velocity is potentially inaccurate')
        data['re_ts'] = data['re_ts'] * 1000
    return data


def _groom_wheel_data_ge5(data, label='file ', path=''):
    """
    The whole purpose of this function is to account for variability and corruption in
    the wheel position files. There are many possible errors described below, but
    nothing excludes getting new ones.
    """
    data = _clean_wheel_dataframe(data, label, path)
    # check if the time scale is in ms
    if (data['re_ts'].iloc[-1] - data['re_ts'].iloc[0]) / 1e6 < 20:
        _logger.warning('Rotary encoder reset logs events in ms instead of us: ' +
                        'RE firmware needs upgrading and wheel velocity is potentially inaccurate')
        data['re_ts'] = data['re_ts'] * 1000
    return data


def save_bool(save, dataset_type):
    if isinstance(save, bool):
        out = save
    elif isinstance(save, list):
        out = (dataset_type in save) or (Path(dataset_type).stem in save)
    if out:
        _logger.debug('extracting' + dataset_type)
    return out


def sync_trials_robust(t0, t1, diff_threshold=0.001, drift_threshold_ppm=200, max_shift=5,
                       return_index=False):
    """
    Attempts to find matching timestamps in 2 time-series that have an offset, are drifting,
    and are most likely incomplete: sizes don't have to match, some pulses may be missing
    in any series.
    Only works with irregular time series as it relies on the derivative to match sync.
    :param t0:
    :param t1:
    :param diff_threshold:
    :param drift_threshold_ppm: (150)
    :param max_shift: (200)
    :param return_index (False)
    :return:
    """
    nsync = min(t0.size, t1.size)
    dt0 = np.diff(t0)
    dt1 = np.diff(t1)
    ind = np.zeros_like(dt0) * np.nan
    i0 = 0
    i1 = 0
    cdt = np.nan  # the current time difference between the two series to compute drift
    while i0 < (nsync - 1):
        # look in the next max_shift events the ones whose derivative match
        isearch = np.arange(i1, min(max_shift + i1, dt1.size))
        dec = np.abs(dt0[i0] - dt1[isearch]) < diff_threshold
        # another constraint is to check the dt for the maximum drift
        if ~np.isnan(cdt):
            drift_ppm = np.abs((cdt - (t0[i0] - t1[isearch])) / dt1[isearch]) * 1e6
            dec = np.logical_and(dec, drift_ppm <= drift_threshold_ppm)
        # if one is found
        if np.any(dec):
            ii1 = np.where(dec)[0][0]
            ind[i0] = i1 + ii1
            i1 += ii1 + 1
            cdt = t0[i0 + 1] - t1[i1 + ii1]
        i0 += 1
    it0 = np.where(~np.isnan(ind))[0]
    it1 = ind[it0].astype(np.int)
    ind0 = np.unique(np.r_[it0, it0 + 1])
    ind1 = np.unique(np.r_[it1, it1 + 1])
    if return_index:
        return t0[ind0], t1[ind1], ind0, ind1
    else:
        return t0[ind0], t1[ind1]


def get_task_extractor_type(task_name):
    """
    Splits the task name according to naming convention:
    -   ignores everything
    _iblrig_tasks_biasedChoiceWorld3.7.0 returns "biased"
    _iblrig_tasks_trainingChoiceWorld3.6.0 returns "training'
    :param task_name:
    :return: one of ['biased', 'habituation', 'training', 'ephys', 'mock_ephys', 'sync_ephys']
    """
    if isinstance(task_name, Path):
        try:
            settings = load_settings(get_session_path(task_name))
        except json.decoder.JSONDecodeError:
            return
        if settings:
            task_name = settings.get('PYBPOD_PROTOCOL', None)
        else:
            return
    if '_biasedChoiceWorld' in task_name:
        return 'biased'
    elif 'biasedScanningChoiceWorld' in task_name:
        return 'biased'
    elif 'biasedVisOffChoiceWorld' in task_name:
        return 'biased'
    elif '_habituationChoiceWorld' in task_name:
        return 'habituation'
    elif '_trainingChoiceWorld' in task_name:
        return 'training'
    elif 'ephysChoiceWorld' in task_name:
        return 'ephys'
    elif 'ephysMockChoiceWorld' in task_name:
        return 'mock_ephys'
    elif task_name and task_name.startswith('_iblrig_tasks_ephys_certification'):
        return 'sync_ephys'


def get_session_extractor_type(session_path):
    """
    From a session path, loads the settings file, finds the task and checks if extractors exist
    task names examples:
    :param session_path:
    :return: bool
    """
    settings = load_settings(session_path)
    if settings is None:
        _logger.error(f'ABORT: No data found in "raw_behavior_data" folder {session_path}')
        return False
    extractor_type = get_task_extractor_type(settings['PYBPOD_PROTOCOL'])
    if extractor_type:
        return extractor_type
    else:
        _logger.warning(str(session_path) +
                        f" No extractors were found for {extractor_type} ChoiceWorld")
        return False


def load_bpod_fronts(session_path: str, data: dict = False) -> list:
    """load_bpod_fronts
    Loads BNC1 and BNC2 bpod channels times and polarities from session_path

    :param session_path: a valid session_path
    :type session_path: str
    :param data: pre-loaded raw data dict, defaults to False
    :type data: dict, optional
    :return: List of dicts BNC1 and BNC2 {"times": np.array, "polarities":np.array}
    :rtype: list
    """
    if not data:
        data = load_data(session_path)

    BNC1_fronts = np.array([[np.nan, np.nan]])
    BNC2_fronts = np.array([[np.nan, np.nan]])
    for tr in data:
        BNC1_fronts = np.append(
            BNC1_fronts,
            np.array(
                [
                    [x, 1]
                    for x in tr["behavior_data"]["Events timestamps"].get("BNC1High", [np.nan])
                ]
            ),
            axis=0,
        )
        BNC1_fronts = np.append(
            BNC1_fronts,
            np.array(
                [
                    [x, -1]
                    for x in tr["behavior_data"]["Events timestamps"].get("BNC1Low", [np.nan])
                ]
            ),
            axis=0,
        )
        BNC2_fronts = np.append(
            BNC2_fronts,
            np.array(
                [
                    [x, 1]
                    for x in tr["behavior_data"]["Events timestamps"].get("BNC2High", [np.nan])
                ]
            ),
            axis=0,
        )
        BNC2_fronts = np.append(
            BNC2_fronts,
            np.array(
                [
                    [x, -1]
                    for x in tr["behavior_data"]["Events timestamps"].get("BNC2Low", [np.nan])
                ]
            ),
            axis=0,
        )

    BNC1_fronts = BNC1_fronts[1:, :]
    BNC1_fronts = BNC1_fronts[BNC1_fronts[:, 0].argsort()]
    BNC2_fronts = BNC2_fronts[1:, :]
    BNC2_fronts = BNC2_fronts[BNC2_fronts[:, 0].argsort()]

    BNC1 = {"times": BNC1_fronts[:, 0], "polarities": BNC1_fronts[:, 1]}
    BNC2 = {"times": BNC2_fronts[:, 0], "polarities": BNC2_fronts[:, 1]}

    return [BNC1, BNC2]


def get_port_events(trial: dict, name: str = '') -> list:
    """get_port_events
    Return all event timestamps from bpod raw data trial that match 'name'
    --> looks in trial['behavior_data']['Events timestamps']

    :param trial: raw trial dict
    :type trial: dict
    :param name: name of event, defaults to ''
    :type name: str, optional
    :return: Sorted list of event timestamps
    :rtype: list
    TODO: add polarities?
    """
    out: list = []
    events = trial['behavior_data']['Events timestamps']
    for k in events:
        if name in k:
            out.extend(events[k])
    out = sorted(out)

    return out

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, July 16th 2018, 1:28:46 pm
"""
Raw Data Loader functions for PyBpod rig

Module contains one loader function per raw datafile
"""
import json
import wave
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from ibllib.io import jsonable
from ibllib.misc import version


logger_ = logging.getLogger('ibllib')


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
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_taskData.raw*.jsonable"), None)
    if not path:
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
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_taskSettings.raw*.json"), None)
    if not path:
        return None
    with open(path, 'r') as f:
        settings = json.load(f)
    if 'IBLRIG_VERSION_TAG' not in settings.keys():
        settings['IBLRIG_VERSION_TAG'] = ''
    return settings


def _load_encoder_events_file_lt5(file_path):
    """
    File loader without the session overhead
    :param file_path:
    :return: dataframe of encoder events
    """
    if file_path.stat().st_size == 0:
        logger_.error("_iblrig_encoderEvents.raw.ssv is an empty file. ")
        raise ValueError("_iblrig_encoderEvents.raw.ssv is an empty file. ABORT EXTRACTION. ")
    data = pd.read_csv(file_path, sep=' ', header=None, error_bad_lines=False)
    data = data.drop([0, 2, 5], axis=1)
    data.columns = ['re_ts', 'sm_ev', 'bns_ts']
    return _groom_wheel_data_lt5(data, label='_iblrig_encoderEvents.raw.ssv', path=file_path)


def _load_encoder_events_file_ge5(file_path):
    """
    File loader without the session overhead
    :param file_path:
    :return: dataframe of encoder events
    """
    if file_path.stat().st_size == 0:
        logger_.error("_iblrig_encoderEvents.raw.ssv is an empty file. ")
        raise ValueError("_iblrig_encoderEvents.raw.ssv is an empty file. ABORT EXTRACTION. ")
    data = pd.read_csv(file_path, sep=' ', header=None, error_bad_lines=False)
    data = data.drop([2], axis=1)
    data.columns = ['re_ts', 'sm_ev']
    return _groom_wheel_data_ge5(data, label='_iblrig_encoderEvents.raw.ssv', path=file_path)


def load_encoder_events(session_path, settings=False):
    """
    Load Rotary Encoder (RE) events raw data file.

    Assumes that a folder called "raw_behavior_data" exists in folder.

    On each trial the RE sends 3 events to Bonsai 1 - meaning trial start/turn
    off the stim; 2 - meaning show the current trial stimulus; and 3 - meaning
    begin the closed loop making the stim move with the RE. These events are
    triggered by the state machine in the corrensponding states: trial_start,
    stim_on, closed_loop

    Raw datafile Columns:
        Event, RE timestamp, Source, data, Bonsai Timestamp

    Event is always equal 'Event' Source is always equal 'StateMachine'. For
    this reason these columns are dropped.

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

    if not path:
        return None
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        return _load_encoder_events_file_ge5(path)
    else:
        return _load_encoder_events_file_lt5(path)


def _load_encoder_positions_file_lt5(file_path):
    # file loader without the session overhead
    if file_path.stat().st_size == 0:
        logger_.error("_iblrig_encoderPositions.raw.ssv is an empty file. ")
        raise ValueError("_iblrig_encoderPositions.raw.ssv is an empty file. ABORT EXTRACTION. ")
    data = pd.read_csv(file_path, sep=' ', header=None, error_bad_lines=False)
    data = data.drop([0, 4], axis=1)
    data.columns = ['re_ts', 're_pos', 'bns_ts']
    return _groom_wheel_data_lt5(data, label='_iblrig_encoderPositions.raw.ssv', path=file_path)


def _load_encoder_positions_file_ge5(file_path):
    # file loader without the session overhead
    if file_path.stat().st_size == 0:
        logger_.error("_iblrig_encoderPositions.raw.ssv is an empty file. ")
        raise ValueError("_iblrig_encoderPositions.raw.ssv is an empty file. ABORT EXTRACTION. ")
    data = pd.read_csv(file_path, sep=' ', header=None, error_bad_lines=False)
    data = data.drop([2], axis=1)
    data.columns = ['re_ts', 're_pos']
    return _groom_wheel_data_ge5(data, label='_iblrig_encoderPositions.raw.ssv', path=file_path)


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

    :param session_path: Absoulte path of session folder
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

    if not path:
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
         'bns_ts' ]       # Bonsai Timestamp                 'pandas.Timestamp'
        # pd.to_datetime(data.bns_ts) to work in datetimes

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :return: dataframe w/ 8 cols and ntrials lines
    :rtype: Pandas.DataFrame
    """
    if session_path is None:
        return
    path = Path(session_path).joinpath("raw_behavior_data")
    path = next(path.glob("_iblrig_encoderTrialInfo.raw*.ssv"), None)
    if not path:
        return None
    data = pd.read_csv(path, sep=' ', header=None)
    data = data.drop([8], axis=1)
    data.columns = ['trial_num', 'stim_pos_init', 'stim_contrast', 'stim_freq',
                    'stim_angle', 'stim_gain', 'stim_sigma', 'bns_ts']
    return _groom_wheel_data_lt5(data, label='_iblrig_encoderEvents.raw.ssv', path=path)


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


def _groom_wheel_data_lt5(data, label='file ', path=''):
    """
    The whole purpose of this function is to account for variability and corruption in
    the wheel position files. There are many possible errors described below, but
    nothing excludes getting new ones.
    """
    # sometimes the text file is cropped
    if np.any(data.isna()):
        logger_.warning(label + ' has missing/incomplete records \n %s', path)
    data.dropna(inplace=True)
    data.drop(data.loc[data.bns_ts.apply(len) != 33].index, inplace=True)
    data.drop_duplicates(keep='first', inplace=True)
    data.reset_index(inplace=True)
    # handle the clock resets when microseconds exceed uint32 max value
    drop_first = False
    if any(np.diff(data['re_ts']) < 0):
        ind = np.where(np.diff(data['re_ts']) < 0)[0]
        for i in ind:
            # the first sample may be corrupt, in this case throw away
            if i <= 1:
                drop_first = i
                logger_.warning(label + ' rotary encoder positions timestamps'
                                        ' first sample corrupt ' + str(path))
            # if it's an uint32 wraparound, the diff should be close to 2 ** 32
            elif 32 - np.log2(data['re_ts'][i] - data['re_ts'][i + 1]) < 0.2:
                data.loc[i + 1:, 're_ts'] = data.loc[i + 1:, 're_ts'] + 2 ** 32
            # there is also the case where 2 positions are swapped and need to be swapped back

            elif data['re_ts'][i] > data['re_ts'][i + 1] > data['re_ts'][i - 1]:
                logger_.warning(label + ' rotary encoder timestamps swapped at index: ' +
                                str(i) + '  ' + str(path))
                a, b = data.iloc[i].copy(), data.iloc[i + 1].copy()
                data.iloc[i], data.iloc[i + 1] = b, a
            # if none of those 3 cases apply, raise an error
            else:
                logger_.error(label + ' Rotary encoder timestamps are not sorted.' + str(path))
                data.sort_values('re_ts', inplace=True)
                data.reset_index(inplace=True)

    if drop_first is not False:
        data.drop(data.loc[:drop_first].index, inplace=True)
        data = data.reindex()
    # check if the time scale is in ms
    sess_len_sec = (datetime.strptime(data['bns_ts'].iloc[-1][:25], '%Y-%m-%dT%H:%M:%S.%f') -
                    datetime.strptime(data['bns_ts'].iloc[0][:25], '%Y-%m-%dT%H:%M:%S.%f')).seconds
    if data['re_ts'].iloc[-1] / (sess_len_sec + 1e-6) < 1e5:  # should be 1e6 normally
        logger_.warning('Rotary encoder reset logs events in ms instead of us: ' +
                        'RE firmware needs upgrading and wheel velocity is potentially inaccurate')
        data['re_ts'] = data['re_ts'] * 1000
    return data


def _groom_wheel_data_ge5(data, label='file ', path=''):
    """
    The whole purpose of this function is to account for variability and corruption in
    the wheel position files. There are many possible errors described below, but
    nothing excludes getting new ones.
    """
    # sometimes the text file is cropped
    if np.any(data.isna()):
        logger_.warning(label + ' has missing/incomplete records \n %s', path)
    data.dropna(inplace=True)
    data.drop_duplicates(keep='first', inplace=True)
    data.reset_index(inplace=True)
    # handle the clock resets when microseconds exceed uint32 max value
    drop_first = False
    if any(np.diff(data['re_ts']) < 0):
        ind = np.where(np.diff(data['re_ts']) < 0)[0]
        for i in ind:
            # the first sample may be corrupt, in this case throw away
            if i <= 1:
                drop_first = i
                logger_.warning(label + ' rotary encoder positions timestamps'
                                        ' first sample corrupt ' + str(path))
            # if it's an uint32 wraparound, the diff should be close to 2 ** 32
            elif 32 - np.log2(data['re_ts'][i] - data['re_ts'][i + 1]) < 0.2:
                data.loc[i + 1:, 're_ts'] = data.loc[i + 1:, 're_ts'] + 2 ** 32
            # there is also the case where 2 positions are swapped and need to be swapped back

            elif data['re_ts'][i] > data['re_ts'][i + 1] > data['re_ts'][i - 1]:
                logger_.warning(label + ' rotary encoder timestamps swapped at index: ' +
                                str(i) + '  ' + str(path))
                a, b = data.iloc[i].copy(), data.iloc[i + 1].copy()
                data.iloc[i], data.iloc[i + 1] = b, a
            # if none of those 3 cases apply, raise an error
            else:
                logger_.error(label + ' Rotary encoder timestamps are not sorted.' + str(path))
                data.sort_values('re_ts', inplace=True)
                data.reset_index(inplace=True)

    if drop_first is not False:
        data.drop(data.loc[:drop_first].index, inplace=True)
        data = data.reindex()
    # check if the time scale is in ms
    if (data['re_ts'].iloc[-1] - data['re_ts'].iloc[0]) / 1e6 < 0:
        logger_.warning('Rotary encoder reset logs events in ms instead of us: ' +
                        'RE firmware needs upgrading and wheel velocity is potentially inaccurate')
        data['re_ts'] = data['re_ts'] * 1000
    return data


def save_bool(save, dataset_type):
    logger = logging.getLogger('ibllib.alf')
    if isinstance(save, bool):
        out = save
    elif isinstance(save, list):
        out = (dataset_type in save) or (Path(dataset_type).stem in save)
    if out:
        logger.debug('extracting' + dataset_type)
    return out


def get_port_events(events: dict, name: str = '') -> list:
    """Return all timestamps from bpod raw data
    trial['behavior_data']['Events timestamps'] that match name"""
    out: list = []
    for k in events:
        if name in k:
            out.extend(events[k])
    out = sorted(out)

    return out

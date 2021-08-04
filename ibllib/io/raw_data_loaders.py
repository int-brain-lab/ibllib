#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi, Miles Wells
# @Date: Monday, July 16th 2018, 1:28:46 pm
"""
Raw Data Loader functions for PyBpod rig

Module contains one loader function per raw datafile
"""
import json
import logging
import wave
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from iblutil.io import jsonable
from ibllib.io.video import assert_valid_label
from ibllib.misc import version
from ibllib.time import uncycle_pgts, convert_pgts

_logger = logging.getLogger('ibllib')


def trial_times_to_times(raw_trial):
    """
    Parse and convert all trial timestamps to "absolute" time.
    Float64 seconds from session start.

    0---BpodStart---TrialStart0---------TrialEnd0-----TrialStart1---TrialEnd1...0---ts0---ts1---
    tsN...absTS = tsN + TrialStartN - BpodStart

    Bpod timestamps are in microseconds (µs)
    PyBpod timestamps are is seconds (s)

    :param raw_trial: raw trial data
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


def load_data(session_path: Union[str, Path], time='absolute'):
    """
    Load PyBpod data files (.jsonable).

    Bpod timestamps are in microseconds (µs)
    PyBpod timestamps are is seconds (s)

    :param session_path: Absolute path of session folder
    :type session_path: str, Path
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


def load_camera_frameData(session_path, camera: str = 'left', raw: bool = False) -> pd.DataFrame:
    """ Loads binary frame data from Bonsai camera recording workflow.

    Args:
        session_path (StrPath): Path to session folder
        camera (str, optional): Load FramsData for specific camera. Defaults to 'left'.
        raw (bool, optional): Whether to return raw or parsed data. Defaults to False.

    Returns:
        parsed: (raw=False, Default)
        pandas.DataFrame: 4 int64 columns: {
                Timestamp,              # float64 (seconds from session start)
                embeddedTimeStamp,      # float64 (seconds from session start)
                embeddedFrameCounter,   # int64 (Frame number from session start)
                embeddedGPIOPinState    # object (State of each of the 4 GPIO pins as a
                                        # list of numpy boolean arrays
                                        # e.g. np.array([True, False, False, False])
            }
        raw:
            pandas.DataFrame: 4 int64 columns: {
                Timestamp,              # UTC ticks from BehaviorPC
                                        # (100's of ns since midnight 1/1/0001)
                embeddedTimeStamp,      # Camera timestamp (Needs unclycling and conversion)
                embeddedFrameCounter,   # Frame counter (int)
                embeddedGPIOPinState    # GPIO pin state integer representation of 4 pins
            }
    """
    camera = assert_valid_label(camera)
    fpath = Path(session_path).joinpath("raw_video_data")
    fpath = next(fpath.glob(f"_iblrig_{camera}Camera.frameData*.bin"), None)
    assert fpath, f"{fpath}\nFile not Found: Could not find bin file for cam <{camera}>"
    rdata = np.fromfile(fpath, dtype=np.float64)
    assert rdata.size % 4 == 0, "Dimension mismatch: bin file length is not mod 4"
    rows = int(rdata.size / 4)
    data = np.reshape(rdata.astype(np.int64), (rows, 4))
    df_dict = dict.fromkeys(
        ["Timestamp", "embeddedTimeStamp", "embeddedFrameCounter", "embeddedGPIOPinState"]
    )
    df = pd.DataFrame(data, columns=df_dict.keys())
    if raw:
        return df

    df_dict["Timestamp"] = (data[:, 0] - data[0, 0]) / 10_000_000  # in seconds from first frame
    camerats = uncycle_pgts(convert_pgts(data[:, 1]))
    df_dict["embeddedTimeStamp"] = camerats - camerats[0]  # in seconds from first frame
    df_dict["embeddedFrameCounter"] = data[:, 2] - data[0, 2]  # from start
    gpio = (np.right_shift(np.tile(data[:, 3], (4, 1)).T, np.arange(31, 27, -1)) & 0x1) == 1
    df_dict["embeddedGPIOPinState"] = [np.array(x) for x in gpio.tolist()]

    parsed_df = pd.DataFrame.from_dict(df_dict)
    return parsed_df


def load_camera_ssv_times(session_path, camera: str):
    """
    Load the bonsai frame and camera timestamps from Camera.timestamps.ssv

    NB: For some sessions the frame times are in the first column, in others the order is reversed.
    NB: If using the new bin file the bonsai_times is a float in seconds since first frame
    :param session_path: Absolute path of session folder
    :param camera: Name of the camera to load, e.g. 'left'
    :return: array of datetimes, array of frame times in seconds
    """
    camera = assert_valid_label(camera)
    video_path = Path(session_path).joinpath('raw_video_data')
    if next(video_path.glob(f'_iblrig_{camera}Camera.frameData*.bin'), None):
        df = load_camera_frameData(session_path, camera=camera)
        return df['Timestamp'].values, df['embeddedTimeStamp'].values

    file = next(video_path.glob(f'_iblrig_{camera.lower()}Camera.timestamps*.ssv'), None)
    if not file:
        raise FileNotFoundError()
    # NB: Numpy has deprecated support for non-naive timestamps.
    # Converting them is extremely slow: 6000 timestamps takes 0.8615s vs 0.0352s.
    # from datetime import timezone
    # c = {0: lambda x: datetime.fromisoformat(x).astimezone(timezone.utc).replace(tzinfo=None)}

    # Determine the order of the columns by reading one line and testing whether the first value
    # is an integer or not.
    with open(file, 'r') as f:
        line = f.readline()
    type_map = OrderedDict(bonsai='<M8[ns]', camera='<u4')
    try:
        int(line.split(' ')[1])
    except ValueError:
        type_map.move_to_end('bonsai')
    ssv_params = dict(names=type_map.keys(), dtype=','.join(type_map.values()), delimiter=' ')
    ssv_times = np.genfromtxt(file, **ssv_params)  # np.loadtxt is slower for some reason
    bonsai_times = ssv_times['bonsai']
    camera_times = uncycle_pgts(convert_pgts(ssv_times['camera']))
    return bonsai_times, camera_times


def load_embedded_frame_data(session_path, label: str, raw=False):
    """
    Load the embedded frame count and GPIO for a given session.  If the file doesn't exist,
    or is empty, None values are returned.
    :param session_path: Absolute path of session folder
    :param label: The specific video to load, one of ('left', 'right', 'body')
    :param raw: If True the raw data are returned without preprocessing, otherwise frame count is
    returned starting from 0 and the GPIO is returned as a dict of indices
    :return: The frame count, GPIO
    """
    count = load_camera_frame_count(session_path, label, raw=raw)
    gpio = load_camera_gpio(session_path, label, as_dicts=not raw)
    return count, gpio


def load_camera_frame_count(session_path, label: str, raw=True):
    """
    Load the embedded frame count for a given session.  If the file doesn't exist, or is empty,
    a None value is returned.
    :param session_path: Absolute path of session folder
    :param label: The specific video to load, one of ('left', 'right', 'body')
    :param raw: If True the raw data are returned without preprocessing, otherwise frame count is
    returned starting from 0
    :return: The frame count
    """
    if session_path is None:
        return

    label = assert_valid_label(label)
    video_path = Path(session_path).joinpath('raw_video_data')
    if next(video_path.glob(f'_iblrig_{label}Camera.frameData*.bin'), None):
        df = load_camera_frameData(session_path, camera=label)
        return df['embeddedFrameCounter'].values

    # Load frame count
    glob = video_path.glob(f'_iblrig_{label}Camera.frame_counter*.bin')
    count_file = next(glob, None)
    count = np.fromfile(count_file, dtype=np.float64).astype(int) if count_file else []
    if len(count) == 0:
        return
    if not raw:
        count -= count[0]  # start from zero
    return count


def load_camera_gpio(session_path, label: str, as_dicts=False):
    """
    Load the GPIO for a given session.  If the file doesn't exist, or is empty, a None value is
    returned.

    The raw binary file contains uint32 values (saved as doubles) where the first 4 bits
    represent the state of each of the 4 GPIO pins. The array is expanded to an n x 4 array by
    shifting each bit to the end and checking whether it is 0 (low state) or 1 (high state).

    :param session_path: Absolute path of session folder
    :param label: The specific video to load, one of ('left', 'right', 'body')
    :param as_dicts: If False the raw data are returned boolean array with shape (n_frames, n_pins)
     otherwise GPIO is returned as a list of dictionaries with keys ('indices', 'polarities').
    :return: An nx4 boolean array where columns represent state of GPIO pins 1-4.
     If as_dicts is True, a list of dicts is returned with keys ('indices', 'polarities'),
     or None if the dictionary is empty.
    """
    if session_path is None:
        return
    raw_path = Path(session_path).joinpath('raw_video_data')
    label = assert_valid_label(label)

    # Load pin state
    if next(raw_path.glob(f'_iblrig_{label}Camera.frameData*.bin'), False):
        df = load_camera_frameData(session_path, camera=label, raw=False)
        gpio = np.array([x for x in df['embeddedGPIOPinState'].values])
        if len(gpio) == 0:
            return [None] * 4 if as_dicts else None
    else:
        GPIO_file = next(raw_path.glob(f'_iblrig_{label}Camera.GPIO*.bin'), None)
        # This deals with missing and empty files the same
        gpio = np.fromfile(GPIO_file, dtype=np.float64).astype(np.uint32) if GPIO_file else []
        # Check values make sense (4 pins = 16 possible values)
        if not np.isin(gpio, np.left_shift(np.arange(2 ** 4, dtype=np.uint32), 32 - 4)).all():
            _logger.warning('Unexpected GPIO values; decoding may fail')
        if len(gpio) == 0:
            return [None] * 4 if as_dicts else None
        # 4 pins represented as uint32
        # For each pin, shift its bit to the end and check the bit is set
        gpio = (np.right_shift(np.tile(gpio, (4, 1)).T, np.arange(31, 27, -1)) & 0x1) == 1

    if as_dicts:
        if not gpio.any():
            _logger.error('No GPIO changes')
            return [None] * 4
        # Find state changes for each pin and construct a dict of indices and polarities for each
        edges = np.vstack((gpio[0, :], np.diff(gpio.astype(int), axis=0)))
        # gpio = [(ind := np.where(edges[:, i])[0], edges[ind, i]) for i in range(4)]
        # gpio = [dict(zip(('indices', 'polarities'), x)) for x in gpio_]  # py3.8
        gpio = [{'indices': np.where(edges[:, i])[0],
                 'polarities': edges[edges[:, i] != 0, i]}
                for i in range(4)]
        # Replace empty dicts with None
        gpio = [None if x['indices'].size == 0 else x for x in gpio]

    return gpio


def load_settings(session_path: Union[str, Path]):
    """
    Load PyBpod Settings files (.json).

    [description]

    :param session_path: Absolute path of session folder
    :type session_path: str, Path
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
    data['bns_ts'] = pd.to_datetime(data['bns_ts'])
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
        if data[col].dtype == object and col not in ['bns_ts']:
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
    it1 = ind[it0].astype(int)
    ind0 = np.unique(np.r_[it0, it0 + 1])
    ind1 = np.unique(np.r_[it1, it1 + 1])
    if return_index:
        return t0[ind0], t1[ind1], ind0, ind1
    else:
        return t0[ind0], t1[ind1]


def load_bpod_fronts(session_path: str, data: list = False) -> list:
    """load_bpod_fronts
    Loads BNC1 and BNC2 bpod channels times and polarities from session_path

    :param session_path: a valid session_path
    :type session_path: str
    :param data: pre-loaded raw data dict, defaults to False
    :type data: list, optional
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

"""Data extraction from widefield binary file"""
from collections import OrderedDict
import logging
from pathlib import Path, PureWindowsPath
import uuid

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import parse_version
# from wfield.decomposition import approximate_svd
# from wfield.plots import plot_summary_motion_correction
# from wfield.registration import motion_correct
import wfield.cli as wfield_cli

from wfield import decomposition, plots, registration, utils, io as wfield_io
from labcams.io import parse_cam_log

import one.alf.io as alfio
from iblutil.util import Bunch
import ibllib.dsp as dsp
import ibllib.exceptions as err
from ibllib.io.raw_data_loaders import load_widefield_mmap
from ibllib.io.extractors import biased_trials, training_trials
from ibllib.io.extractors.base import BaseExtractor
from ibllib.dsp.utils import sync_timestamps
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync, get_sync_fronts
from ibllib.io.video import get_video_meta

_logger = logging.getLogger('ibllib')
FILENAME_MAP = {
    'motioncorrect_*.bin': 'widefield.raw_motionCorrected.bin',
    'motion_correction_shifts.npy': 'widefield.motionCorrentionShifts.npy',
    'motion_correction_rotation.npy': 'widefield.motionCorrentionRotation.npy',
    # 'frames_average.npy': 'frameAverage.widefield.npy',
    'U.npy': 'widefieldU.images.npy',
    'SVT.npy': 'widefieldSVT.uncorrected.npy',
    # 'rcoeffs.npy': '',
    # 'T.npy': 'widefieldT.uncorrected.npy',
    'SVTcorr.npy': 'widefieldSVT.haemoCorrected.npy'
}


class Widefield(BaseExtractor):
    save_names = ('_ibl_trials.feedbackType.npy',)
    raw_names = ('motioncorrect_2_540_640_uint16.bin', 'motion_correction_shifts.npy', 'motion_correction_rotation.npy',
                 'frames_average.npy', 'U.npy', 'SVT.npy', 'rcoeffs.npy', 'T.npy', 'SVTcorr.npy')
    var_names = ('feedbackType',)

    def __init__(self, *args, **kwargs):
        """An extractor for all widefield data"""
        super().__init__(*args, **kwargs)

    def _extract(self, **kwargs):
        """
        NB: kwargs should be loaded from meta file
        Parameters
        ----------
        n_channels
        dtype
        shape
        kwargs

        Returns
        -------

        """
        self.preprocess(**kwargs)
        ##########################################################
        # dat = load_widefield_mmap(self.session_path, dtype=dtype, shape=shape, mode='r+')

        # return [out[k] for k in out] + [wheel['timestamps'], wheel['position'],
        #                                 moves['intervals'], moves['peakAmplitude']]

    def _symlink_files(self):
        ...

    def preprocess(self, fs=30, functional_channel=0, nbaseline_frames=30, k=200):
        data_path = self.session_path.joinpath('raw_widefield_data')

        # MOTION CORRECTION
        wfield_cli._motion(data_path)
        # COMPUTE AVERAGE FOR BASELINE
        wfield_cli._baseline(data_path, nbaseline_frames)
        # DATA REDUCTION
        wfield_cli._decompose(data_path, k=k)
        # HAEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = wfield_cli.load_stack(data_path)
        if dat.shape[1] == 2:
            del dat
            wfield_cli._hemocorrect(data_path, fs=fs, functional_channel=functional_channel)

    def rename_files(self, session_folder) -> bool:
        """
        Rename the raw widefield data for a given session.

        Parameters
        ----------
        session_folder : str, pathlib.Path
            A session path containing widefield data.

        Returns
        -------
        success : bool
            True if all files were successfully renamed.
        TODO Double-check filenames and call this function
        """
        session_path = Path(session_folder).joinpath('raw_widefield_data')
        if not session_path.exists():
            _logger.warning(f'Path does not exist: {session_path}')
            return False
        renames = (
            ('dorsal_cortex_landmarks.json', 'widefieldLandmarks.dorsalCortex.json'),
            ('*.dat', 'widefield.raw.dat'),
            ('*.camlog', 'widefieldEvents.raw.camlog')
        )
        success = True
        for before, after in renames:
            try:
                filename = next(session_path.glob(before))
                filename.rename(after)
                # TODO Save nchannels and frame size from filename?
            except StopIteration:
                _logger.warning(f'File not found: {before}')
                success = False
        return success

    def sync_timestamps(self, bin_exists=False):
        filepath = next(self.session_path.joinpath('raw_widefield_data').glob('*.camlog'))
        fpga_sync, chmap = get_main_probe_sync(self.session_path, bin_exists=bin_exists)
        bpod = get_sync_fronts(fpga_sync, chmap['bpod'])
        logdata, led, sync, ncomm = parse_cam_log(filepath, readTeensy=True)
        if bpod.times.size == 0:
            raise err.SyncBpodFpgaException('No Bpod event found in FPGA. No behaviour extraction. '
                                            'Check channel maps.')

        fcn, drift, iteensy, ifpga = dsp.utils.sync_timestamps(sync.timestamp.values, bpod['times'], return_indices=True)
        _logger.debug(f'Widefield-FPGA clock drift: {drift} ppm')
        assert led.frame.is_monotonic_increasing
        video_path = next(self.session_path.joinpath('raw_widefield_data').glob('widefield.raw*.mov'))
        video_meta = get_video_meta(video_path)
        assert video_meta.length == len(led)
        widefield_times = fcn(led.timestamp.values)



######## SYNC TO TRIALS ########
from labcams import parse_cam_log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


####################### Functions for fetching and organizing data ################################

ch = np.load(Path(sync_path, '_spikeglx_sync.channels.npy'))
times = np.load(Path(sync_path, '_spikeglx_sync.times.npy'))
bpod_ch = 16
use_ch = ch == bpod_ch
bpod_times = times[use_ch]
np.save(Path(localdisk, 'bpod_times.npy'), bpod_times)

def fetch_task_data(subject, date, exp='001', FlatIron='G:\\FlatIron\\zadorlab\\Subjects\\'):
    """
    fetches the task data for a given session and puts it into a dataframe
    Input:
        subject: subject_nickname
        date: date of the experiment
        exp: experiment number
        FlatIron: folder for the downloaded FlatIron data

    Output:
        DataFrame of length ntrials containing all alf trial information
    """

    import os
    alf_path = pjoin(FlatIron, subject, date, exp, 'alf')
    files = os.listdir(alf_path)
    df = pd.DataFrame(columns=['choice', 'contrastLeft', 'contrastRight', 'feedback_times', 'feedbackType', 'firstMovement_times',
                               'goCue_times', 'goCueTrigger_times', 'intervals', 'intervals_bpod', 'probabilityLeft',
                               'response_times', 'rewardVolume', 'stimOff_times', 'stimOn_times'])
    for file in files:
        for column in df.columns:
            if (column in str(file)) & ('_ibl_trial' in str(file)):
                df[column] = np.load(pjoin(alf_path, file))
    df['contrastRight'][np.isnan(df['contrastRight'])] = 0
    df['contrastLeft'][np.isnan(df['contrastLeft'])] = 0
    df['signedContrast'] = df['contrastLeft'] - df['contrastRight']
    return df


def sync_to_task(localdisk):
    """
    synchronize the task data to the imaging data.
    Input:
        localdisk: directory where the imaging data is, and where bpod_times file has been saved
                   this requires you to have already run the function fetchONE script for this
                   session to make sure that alf data is on local computer
    Output:
        a dataframe that has the sync times and frames from both imaging and bpod
    """

    logdata, led, sync, ncomm = parse_cam_log(glob(pjoin(localdisk, '*.camlog'))[0], readTeensy=True)
    bpod_times = np.load(pjoin(localdisk, 'bpod_times.npy'))
    bpod_gaps = np.diff(bpod_times)
    sync.timestamp = sync.timestamp / 1000  # convert from milliseconds to seconds
    sync_gaps = np.diff(sync.timestamp)
    sync['task_time'] = np.nan
    assert len(bpod_times) == len(sync)

    for i in range(len(bpod_gaps)):
        # make sure the pulses are the same
        if math.isclose(bpod_gaps[i], sync_gaps[i], abs_tol=.005):
            sync['task_time'].iloc[i] = bpod_times[i]
        else:
            print('WARNING: syncs do not line up for index {}!!!'.format(i))
    sync['frame'] = (sync['frame'] / 2).astype(int)
    return sync.dropna(axis=0)


def find_nearest(array, value):
    array = np.asarray(array)
    # return (np.abs(array - value)).argmin()
    return (array >= value)[0]


def time_to_frames(sync, led, event_times, dropna=True):
    """
    Attributes the closest frame of the imaging data to an array of events, such as
    stimulus onsets.
    sync: the synchronization pandas DataFrame including the frame and timestamp from imaging, and the
          timestamp for the bpod events from the fpga
    event_times: the time in seconds of the bpod event associated with

    returns an array of len(event_times) with a frame attributed to each event
    """
    event_times = np.array(event_times)
    if dropna:
        event_times = event_times[~np.isnan(event_times)]
    # logdata, led, _, ncomm = parse_cam_log(glob(pjoin(localdisk, '*.camlog'))[0], readTeensy=True)
    sync['conversion'] = sync['timestamp'] - sync['task_time']

    temp_led = led['timestamp'] / 1000  # this is the time of each frame

    event_frames = np.empty(event_times.shape)
    for i in range(len(event_times)):
        offset = sync.iloc[find_nearest(sync['task_time'], event_times[i])]['conversion']
        event_frames[i] = led.iloc[find_nearest(temp_led, event_times[i] + offset)]['frame']
    # print(abs(np.nanmax(event_times)-np.nanmax(event_frames)/15),np.nanmax(event_times))
    # assert abs(np.nanmax(event_times)-np.nanmax(event_frames)/15) < np.nanmax(event_times)/3, 'seems misaligned'
    return (event_frames / 2).astype(int)


def time_to_frameDF(behavior, sync_behavior, localdisk):
    """
    Makes a dataframe that has all the timing events converted to frames
    Inputs:
        behavior: df with len(num_trials) and different columns witih trial info
        sync_behavior: the sync dataframe to go between task time and frames
        localdisk: the directory where the above two are saved for this session
    outputs:
        frameDF: a dataframe with all the behavioral events that end with _times, that are instead
        aligned to the camera and displayed as frames
    """
    mask = ['times' in key for key in behavior.keys()]
    time_df = behavior[behavior.keys()[mask]]
    frame_df = pd.DataFrame(columns=time_df.keys())
    logdata, led, _, ncomm = parse_cam_log(glob(pjoin(localdisk, '*.camlog'))[0], readTeensy=True)
    for (columnName, columnData) in time_df.iteritems():
        frame_df[columnName] = time_to_frames(sync_behavior, led, np.array(columnData), dropna=False)
    frameDF = frame_df.astype(np.int64)
    frameDF[frameDF == 0] = np.nan

    return frameDF

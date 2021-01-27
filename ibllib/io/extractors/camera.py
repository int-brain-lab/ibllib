""" Camera extractor functions
This module handles extraction of camera timestamps for both Bpod and FPGA.
"""
import logging
from pathlib import Path
from functools import partial

import numpy as np
import cv2

from oneibl.stream import VideoStreamer
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors.ephys_fpga import _get_sync_fronts, get_main_probe_sync
from ibllib.io.raw_data_loaders import load_bpod_fronts
from ibllib.io.extractors.base import (
    BaseBpodTrialsExtractor,
    BaseExtractor,
    run_extractor_classes,
)

_logger = logging.getLogger('ibllib')
PIN_STATE_THRESHOLD = 1


def extract_camera_sync(sync, chmap=None):
    """
    Extract camera timestamps from the sync matrix

    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace
    :param chmap: dictionary containing channel indices. Default to constant.
    :return: dictionary containing camera timestamps
    """
    # NB: should we check we opencv the expected number of frames ?
    assert(chmap)
    sr = _get_sync_fronts(sync, chmap['right_camera'])
    sl = _get_sync_fronts(sync, chmap['left_camera'])
    sb = _get_sync_fronts(sync, chmap['body_camera'])
    return {'right': sr.times[::2],
            'left': sl.times[::2],
            'body': sb.times[::2]}


def get_video_length(video_path):
    """
    Returns video length
    :param video_path: A path to the video
    :return:
    TODO Use get_video_meta with key arg instead?
    """
    is_url = isinstance(video_path, str) and video_path.startswith('http')
    cap = VideoStreamer(video_path).cap if is_url else cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f'Failed to open video file {video_path}'
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


class CameraTimestampsFPGA(BaseExtractor):

    def __init__(self, label, session_path=None):
        super().__init__(session_path)
        assert label in ('left', 'right', 'body'), f'unknown video label "{label}"'
        self.label = label
        self.save_names = f'_ibl_{label}Camera.times.npy'
        self.var_names = f'{label}_camera_timestamps'

    def _extract(self, sync=None, chmap=None, video_path=None):
        """
        The raw timestamps are taken from the FPGA.  These are the times of the camera's frame
        TTLs.
        If the pin state file exists, these timestamps are aligned to the video frames using the
        audio TTLs.  Frames missing from the embedded frame count are removed from the timestamps
        array.
        If the pin state file does not exist, the left and right camera timestamps are aligned
        using the wheel data.
        :param sync:
        :param chmap:
        :return:
        """
        fpga_times = extract_camera_sync(sync=sync, chmap=chmap)
        count, pin_state = load_embedded_frame_data(self.session_path, self.label, raw=False)

        if pin_state is not None and any(pin_state):
            _logger.info('Aligning to audio TTLs')
            # Extract audio TTLs
            audio = _get_sync_fronts(sync, chmap['audio'])
            # make sure that there are no 2 consecutive fall or consecutive rise events
            assert (np.all(np.abs(np.diff(audio['polarities'])) == 2))
            """
            NB: Some of the audio TTLs occur very close together, and are therefore not 
            reflected in the pin state.
            """
            import matplotlib.pyplot as plt
            from ibllib.plots import vertical_lines
            plt.subplots(1, 1)
            ups = audio['polarities'] == 1
            downs = audio['polarities'] == -1
            vertical_lines(audio['times'][ups], ymin=0, ymax=1e-5, color='g', linestyle=':')
            vertical_lines(audio['times'][downs], ymin=0, ymax=1e-5, color='r', linestyle=':')
            # 0.00032999999999994145
            aud_diff = np.diff(audio['times'])
            plt.hist(aud_diff[aud_diff < 15], 1000)

            # make sure first TTL is high
            assert audio['polarities'][0] == 1
            # make sure first GPIO state is low
            assert pin_state[0] == False
            """
            The length of the count and pin state are regularly longer than the length of 
            the video file.  Here we assert that the video is either shorter or the same 
            length as the arrays, and  we make an assumption that the missing frames are 
            right at the end of the video.  We therefore simply shorten the arrays to match
            the length of the video.
            """
            if video_path is None:
                filename = f'_iblrig_{self.label}Camera.raw.mp4'
                video_path = self.session_path / 'raw_video_data' / filename
            length = get_video_length(video_path)
            if count.size > length:
                count = count[:length]
                pin_state = pin_state[:length]
            else:
                assert length == count.size, 'fewer counts than frames'
            raw_ts = fpga_times[self.label]
            timestamps = align_with_audio(raw_ts, audio['times'][::2], pin_state,  count)
        else:
            _logger.warning('Alignment by wheel data not yet implemented')
            timestamps = fpga_times[self.label]

        return timestamps


class CameraTimestampsBpod(BaseBpodTrialsExtractor):
    """
    Get the camera timestamps from the Bpod

    The camera events are logged only during the events not in between, so the times need
    to be interpolated
    """
    save_names = '_ibl_leftCamera.times.npy'
    var_names = 'left_camera_timestamps'

    def _extract(self):
        ts = self._times_from_bpod()  # FIXME Extrapolate after alignment
        count, pin_state = load_embedded_frame_data(self.session_path, 'left', raw=False)

        if pin_state is not None and any(pin_state):
            _logger.info('Aligning to audio TTLs')
            # Extract audio TTLs
            _, audio = load_bpod_fronts(self.session_path, self.bpod_trials)
            # make sure that there are no 2 consecutive fall or consecutive rise events
            assert (np.all(np.abs(np.diff(audio['polarities'])) == 2))
            # make sure first TTL is high
            assert audio['polarities'][0] == 1
            return align_with_audio(ts, audio['times'][::2], pin_state, count)
        else:
            _logger.warning('Alignment by wheel data not yet implemented')

    def _times_from_bpod(self):
        ntrials = len(self.bpod_trials)

        cam_times = []
        n_frames = 0
        n_out_of_sync = 0
        for ind in np.arange(ntrials):
            # get upgoing and downgoing fronts
            pin = np.array(self.bpod_trials[ind]['behavior_data']
                           ['Events timestamps'].get('Port1In'))
            pout = np.array(self.bpod_trials[ind]['behavior_data']
                            ['Events timestamps'].get('Port1Out'))
            # some trials at startup may not have the camera working, discard
            if np.all(pin) is None:
                continue
            # if the trial starts in the middle of a square, discard the first downgoing front
            if pout[0] < pin[0]:
                pout = pout[1:]
            # same if the last sample is during an upgoing front,
            # always put size as it happens last
            pin = pin[:pout.size]
            frate = np.median(np.diff(pin))
            if ind > 0:
                """
                assert that the pulses have the same length and that we don't miss frames during
                the trial, the refresh rate of bpod is 100us
                """
                test1 = np.all(np.abs(1 - (pin - pout) / np.median(pin - pout)) < 0.1)
                test2 = np.all(np.abs(np.diff(pin) - frate) <= 0.00011)
                if not all([test1, test2]):
                    n_out_of_sync += 1
            # grow a list of cam times for ech trial
            cam_times.append(pin)
            n_frames += pin.size

        if n_out_of_sync > 0:
            _logger.warning(f"{n_out_of_sync} trials with bpod camera frame times not within"
                            f" 10% of the expected sampling rate")

        t_first_frame = np.array([c[0] for c in cam_times])
        t_last_frame = np.array([c[-1] for c in cam_times])
        frate = 1 / np.nanmedian(np.array([np.median(np.diff(c)) for c in cam_times]))
        intertrial_duration = t_first_frame[1:] - t_last_frame[:-1]
        intertrial_missed_frames = np.int32(np.round(intertrial_duration * frate)) - 1

        # initialize the full times array
        frame_times = np.zeros(n_frames + int(np.sum(intertrial_missed_frames)))
        ii = 0
        for trial, cam_time in enumerate(cam_times):
            if cam_time is not None:
                # populate first the recovered times within the trials
                frame_times[ii: ii + cam_time.size] = cam_time
                ii += cam_time.size
            if trial == (len(cam_times) - 1):
                break
            # then extrapolate in-between
            nmiss = intertrial_missed_frames[trial]
            frame_times[ii: ii + nmiss] = (cam_time[-1] + intertrial_duration[trial] /
                                           (nmiss + 1) * (np.arange(nmiss) + 1))
            ii += nmiss
        # import matplotlib.pyplot as plt
        # plt.plot(np.diff(frame_times))
        """
        if we find a video file, get the number of frames and extrapolate the times
         using the median frame rate as the video stops after the bpod
        """
        video_file, = list(self.session_path
                           .joinpath('raw_video_data')
                           .glob('_iblrig_leftCamera*.mp4'))
        if video_file:
            cap = cv2.VideoCapture(str(video_file))
            nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if nframes > len(frame_times):
                n_missing = int(nframes - frame_times.size)
                to_app = (np.arange(n_missing,) + 1) / frate + frame_times[-1]
                frame_times = np.r_[frame_times, to_app]
        assert all(np.diff(frame_times) > 0)  # negative diffs implies a big problem
        return frame_times


def align_with_audio(timestamps, audio, pin_state, count,
                     extrapolate_missing=True, display=False):
    """
    Groom the raw FPGA or Bpod camera timestamps using the frame embedded audio TTLs and frame
    counter.
    :param timestamps: An array of raw FPGA or Bpod camera timestamps
    :param audio: An array of FPGA or Bpod audio TTL times
    :param pin_state: An array of camera pin states
    :param count: An array of frame numbers
    :param extrapolate_missing: If true and the number of timestamps is fewer than the number of
    frame counts, the remaining timestamps are extrapolated based on the frame rate, otherwise
    they are NaNs
    :param display: Plot the resulting timestamps
    :return: The corrected frame timestamps
    """
    # Some assertions made on the raw data
    assert count.size == pin_state.size, 'frame count and pin state size mismatch'
    assert all(np.diff(count) > 0), 'frame count not strictly increasing'
    assert all(np.diff(timestamps) > 0), 'FPGA camera times not strictly increasing'
    low2high = np.diff(pin_state.astype(int)) == 1
    assert sum(low2high) <= audio.size, 'more audio TTLs detected on camera than TTLs sent'

    """Here we will ensure that the FPGA camera times match the number of video frames in 
    length.  We will make the following assumptions: 

    1. The number of FPGA camera times is equal to or greater than the number of video frames.
    2. No TTLs were missed between the camera and FPGA.
    3. No pin states were missed by Bonsai.
    4  No pixel count data was missed by Bonsai.

    In other words the count and pin state arrays accurately reflect the number of frames 
    sent by the camera and should therefore be the same length, and the length of the frame 
    counter should match the number of saved video frames.

    The missing frame timestamps are removed in three stages:

    1. Remove any timestamps that occurred before video frame acquisition in Bonsai.
    2. Remove any timestamps where the frame counter reported missing frames, i.e. remove the
    dropped frames which occurred throughout the session.
    3. Remove the trailing timestamps at the end of the session if the camera was turned off
    in the wrong order.
    """
    # Align on first pin state change
    first_uptick = (pin_state > 0).argmax()
    first_ttl = np.searchsorted(timestamps, audio[0])
    """Here we find up to which index in the FPGA times we discard by taking the difference 
    between the index of the first pin state change (when the audio TTL was reported by the 
    camera) and the index of the first audio TTL in FPGA time.  We subtract the difference 
    between the frame count at the first pin state change and the index to account for any 
    video frames that were not saved during this period (we will remove those from the 
    camera FPGA times later).
    """
    # Minus any frames that were dropped between the start of frame acquisition and the
    # first TTL
    start = first_ttl - first_uptick - (count[first_uptick] - first_uptick)
    assert start >= 0

    # Remove the extraneous timestamps from the beginning and end
    end = count[-1] + 1 + start
    ts = timestamps[start:end]
    if ts.size < count.size:
        """
        For ephys sessions there may be fewer FPGA times than frame counts if SpikeGLX is turned 
        off before the video acquisition workflow.  For Bpod this always occurs because Bpod 
        finishes before the camera workflow.  For Bpod the times are already extrapolated for 
        these late frames."""
        n_missing = count.size - ts.size
        _logger.warning(f'{n_missing} fewer FPGA timestamps than frame counts')
        frate = round(1 / np.nanmedian(np.diff(ts)))
        to_app = ((np.arange(n_missing, ) + 1) / frate + ts[-1]
                  if extrapolate_missing
                  else np.full(n_missing, np.nan))
        ts = np.r_[ts, to_app]  # Append the missing times
    assert ts.size >= count.size
    assert ts.size == count[-1] + 1

    # Remove the rest of the dropped frames
    ts = ts[count]
    assert np.searchsorted(ts, audio[0]) == first_uptick

    if display:
        # Plot to check
        import matplotlib.pyplot as plt
        from ibllib.plots import vertical_lines
        fig, axes = plt.subplots(2, 1)
        y = (pin_state > 0).astype(float)
        y *= 1e-5  # For scale when zoomed in
        axes[0].plot(ts, y, marker='d', color='blue', drawstyle='steps-pre')
        axes[0].plot(ts, np.zeros_like(ts), 'kx')
        vertical_lines(audio, ymin=0, ymax=1e-5, color='r', linestyle=':', ax=axes[0])
        # gpio_ttl_diff = ts[low2high] - audio[:sum(low2high)]
        # axes[1].hist(gpio_ttl_diff, 1000)
        # _logger.info('%i timestamps negative when taking diff between audio TTLs and GPIO',
        #              np.sum(gpio_ttl_diff < 0))

    return ts


def align_with_audio_safe(timestamps, audio, pin_state, count,
                          extrapolate_missing=True, display=False):
    """
    Groom the raw FPGA or Bpod camera timestamps using the frame embedded audio TTLs and frame
    counter.
    :param timestamps: An array of raw FPGA or Bpod camera timestamps
    :param audio: An array of FPGA or Bpod audio TTL times
    :param pin_state: An array of camera pin states
    :param count: An array of frame numbers
    :param extrapolate_missing: If true and the number of timestamps is fewer than the number of
    frame counts, the remaining timestamps are extrapolated based on the frame rate, otherwise
    they are NaNs
    :param display: Plot the resulting timestamps
    :return: The corrected frame timestamps
    """
    # Some assertions made on the raw data
    assert count.size == pin_state.size, 'frame count and pin state size mismatch'
    assert all(np.diff(count) > 0), 'frame count not strictly increasing'
    assert all(np.diff(timestamps) > 0), 'FPGA camera times not strictly increasing'
    low2high = np.diff(pin_state.astype(int)) == 1
    assert sum(low2high) <= audio.size, 'more audio TTLs detected on camera than TTLs sent'

    # Align on first pin state change
    first_uptick = (pin_state > 0).argmax()
    first_ttl = np.searchsorted(timestamps, audio[0])
    """Here we find up to which index in the FPGA times we discard by taking the difference 
    between the index of the first pin state change (when the audio TTL was reported by the 
    camera) and the index of the first audio TTL in FPGA time.  We subtract the difference 
    between the frame count at the first pin state change and the index to account for any 
    video frames that were not saved during this period (we will remove those from the 
    camera FPGA times later).
    """
    # NB: The pin state to be high for 2 consecutive frames
    low2high = np.insert(np.diff((pin_state > 0).astype(int)) == 1, 0, False)
    upticks = np.insert(*np.where(low2high), 0, 0)

    start = first_ttl - first_uptick - (count[first_uptick] - first_uptick)
    assert start >= 0
    # Remove the timestamps that occur before bonsai starts acquiring
    ts = timestamps[start:]
    ttls = np.searchsorted(ts, audio)
    timestamps_split = np.split(ts, ttls)
    """
    This fails either because some pin state changes are a frame out, or because bonsai messed 
    up the count.
    """
    for i in range(len(upticks) - 1):
        count_frag = count[upticks[i]:upticks[i + 1]] - count[upticks[i]]
        missing = np.setdiff1d(np.arange(0, count_frag[-1]), count_frag)
        assert (timestamps_split[i].size - count_frag.size) == len(missing)
        timestamps_split[i] = timestamps_split[i][count_frag]

    return ts


def load_embedded_frame_data(session_path, label: str, raw=False):
    """

    :param session_path:
    :param label: The specific camera to load, one of ('left', 'right', 'body')
    :param raw: If True the raw data are returned without preprocessing (thresholding, etc.)
    :return: The frame counter, the GPIO pin state
    """
    if session_path is None:
        return None, None
    raw_path = Path(session_path).joinpath('raw_video_data')

    # Load frame count
    count_file = raw_path / f'_iblrig_{label}Camera.frame_counter.bin'
    count = np.fromfile(count_file, dtype=np.float64).astype(int) if count_file.exists() else None
    if not (count is None or raw):
        count -= count[0]  # start from zero

    # Load pin state
    GPIO_file = raw_path / f'_iblrig_{label}Camera.GPIO.bin'
    gpio = np.fromfile(GPIO_file, dtype=np.float64).astype(int) if GPIO_file.exists() else None
    if not (gpio is None or raw):
        assert np.unique(gpio).size == 2  # TODO relax assertion, rely on QC check
        # min-max normalize + threshold
        gpio = (gpio - gpio.min()) / (gpio.max() - gpio.min()) > 0.5

    return count, gpio


def extract_all(session_path, session_type=None, save=True, **kwargs):
    """
    For the IBL ephys task, reads ephys binary file and extract:
        -   video time stamps
    :param session_path: '/path/to/subject/yyyy-mm-dd/001'
    :param session_type: the session type to extract, i.e. 'ephys', 'training' or 'biased'. If
    None the session type is inferred from the settings file.
    :param save: Bool, defaults to False
    :param kwargs: parameters to pass to the extractor
    :return: outputs, files
    """
    if session_type is None:
        session_type = get_session_extractor_type(session_path)
    if session_type == 'ephys':
        extractor = [partial(CameraTimestampsFPGA, label) for label in ('left', 'right', 'body')]
        if 'sync' not in kwargs:
            kwargs['sync'], kwargs['chmap'] = \
                get_main_probe_sync(session_path, bin_exists=kwargs.pop('bin_exists', False))
    elif session_type in ['biased', 'training']:
        extractor = CameraTimestampsBpod
    else:
        raise ValueError(f"Session type {session_type} as no matching extractor {session_path}")

    outputs, files = run_extractor_classes(
        extractor, session_path=session_path, save=save, **kwargs)
    return outputs, files

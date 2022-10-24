""" Camera extractor functions
This module handles extraction of camera timestamps for both Bpod and FPGA.
"""
import logging
from functools import partial

import cv2
import numpy as np
import matplotlib.pyplot as plt
from iblutil.util import range_str

import neurodsp.utils as dsp
from ibllib.plots import squares, vertical_lines
from ibllib.io.video import assert_valid_label, VideoStreamer
from iblutil.numerical import within_ranges
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors.ephys_fpga import get_sync_fronts, get_sync_and_chn_map
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import (
    BaseBpodTrialsExtractor,
    BaseExtractor,
    run_extractor_classes,
    _get_task_types_json_config
)

_logger = logging.getLogger(__name__)


def extract_camera_sync(sync, chmap=None):
    """
    Extract camera timestamps from the sync matrix

    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace
    :param chmap: dictionary containing channel indices. Default to constant.
    :return: dictionary containing camera timestamps
    """
    assert chmap
    sr = get_sync_fronts(sync, chmap['right_camera'])
    sl = get_sync_fronts(sync, chmap['left_camera'])
    sb = get_sync_fronts(sync, chmap['body_camera'])
    return {'right': sr.times[::2],
            'left': sl.times[::2],
            'body': sb.times[::2]}


def get_video_length(video_path):
    """
    Returns video length
    :param video_path: A path to the video
    :return:
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
        self.label = assert_valid_label(label)
        self.save_names = f'_ibl_{label}Camera.times.npy'
        self.var_names = f'{label}_camera_timestamps'
        self._log_level = _logger.level
        _logger.setLevel(logging.DEBUG)

    def __del__(self):
        _logger.setLevel(self._log_level)

    def _extract(self, sync=None, chmap=None, video_path=None,
                 display=False, extrapolate_missing=True, **kwargs):
        """
        The raw timestamps are taken from the FPGA. These are the times of the camera's frame TTLs.
        If the pin state file exists, these timestamps are aligned to the video frames using the
        audio TTLs.  Frames missing from the embedded frame count are removed from the timestamps
        array.
        If the pin state file does not exist, the left and right camera timestamps may be aligned
        using the wheel data.
        :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace.
        :param chmap: dictionary containing channel indices. Default to constant.
        :param video_path: an optional path for fetching the number of frames.  If None,
        the video is loaded from the session path.  If an int is provided this is taken to be
        the total number of frames.
        :param display: if True, the audio and GPIO fronts are plotted.
        :param extrapolate_missing: if True, any missing timestamps at the beginning and end of
        the session are extrapolated based on the median frame rate, otherwise they will be NaNs.
        :return: a numpy array of camera timestamps
        """
        fpga_times = extract_camera_sync(sync=sync, chmap=chmap)
        count, (*_, gpio) = raw.load_embedded_frame_data(self.session_path, self.label)
        raw_ts = fpga_times[self.label]

        if video_path is None:
            filename = f'_iblrig_{self.label}Camera.raw.mp4'
            video_path = self.session_path.joinpath('raw_video_data', filename)
        # Permit the video path to be the length for development and debugging purposes
        length = (video_path if isinstance(video_path, int) else get_video_length(video_path))
        _logger.debug(f'Number of video frames = {length}')

        if gpio is not None and gpio['indices'].size > 1:
            _logger.info('Aligning to audio TTLs')
            # Extract audio TTLs
            audio = get_sync_fronts(sync, chmap['audio'])
            _, ts = raw.load_camera_ssv_times(self.session_path, self.label)
            try:
                """
                NB: Some of the audio TTLs occur very close together, and are therefore not
                reflected in the pin state.  This function removes those.  Also converts frame
                times to FPGA time.
                """
                gpio, audio, ts = groom_pin_state(gpio, audio, ts, display=display)
                """
                The length of the count and pin state are regularly longer than the length of
                the video file.  Here we assert that the video is either shorter or the same
                length as the arrays, and  we make an assumption that the missing frames are
                right at the end of the video.  We therefore simply shorten the arrays to match
                the length of the video.
                """
                if count.size > length:
                    count = count[:length]
                else:
                    assert length == count.size, 'fewer counts than frames'
                raw_ts = fpga_times[self.label]
                assert raw_ts.shape[0] > 0, 'no timestamps found in channel indicated for ' \
                                            f'{self.label} camera'
                return align_with_audio(raw_ts, audio, gpio, count,
                                        display=display,
                                        extrapolate_missing=extrapolate_missing)
            except AssertionError as ex:
                _logger.critical('Failed to extract using audio: %s', ex)

        # If you reach here extracting using audio TTLs was not possible
        _logger.warning('Alignment by wheel data not yet implemented')
        if length < raw_ts.size:
            df = raw_ts.size - length
            _logger.info(f'Discarding first {df} pulses')
            raw_ts = raw_ts[df:]
        return raw_ts


class CameraTimestampsCamlog(BaseExtractor):
    def __init__(self, label, session_path=None):
        super().__init__(session_path)
        self.label = assert_valid_label(label)
        self.save_names = f'_ibl_{label}Camera.times.npy'
        self.var_names = f'{label}_camera_timestamps'
        self._log_level = _logger.level
        _logger.setLevel(logging.DEBUG)

    def __del__(self):
        _logger.setLevel(self._log_level)

    def _extract(self, sync=None, chmap=None, video_path=None,
                 display=False, extrapolate_missing=True, **kwargs):

        fpga_times = extract_camera_sync(sync=sync, chmap=chmap)
        video_frames = get_video_length(self.session_path.joinpath('raw_video_data', f'_iblrig_{self.label}Camera.raw.mp4'))
        raw_ts = fpga_times[self.label]

        # For left camera sometimes we have one extra pulse than video frame
        if (raw_ts.size - video_frames) == 1:
            _logger.warning(f'One extra sync pulse detected for {self.label} camera')
            raw_ts = raw_ts[:-1]
        elif (raw_ts.size - video_frames) == -1:
            _logger.warning(f'One extra video frame detected for {self.label} camera')
            med_time = np.median(np.diff(raw_ts))
            raw_ts = np.r_[raw_ts, np.array([raw_ts[-1] + med_time])]

        assert video_frames == raw_ts.size, f'dimension mismatch between video frames and TTL pulses for {self.label} camera' \
                                            f'by {np.abs(video_frames - raw_ts.size)} frames'

        return raw_ts


class CameraTimestampsBpod(BaseBpodTrialsExtractor):
    """
    Get the camera timestamps from the Bpod

    The camera events are logged only during the events not in between, so the times need
    to be interpolated
    """
    save_names = '_ibl_leftCamera.times.npy'
    var_names = 'left_camera_timestamps'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_level = _logger.level
        _logger.setLevel(logging.DEBUG)

    def __del__(self):
        _logger.setLevel(self._log_level)

    def _extract(self, video_path=None, display=False, extrapolate_missing=True, **kwargs):
        """
        The raw timestamps are taken from the Bpod. These are the times of the camera's frame TTLs.
        If the pin state file exists, these timestamps are aligned to the video frames using the
        audio TTLs.  Frames missing from the embedded frame count are removed from the timestamps
        array.
        If the pin state file does not exist, the left camera timestamps may be aligned using the
        wheel data.
        :param video_path: an optional path for fetching the number of frames.  If None,
        the video is loaded from the session path.  If an int is provided this is taken to be
        the total number of frames.
        :param display: if True, the audio and GPIO fronts are plotted.
        :param extrapolate_missing: if True, any missing timestamps at the beginning and end of
        the session are extrapolated based on the median frame rate, otherwise they will be NaNs.
        :return: a numpy array of camera timestamps
        """
        raw_ts = self._times_from_bpod()
        count, (*_, gpio) = raw.load_embedded_frame_data(self.session_path, 'left')
        if video_path is None:
            filename = '_iblrig_leftCamera.raw.mp4'
            video_path = self.session_path.joinpath('raw_video_data', filename)
        # Permit the video path to be the length for development and debugging purposes
        length = video_path if isinstance(video_path, int) else get_video_length(video_path)
        _logger.debug(f'Number of video frames = {length}')

        # Check if the GPIO is usable for extraction.  GPIO is None if the file does not exist,
        # is empty, or contains only one value (i.e. doesn't change)
        if gpio is not None and gpio['indices'].size > 1:
            _logger.info('Aligning to audio TTLs')
            # Extract audio TTLs
            _, audio = raw.load_bpod_fronts(self.session_path, data=self.bpod_trials)
            _, ts = raw.load_camera_ssv_times(self.session_path, 'left')
            """
            There are many audio TTLs that are for some reason missed by the GPIO.  Conversely
            the last GPIO doesn't often correspond to any audio TTL.  These will be removed.
            The drift appears to be less severe than the FPGA, so when assigning TTLs we'll take
            the nearest TTL within 500ms.  The go cue TTLs comprise two short pulses ~3ms apart.
            We will fuse any TTLs less than 5ms apart to make assignment more accurate.
            """
            try:
                gpio, audio, ts = groom_pin_state(gpio, audio, ts, take='nearest',
                                                  tolerance=.5, min_diff=5e-3, display=display)
                if count.size > length:
                    count = count[:length]
                else:
                    assert length == count.size, 'fewer counts than frames'

                return align_with_audio(raw_ts, audio, gpio, count,
                                        extrapolate_missing, display=display)
            except AssertionError as ex:
                _logger.critical('Failed to extract using audio: %s', ex)

        # If you reach here extracting using audio TTLs was not possible
        _logger.warning('Alignment by wheel data not yet implemented')
        # Extrapolate at median frame rate
        n_missing = length - raw_ts.size
        if n_missing > 0:
            _logger.warning(f'{n_missing} fewer Bpod timestamps than frames; '
                            f'{"extrapolating" if extrapolate_missing else "appending nans"}')
            frate = np.median(np.diff(raw_ts))
            to_app = ((np.arange(n_missing, ) + 1) / frate + raw_ts[-1]
                      if extrapolate_missing
                      else np.full(n_missing, np.nan))
            raw_ts = np.r_[raw_ts, to_app]  # Append the missing times
        elif n_missing < 0:
            _logger.warning(f'{abs(n_missing)} fewer frames than Bpod timestamps')
            _logger.info(f'Discarding first {abs(n_missing)} pulses')
            raw_ts = raw_ts[abs(n_missing):]

        return raw_ts

    def _times_from_bpod(self):
        ntrials = len(self.bpod_trials)

        cam_times = []
        n_frames = 0
        n_out_of_sync = 0
        missed_trials = []
        for ind in np.arange(ntrials):
            # get upgoing and downgoing fronts
            pin = np.array(self.bpod_trials[ind]['behavior_data']
                           ['Events timestamps'].get('Port1In'))
            pout = np.array(self.bpod_trials[ind]['behavior_data']
                            ['Events timestamps'].get('Port1Out'))
            # some trials at startup may not have the camera working, discard
            if np.all(pin) is None:
                missed_trials.append(ind)
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

        if missed_trials:
            _logger.debug('trial(s) %s missing TTL events', range_str(missed_trials))
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
    # assert count.size == pin_state.size, 'frame count and pin state size mismatch'
    assert all(np.diff(count) > 0), 'frame count not strictly increasing'
    assert all(np.diff(timestamps) > 0), 'FPGA/Bpod camera times not strictly increasing'
    same_n_ttl = pin_state['times'].size == audio['times'].size
    assert same_n_ttl, 'more audio TTLs detected on camera than TTLs sent'

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
    first_uptick = pin_state['indices'][0]
    first_ttl = np.searchsorted(timestamps, audio['times'][0])
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
    # Get approximate frame rate for extrapolating timestamps (if required)
    frate = round(1 / np.nanmedian(np.diff(timestamps)))

    if start < 0:
        n_missing = abs(start)
        _logger.warning(f'{n_missing} missing FPGA/Bpod timestamp(s) at start; '
                        f'{"extrapolating" if extrapolate_missing else "prepending nans"}')
        to_app = (timestamps[0] - (np.arange(n_missing, 0, -1) + 1) / frate
                  if extrapolate_missing
                  else np.full(n_missing, np.nan))
        timestamps = np.r_[to_app, timestamps]  # Prepend the missing times
        start = 0

    # Remove the extraneous timestamps from the beginning and end
    end = count[-1] + 1 + start
    ts = timestamps[start:end]
    n_missing = count[-1] - ts.size + 1
    if n_missing > 0:
        # if (n_missing := count[-1] - ts.size + 1) > 0:  # py3.8
        """
        For ephys sessions there may be fewer FPGA times than frame counts if SpikeGLX is turned
        off before the video acquisition workflow.  For Bpod this always occurs because Bpod
        finishes before the camera workflow.  For Bpod the times are already extrapolated for
        these late frames."""
        _logger.warning(f'{n_missing} fewer FPGA/Bpod timestamps than frame counts; '
                        f'{"extrapolating" if extrapolate_missing else "appending nans"}')
        to_app = ((np.arange(n_missing, ) + 1) / frate + ts[-1]
                  if extrapolate_missing
                  else np.full(n_missing, np.nan))
        ts = np.r_[ts, to_app]  # Append the missing times
    assert ts.size >= count.size, 'fewer timestamps than frame counts'
    assert ts.size == count[-1] + 1, 'more frames recorded in frame count than timestamps '

    # Remove the rest of the dropped frames
    ts = ts[count]
    assert np.searchsorted(ts, audio['times'][0]) == first_uptick,\
        'time of first audio TTL doesn\'t match after alignment'
    if ts.size != count.size:
        _logger.error('number of timestamps and frames don\'t match after alignment')

    if display:
        # Plot to check
        fig, axes = plt.subplots(1, 1)
        y = within_ranges(np.arange(ts.size), pin_state['indices'].reshape(-1, 2)).astype(float)
        y *= 1e-5  # For scale when zoomed in
        axes.plot(ts, y, marker='d', color='blue', drawstyle='steps-pre', label='GPIO')
        axes.plot(ts, np.zeros_like(ts), 'kx', label='FPGA timestamps')
        vertical_lines(audio['times'], ymin=0, ymax=1e-5,
                       color='r', linestyle=':', ax=axes, label='audio TTL')
        plt.legend()

    return ts


def attribute_times(arr, events, tol=.1, injective=True, take='first'):
    """
    Returns the values of the first array that correspond to those of the second.

    Given two arrays of timestamps, the function will return the values of the first array
    that most likely correspond to the values of the second.  For each of the values in the
    second array, the absolute difference is taken and the index of either the first sufficiently
    close value, or simply the closest one, is assigned.

    If injective is True, once a value has been assigned, to a value it can't be assigned to
    another.  In other words there is a one-to-one mapping between the two arrays.

    :param arr: An array of event times to attribute to those in `events`
    :param events: An array of event times considered a subset of `arr`
    :param tol: The max absolute difference between values in order to be considered a match
    :param injective: If true, once a value has been assigned it will not be assigned again
    :param take: If 'first' the first value within tolerance is assigned; if 'nearest' the
    closest value is assigned
    :returns Numpy array the same length as `values`
    """
    take = take.lower()
    if take not in ('first', 'nearest'):
        raise ValueError('Parameter `take` must be either "first" or "nearest"')
    stack = np.ma.masked_invalid(arr, copy=False)
    stack.fill_value = np.inf
    assigned = np.full(events.shape, -1, dtype=int)  # Initialize output array
    for i, x in enumerate(events):
        dx = np.abs(stack.filled() - x)
        if dx.min() < tol:  # is any value within tolerance
            idx = np.where(dx < tol)[0][0] if take == 'first' else dx.argmin()
            assigned[i] = idx
            stack.mask[idx] = injective  # If one-to-one, remove the assigned value
    return assigned


def groom_pin_state(gpio, audio, ts, tolerance=2., display=False, take='first', min_diff=0.):
    """
    Align the GPIO pin state to the FPGA audio TTLs.  Any audio TTLs not reflected in the pin
    state are removed from the dict and the times of the detected fronts are converted to FPGA
    time.  At the end of this the number of GPIO fronts should equal the number of audio fronts.

    Note:
      - This function is ultra safe: we probably don't need assign all the ups and down fronts
      separately and could potentially even align the timestamps without removing the missed fronts
      - The input gpio and audio dicts may be modified by this function
      - For training sessions the frame rate is only 30Hz and the TTLs tend to be broken up by
      small gaps.  Setting the min_diff to 5ms helps the timestamp assignment accuracy.
    :param gpio: array of GPIO pin state values
    :param audio: dict of FPGA audio TTLs (see ibllib.io.extractors.ephys_fpga._get_sync_fronts)
    :param ts: camera frame times
    :param tolerance: two pulses need to be within this many seconds to be considered related
    :param take:  If 'first' the first value within tolerance is assigned; if 'nearest' the
    closest value is assigned
    :param display: If true, the resulting timestamps are plotted against the raw audio signal
    :param min_diff: Audio TTL fronts less than min_diff seconds apart will be removed
    :returns: dict of GPIO FPGA front indices, polarities and FPGA aligned times
    :returns: audio times and polarities sans the TTLs not detected in the frame data
    :returns: frame times in FPGA time
    """
    # Check that the dimensions match
    if np.any(gpio['indices'] >= ts.size):
        _logger.warning('GPIO events occurring beyond timestamps array length')
        keep = gpio['indices'] < ts.size
        gpio = {k: gpio[k][keep] for k, v in gpio.items()}
    assert audio and audio['times'].size > 0, 'no audio TTLs for session'
    assert audio['times'].size == audio['polarities'].size, 'audio data dimension mismatch'
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert np.all(np.abs(np.diff(audio['polarities'])) == 2), 'consecutive high/low audio events'
    # make sure first TTL is high
    assert audio['polarities'][0] == 1
    # make sure audio times in order
    assert np.all(np.diff(audio['times']) > 0)
    # make sure raw timestamps increase
    assert np.all(np.diff(ts) > 0), 'timestamps must strictly increase'
    # make sure there are state changes
    assert gpio['indices'].any(), 'no TTLs detected in GPIO'
    # # make sure first GPIO state is high
    assert gpio['polarities'][0] == 1
    """
    Some audio TTLs appear to be so short that they are not recorded by the camera.  These can
    be as short as a few microseconds.  Applying a cutoff based on framerate was unsuccessful.
    Assigning each audio TTL to each pin state change is not easy because some onsets occur very
    close together (sometimes < 70ms), on the order of the delay between TTL and frame time.
    Also, the two clocks have some degree of drift, so the delay between audio TTL and pin state
    change may be zero or even negative.

    Here we split the events into audio onsets (lo->hi) and audio offsets (hi->lo).  For each
    uptick in the GPIO pin state, we take the first audio onset time that was within 100ms of it.
    We ensure that each audio TTL is assigned only once, so a TTL that is closer to frame 3 than
    frame 1 may still be assigned to frame 1.
    """
    ifronts = gpio['indices']  # The pin state flips
    audio_times = audio['times']
    if ifronts.size != audio['times'].size:
        _logger.warning('more audio TTLs than GPIO state changes, assigning timestamps')
        to_remove = np.zeros(ifronts.size, dtype=bool)  # unassigned GPIO fronts to remove
        low2high = ifronts[gpio['polarities'] == 1]
        high2low = ifronts[gpio['polarities'] == -1]
        assert low2high.size >= high2low.size

        # Remove and/or fuse short TTLs
        if min_diff > 0:
            short, = np.where(np.diff(audio['times']) < min_diff)
            audio_times = np.delete(audio['times'], np.r_[short, short + 1])
            _logger.debug(f'Removed {short.size * 2} fronts TLLs less than '
                          f'{min_diff * 1e3:.0f}ms apart')
            assert audio_times.size > 0, f'all audio TTLs less than {min_diff}s'

        # Onsets
        ups = ts[low2high] - ts[low2high][0]  # times relative to first GPIO high
        onsets = audio_times[::2] - audio_times[0]  # audio times relative to first onset
        # assign GPIO fronts to audio onset
        assigned = attribute_times(onsets, ups, tol=tolerance, take=take)
        unassigned = np.setdiff1d(np.arange(onsets.size), assigned[assigned > -1])
        if unassigned.size > 0:
            _logger.debug(f'{unassigned.size} audio TTL rises were not detected by the camera')
        # Check that all pin state upticks could be attributed to an onset TTL
        missed = assigned == -1
        if np.any(missed):
            # if np.any(missed := assigned == -1):  # py3.8
            _logger.warning(f'{sum(missed)} pin state rises could '
                            f'not be attributed to an audio TTL')
            if display:
                ax = plt.subplot()
                vertical_lines(ups[assigned > -1],
                               linestyle='-', color='g', ax=ax,
                               label='assigned GPIO up state')
                vertical_lines(ups[missed],
                               linestyle='-', color='r', ax=ax,
                               label='unassigned GPIO up state')
                vertical_lines(onsets[unassigned],
                               linestyle=':', color='k', ax=ax,
                               alpha=0.3, label='audio onset')
                vertical_lines(onsets[assigned],
                               linestyle=':', color='b', ax=ax, label='assigned audio onset')
                plt.legend()
                plt.show()
            # Remove the missed fronts
            to_remove = np.in1d(gpio['indices'], low2high[missed])
            assigned = assigned[~missed]
        onsets_ = audio_times[::2][assigned]

        # Offsets
        downs = ts[high2low] - ts[high2low][0]
        offsets = audio_times[1::2] - audio_times[1]
        assigned = attribute_times(offsets, downs, tol=tolerance, take=take)
        unassigned = np.setdiff1d(np.arange(offsets.size), assigned[assigned > -1])
        if unassigned.size > 0:
            _logger.debug(f'{unassigned.size} audio TTL falls were not detected by the camera')
        # Check that all pin state downticks could be attributed to an offset TTL
        missed = assigned == -1
        if np.any(missed):
            # if np.any(missed := assigned == -1):  # py3.8
            _logger.warning(f'{sum(missed)} pin state falls could '
                            f'not be attributed to an audio TTL')
            # Remove the missed fronts
            to_remove |= np.in1d(gpio['indices'], high2low[missed])
            assigned = assigned[~missed]
        offsets_ = audio_times[1::2][assigned]

        # Audio groomed
        if np.any(to_remove):
            # Check for any orphaned fronts (only one pin state edge was assigned)
            to_remove = np.pad(to_remove, (0, to_remove.size % 2), 'edge')  # Ensure even size
            # Perform xor to find GPIOs where only onset or offset is marked for removal
            orphaned = to_remove.reshape(-1, 2).sum(axis=1) == 1
            if orphaned.any():
                """If there are orphaned GPIO fronts (i.e. only one edge was assigned to an
                audio front), remove the orphaned front its assigned audio TTL. In other words
                if both edges cannot be assigned to an audio TTL, we ignore the TTL entirely.
                This is a sign that the assignment was bad and extraction may fail."""
                _logger.warning('Some onsets but not offsets (or vice versa) were not assigned; '
                                'this may be a sign of faulty wiring or clock drift')
                # Find indices of GPIO upticks where only the downtick was marked for removal
                orphaned_onsets, =  np.where(~to_remove.reshape(-1, 2)[:, 0] & orphaned)
                # The onsets_ array already has the other TTLs removed (same size as to_remove ==
                # False) so subtract the number of removed elements from index.
                for i, v in enumerate(orphaned_onsets):
                    orphaned_onsets[i] -= to_remove.reshape(-1, 2)[:v, 0].sum()
                # Same for offsets...
                orphaned_offsets, =  np.where(~to_remove.reshape(-1, 2)[:, 1] & orphaned)
                for i, v in enumerate(orphaned_offsets):
                    orphaned_offsets[i] -= to_remove.reshape(-1, 2)[:v, 1].sum()
                # Remove orphaned audio onsets and offsets
                onsets_ = np.delete(onsets_, orphaned_onsets[orphaned_onsets < onsets_.size])
                offsets_ = np.delete(offsets_, orphaned_offsets[orphaned_offsets < offsets_.size])
                _logger.debug(f'{orphaned.sum()} orphaned TTLs removed')
                to_remove.reshape(-1, 2)[orphaned] = True

            # Remove those unassigned GPIOs
            gpio = {k: v[~to_remove[:v.size]] for k, v in gpio.items()}
            ifronts = gpio['indices']

            # Assert that we've removed discrete TTLs
            # A failure means e.g. an up-going front of one TTL was missed
            # but not the down-going one.
            assert np.all(np.abs(np.diff(gpio['polarities'])) == 2)
            assert gpio['polarities'][0] == 1

        audio_ = {'times': np.empty(ifronts.size), 'polarities': gpio['polarities']}
        audio_['times'][::2] = onsets_
        audio_['times'][1::2] = offsets_
    else:
        audio_ = audio

    # Align the frame times to FPGA
    fcn_a2b, drift_ppm = dsp.sync_timestamps(ts[ifronts], audio_['times'])
    _logger.debug(f'frame audio alignment drift = {drift_ppm:.2f}ppm')
    # Add times to GPIO dict
    gpio['times'] = fcn_a2b(ts[ifronts])

    if display:
        # Plot all the onsets and offsets
        ax = plt.subplot()
        # All Audio TTLS
        squares(audio['times'], audio['polarities'],
                ax=ax, label='audio TTLs', linestyle=':', color='k', yrange=[0, 1], alpha=0.3)
        # GPIO
        x = np.insert(gpio['times'], 0, 0)
        y = np.arange(x.size) % 2
        squares(x, y, ax=ax, label='GPIO')
        y = within_ranges(np.arange(ts.size), ifronts.reshape(-1, 2))  # 0 or 1 for each frame
        ax.plot(fcn_a2b(ts), y, 'kx', label='cam times')
        # Assigned audio
        squares(audio_['times'], audio_['polarities'],
                ax=ax, label='assigned audio TTL', linestyle=':', color='g', yrange=[0, 1])
        ax.legend()
        plt.xlabel('FPGA time (s)')
        ax.set_yticks([0, 1])
        ax.set_title('GPIO - audio TTL alignment')
        plt.show()

    return gpio, audio_, fcn_a2b(ts)


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

    sync_collection = kwargs.get('sync_collection', 'raw_ephys_data')
    camlog = kwargs.get('camlog', False)

    if session_type is None:  # infer the session type
        session_type = get_session_extractor_type(session_path)
        if not session_type or session_type not in _get_task_types_json_config().values():
            raise ValueError(f"Session type {session_type} has no matching extractor")
    sync_type = kwargs.get('sync_type', 'nidq' if session_type == 'ephys' else 'bpod')

    if sync_type == 'nidq':
        labels = assert_valid_label(kwargs.pop('labels', ('left', 'right', 'body')))
        labels = (labels,) if isinstance(labels, str) else labels  # Ensure list/tuple
        CamExtractor = CameraTimestampsCamlog if camlog else CameraTimestampsFPGA
        extractor = [partial(CamExtractor, label) for label in labels]
        if 'sync' not in kwargs:
            kwargs['sync'], kwargs['chmap'] = get_sync_and_chn_map(session_path, sync_collection)
    else:  # assume Bpod otherwise
        assert kwargs.pop('labels', 'left'), 'only left camera is currently supported'
        extractor = CameraTimestampsBpod

    outputs, files = run_extractor_classes(
        extractor, session_path=session_path, save=save, **kwargs)
    return outputs, files

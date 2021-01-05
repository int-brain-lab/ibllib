"""Camera QC
This module runs a list of quality control metrics on the camera and extracted video data.

Example - Run camera QC, downloading all but video file
    qc = CameraQC(eid, download_data=True, stream=True)
    qc.run()

Question:
    We're not extracting the audio based on TTL length.  Is this a problem?
"""
import logging
from inspect import getmembers, isfunction
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

from ibllib.io.extractors.camera import (
    load_embedded_frame_data, extract_camera_sync, PIN_STATE_THRESHOLD, extract_all
)
from ibllib.exceptions import ALFObjectNotFound
from ibllib.io.extractors import ephys_fpga, training_wheel
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io import raw_data_loaders as raw
import alf.io as alfio
from brainbox.core import Bunch
import brainbox.behavior.wheel as wh
from brainbox.io.one import path_to_url, datasets_from_type
from brainbox.io.video import get_video_meta, get_video_frames_preload
from brainbox.video import frame_diff
from . import base

_log = logging.getLogger('ibllib')


class CameraQC(base.QC):
    """A class for computing camera QC metrics"""
    dstypes = [
        '_iblrig_Camera.frame_counter',
        '_iblrig_Camera.GPIO',
        '_iblrig_Camera.timestamps',
        '_iblrig_taskData.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_Camera.raw',
        'camera.times'
    ]
    dstypes_fpga = [
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        'ephysData.raw.meta',
        'ephysData.raw.wiring'
    ]
    """Recall that for the training rig there is only one side camera at 30 Hz and 1280 x 1024 px. 
    For the recording rig there are two side cameras (left: 60 Hz, 1280 x 1024 px; 
    right: 150 Hz, 640 x 512 px) and one body camera (30 Hz, 640 x 512 px). """
    video_meta = {
        'training': {
            'left': {
                'fps': 30,
                'width': 1280,
                'height': 1024
            }
        },
        'ephys': {
            'left': {
                'fps': 60,
                'width': 1280,
                'height': 1024
            },
            'right': {
                'fps': 150,
                'width': 640,
                'height': 512
            },
            'body': {
                'fps': 30,
                'width': 640,
                'height': 512
            },
        }
    }

    def __init__(self, session_path_or_eid, side, **kwargs):
        """
        :param session_path_or_eid: A session eid or path
        :param log: A logging.Logger instance, if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        :param camera: The camera to run QC on, if None QC is run for all three cameras
        :param stream: If true and local video files not available, the data are streamed from
        the remote source.
        :param n_samples: The number of frames to sample for the position and brightness QC
        """
        # When an eid is provided, we will download the required data by default (if necessary)
        download_data = not alfio.is_session_path(session_path_or_eid)
        self.download_data = kwargs.pop('download_data', download_data)
        self.stream = kwargs.pop('stream', True)
        self.n_samples = kwargs.pop('n_samples', 100)
        super().__init__(session_path_or_eid, **kwargs)

        # Data
        self.side = side
        filename = f'_iblrig_{self.side}Camera.raw.mp4'
        self.video_path = self.session_path / 'raw_video_data' / filename
        # If local video doesn't exist, change video path to URL
        if not self.video_path.exists() and self.stream and self.one is not None:
            self.video_path = path_to_url(self.video_path, self.one)

        logging.disable(logging.CRITICAL)
        self.type = get_session_extractor_type(self.session_path) or None
        logging.disable(logging.NOTSET)
        keys = ('count', 'pin_state', 'audio', 'fpga_times', 'wheel', 'video', 'frame_samples')
        self.data = Bunch.fromkeys(keys)
        self.frame_samples_idx = None

        # QC outcomes map
        self.metrics = None
        self.outcome = 'NOT_SET'

    def load_data(self, download_data: bool = None, extract_times: bool = False) -> None:
        """Extract the data from raw data files
        Extracts all the required task data from the raw data files.

        :param download_data: if True, any missing raw data is downloaded via ONE.
        :param extract_times: if True, the camera.times are re-extracted from the raw data
        """
        if download_data is not None:
            self.download_data = download_data
        if self.one:
            self._ensure_required_data()
        _log.info('Gathering data for QC')

        # Get frame count and pin state
        self.data['count'], self.data['pin_state'] = \
            load_embedded_frame_data(self.session_path, self.side, raw=True)

        # Get audio and wheel data
        wheel_keys = ('timestamps', 'position')
        if self.type == 'ephys':
            sync, chmap = ephys_fpga.get_main_probe_sync(self.session_path)
            wheel_data = ephys_fpga.extract_wheel_sync(sync, chmap)
            audio_ttls = ephys_fpga._get_sync_fronts(sync, chmap['audio'])
            self.data['audio'] = audio_ttls['times'][::2]  # Get rises
            # Load raw FPGA times
            cam_ts = extract_camera_sync(sync, chmap)
            self.data['fpga_times'] = cam_ts[f'{self.side}_camera']
        else:
            bpod_data = raw.load_data(self.session_path)
            wheel_data = training_wheel.get_wheel_position(self.session_path)
            _, audio_ttls = raw.load_bpod_fronts(self.session_path, bpod_data)
            self.data['audio'] = audio_ttls['times'][::2]
        self.data['wheel'] = Bunch(zip(wheel_keys, wheel_data))

        # Find short period of wheel motion for motion correlation.  For speed start with the
        # fist 2 minutes (nearly always enough), extract wheel movements and pick one.
        # TODO Pick movement towards the end of the session (but not right at the end as some
        #  are extrapolated).  Make sure the movement isn't too long.
        START = 1 * 60  # Start 1 minute in
        SEARCH_PERIOD = 2 * 60
        ts, pos = wheel_data
        while True:
            win = np.logical_and(
                ts > START,
                ts < SEARCH_PERIOD + START
            )
            if np.sum(win) > 1000:
                break
            SEARCH_PERIOD *= 2
        wheel_moves = training_wheel.extract_wheel_moves(ts[win], pos[win])
        move_ind = np.argmax(np.abs(wheel_moves['peakAmplitude']))
        # TODO Save only the wheel fragment we need
        self.data['wheel'].period = wheel_moves['intervals'][move_ind, :]

        # Load extracted frame times
        alf_path = self.session_path / 'alf'
        try:
            assert not extract_times
            self.data['frame_times'] = alfio.load_object(alf_path, f'{self.side}Camera')['times']
        except (ALFObjectNotFound, AssertionError):  # Re-extract
            kwargs = dict(video_paths=self.video_path, camera=self.side)
            if self.type == 'ephys':
                kwargs = {**kwargs, 'sync': sync, 'chmap': chmap}  # noqa
            outputs, _ = extract_all(self.session_path, self.type, save=False, **kwargs)
            self.data['frame_times'] = outputs[f'{self.side}_camera_timestamps'][self.side]

        # Gather information from video file
        _log.info('Inspecting video file...')
        # Get basic properties of video
        try:
            self.data['video'] = get_video_meta(self.video_path, one=self.one)
            # Sample some frames from the video file
            indices = np.linspace(100, self.data['video'].length - 100, self.n_samples).astype(int)
            self.frame_samples_idx = indices
            self.data['frame_samples'] = get_video_frames_preload(self.video_path, indices,
                                                                  mask=np.s_[:, :, 0])
            if self.data['wheel']:
                ROI = {
                    'left': ((800, 1020), (233, 1096)),
                    'right': ((426, 510), (104, 545)),
                    'body': ((402, 481), (31, 103))
                }
                roi = (*[slice(*r) for r in ROI[self.side]], 0)
                indices, = np.where(np.logical_and(
                    self.data['frame_times'] > self.data.wheel.period[0],
                    self.data['frame_times'] < self.data.wheel.period[1]
                ))
                self.data['wheel_frames'] = get_video_frames_preload(self.video_path, indices,
                                                                     mask=roi)
        except AssertionError:
            _log.error('Failed to read video file; setting outcome to CRITICAL')
            self._outcome = 'CRITICAL'

        # Load Bonsai frame timestamps
        ssv_times = raw.load_camera_ssv_times(self.session_path, self.side)
        self.data['bonsai_times'], self.data['camera_times'] = ssv_times

    def _ensure_required_data(self):
        """
        Ensures the datasets required for QC are local.  If the download_data attribute is True,
        any missing data are downloaded.  If all the data are not present locally at the end of
        it an exception is raised.  If the stream attribute is True, the video file is not
        required to be local, however it must be remotely accessible.

        TODO make static method with side as optional arg
        :return:
        """
        assert self.one is not None, 'ONE required to download data'
        # Get extractor type
        is_ephys = 'ephys' in (self.type or self.one.get_details(self.eid)['task_protocol'])
        dtypes = self.dstypes + self.dstypes_fpga if is_ephys else self.dstypes
        for dstype in dtypes:
            dataset = datasets_from_type(self.eid, dstype, self.one)
            if 'camera' in dstype.lower():  # Download individual camera file
                dataset = [d for d in dataset if self.side in d]
            if any(x.endswith('.mp4') for x in dataset) and self.stream:
                names = [x.name for x in self.one.list(self.eid)]
                assert f'_iblrig_{self.side}Camera.raw.mp4' in names, 'No remote video file found'
                continue
            required = (dstype not in ('camera.times', '_iblrig_Camera.raw'))
            collection = 'raw_behavior_data' if dstype == '_iblrig_taskSettings.raw' else None
            kwargs = {'download_only': True, 'collection': collection}
            present = (
                (self.one.load_dataset(self.eid, d, **kwargs) for d in dataset)
                if self.download_data
                else (next(self.session_path.rglob(d), None) for d in dataset)
            )
            assert (dataset and all(present)) or not required, f'Dataset {dstype} not found'
        self.type = get_session_extractor_type(self.session_path)

    def run(self, update: bool = False, **kwargs) -> (str, dict):
        """
        Run video QC checks and return outcome
        :param update: if True, updates the session QC fields on Alyx
        :param download_data: if True, downloads any missing data if required
        :param extract_times: if True, re-extracts the camera timestamps from the raw data
        :returns: overall outcome as a str, a dict of checks and their outcomes
        TODO Ensure that when pinstate QC NOT_SET it is not used in overall outcome
        """
        _log.info('Computing QC outcome')
        namespace = f'video{self.side.capitalize()}'
        if not self.data:
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(CameraQC, is_metric)
        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}
        all_pass = all(x is None or x == 'PASS' for x in self.metrics.values())
        self.outcome = 'PASS' if all_pass else 'FAIL'
        if update:
            bool_map = {k: None if v is None else v == 'PASS' for k, v in self.metrics.items()}
            self.update_extended_qc(bool_map)
            self.update(self.outcome, namespace)
        return self.outcome, self.metrics

    def check_brightness(self, bounds=(20, 100), max_std=20, display=False):
        """Check that the video brightness is within a given range
        The mean brightness of each frame must be with the bounds provided, and the standard
        deviation across samples frames should be less then the given value.  Assumes that the
        frame samples are 2D (no colour channels).

        :param bounds: For each frame, check that: bounds[0] < M < bounds[1], where M = mean(frame)
        :param max_std: The standard deviation of the frame luminance means must be less than this
        :param display: When True the mean frame luminance is plotted against sample frames.
        The sample frames with the lowest and highest mean luminance are shown.
        """
        if self.data['frame_samples'] is None:
            return 'NOT_SET'
        brightness = self.data['frame_samples'].mean(axis=(1, 2))
        # dims = self.data['frame_samples'].shape
        # brightness /= np.array((*dims[1:], 255)).prod()  # Normalize

        within_range = np.logical_and(brightness > bounds[0],
                                      brightness < bounds[1])
        passed = within_range.all() and np.std(brightness) < max_std
        if display:
            f = plt.figure()
            gs = f.add_gridspec(2, 3)
            indices = self.frame_samples_idx
            # Plot mean frame luminance
            ax = f.add_subplot(gs[:2, :2])
            plt.plot(indices, brightness, label='brightness')
            ax.set(
                xlabel='frame #',
                ylabel='brightness (mean pixel)',
                title='Brightness')
            ax.hlines(bounds, 0, indices[-1], colors='r', linestyles=':', label='bounds')
            ax.legend()
            # Plot min-max frames
            for i, idx in enumerate((np.argmax(brightness), np.argmin(brightness))):
                a = f.add_subplot(gs[i, 2])
                ax.annotate('*',  (indices[idx], brightness[idx]),  # this is the point to label
                            textcoords="offset points", xytext=(0, 1),  ha='center')
                frame = self.data['frame_samples'][idx]
                title = ('min' if i else 'max') + ' mean luminance = %.2f' % brightness[idx]
                self.imshow(frame, ax=a, title=title)
        return 'PASS' if passed else 'FAIL'

    def check_file_headers(self):
        """Check reported frame rate matches FPGA frame rate"""
        if None in (self.data['video'], self.video_meta):
            return 'NOT_SET'
        expected = self.video_meta[self.type][self.side]
        return 'PASS' if self.data['video']['fps'] == expected['fps'] else 'FAIL'

    def check_framerate(self, threshold=1.):
        """Check camera times match specified frame rate for camera

        :param threshold: The maximum absolute difference between timestamp sample rate and video
        frame rate.  NB: Does not take into account dropped frames.
        """
        if any( x is None for x in (self.data['frame_times'], self.video_meta)):
            return 'NOT_SET'
        fps = self.video_meta[self.type][self.side]['fps']
        Fs = 1 / np.median(np.diff(self.data['frame_times']))  # Approx. frequency of camera
        return 'PASS' if abs(Fs - fps) < threshold else 'FAIL'

    def check_pin_state(self, display=False):
        """Check the pin state reflects Bpod TTLs"""
        if self.data['pin_state'] is None:
            return 'NOT_SET'
        size_matches = self.data['video']['length'] == self.data['pin_state'].size
        # There should be only one value below our threshold
        correct_threshold = sum(np.unique(self.data['pin_state']) < PIN_STATE_THRESHOLD) == 1
        state = self.data['pin_state'] > PIN_STATE_THRESHOLD
        # NB: The pin state to be high for 2 consecutive frames
        low2high = np.insert(np.diff(state.astype(int)) == 1, 0, False)
        state_ttl_matches = sum(low2high) == self.data['audio'].size
        # Check within ms of audio times
        if display:
            plt.Figure()
            plt.plot(self.data['frame_times'][low2high], np.zeros(sum(low2high)), 'o',
                     label='GPIO Low -> High')
            plt.plot(self.data['audio'], np.zeros(self.data['audio'].size), 'rx',
                     label='Audio TTL High')
            plt.xlabel('FPGA frame times / s')
            plt.gca().set(yticklabels=[])
            plt.gca().tick_params(left=False)
            plt.legend()
        # idx = [i for i, x in enumerate(self.data['audio'])
        #        if np.abs(x - self.data['frame_times'][low2high]).min() > 0.01]
        # mins = [np.abs(x - self.data['frame_times'][low2high]).min() for x in self.data['audio']]

        return 'PASS' if size_matches and correct_threshold and state_ttl_matches else 'FAIL'

    def check_dropped_frames(self, threshold=.1):
        """Check how many frames were reported missing

        :param threshold: The maximum allowable percentage of dropped frames
        """
        if None in (self.data['video'], self.data['count']):
            return 'NOT_SET'
        size_matches = self.data['video']['length'] == self.data['count'].size
        assert np.all(np.diff(self.data['count']) > 0), 'frame count not strictly increasing'
        dropped = np.diff(self.data['count']).astype(int) - 1
        pct_dropped = (sum(dropped) / len(dropped) * 100)
        return 'PASS' if size_matches and pct_dropped < threshold else 'FAIL'

    def check_timestamps(self):
        """Check that the camera.times array is reasonable"""
        if None in (self.data['frame_times'], self.data['video']):
            return 'NOT_SET'
        # Check frame rate matches what we expect
        expected = 1 / self.video_meta[self.type][self.side]['fps']
        # TODO Remove dropped frames from test
        frame_delta = np.diff(self.data['frame_times'])
        fps_matches = np.isclose(frame_delta.mean(), expected, atol=0.001)
        # Check number of timestamps matches video
        length_matches = self.data['frame_times'].size == self.data['video'].length
        # Check times are strictly increasing
        increasing = all(np.diff(self.data['frame_times']) > 0)
        # Check times do not contain nans
        nanless = not np.isnan(self.data['frame_times']).any()
        return 'PASS' if increasing and fps_matches and length_matches and nanless else 'FAIL'

    def check_resolution(self):
        """Check that the timestamps and video file resolution match what we expect"""
        if self.data['video'] is None:
            return 'NOT_SET'
        actual = self.data['video']
        expected = self.video_meta[self.type][self.side]
        match = actual['width'] == expected['width'] and actual['height'] == expected['height']
        return 'PASS' if match else 'FAIL'

    def check_wheel_alignment(self, display=False):
        """Check wheel motion in video correlates with the rotary encoder signal"""
        # TODO Try normalizing velocity first; try sampling wheel at frame rate
        if self.data['wheel_frames'] is None or self.data['wheel'] is None:
            return 'NOT_SET'

        def to_mask(ts):
            a, b = self.data['wheel'].get('period', (ts[0], ts[1]))
            return np.logical_and(ts > a, ts < b)

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        def frame_diffs(frames):
            df = []
            frame0 = frames[0]
            frame1 = frames[1]

            for i in range(2, len(frames)):
                frame2 = frames[i]
                df.append(frame_diff(frame0, frame2))
                frame0 = frame1
                frame1 = frame2
            return np.array(df)

        # Calculate rotary encoder velocity trace
        Fs = self.video_meta[self.type][self.side]['fps']
        ts, pos = self.data['wheel'].timestamps, self.data['wheel'].position
        mask = to_mask(ts)
        pos, ts = wh.interpolate_position(ts[mask], pos[mask], freq=Fs)
        v, _ = wh.velocity_smoothed(pos, Fs)
        # Calculate video motion trace
        frame_times = self.data['frame_times']
        frame_times = frame_times[to_mask(frame_times)]
        df_ = frame_diffs(self.data['wheel_frames'])

        xs = np.unique([find_nearest(ts, x) for x in frame_times])
        vs = np.abs(v[xs])
        vs = (vs - np.min(vs)) / (np.max(vs) - np.min(vs))

        # FIXME This can be used as a goodness of fit measure
        xcorr = signal.correlate(df_, vs)
        c = max(xcorr)
        xcorr = np.argmax(xcorr)
        dt_i = xcorr - xs.size
        self.log.info(f'{self.side} camera, adjusted by {dt_i} frames')

        if display:
            # mask = np.logical_and(ts > period[0], ts < period[1])
            # plt.plot(ts[mask], pos[mask])
            # x = x[1::2]
            fig, axes = plt.subplots(4, 1)
            axes[0].plot(ts, pos, label='wheel position')
            axes[0].set(
                xlabel='time / s',
                ylabel='position / rad',
                title='Wheel position'
            )
            axes[1].plot(ts, np.abs(v), label='rotary encoder velocity')
            axes[1].set(
                xlabel='Time (s)',
                ylabel='Abs wheel velocity (rad / s)',
                title='Wheel velocity'
            )
            # FIXME Shouldn't be missing frames in diff
            axes[2].plot(frame_times[1:-1], df_, '-x', label='wheel motion energy')
            # ax[0].vlines(x[np.array(thresh)], 0, 1,
            #              linewidth=0.5, linestyle=':', label='>%i s.d. diff' % sd_thresh)
            v_abs = np.abs(v)
            v_normed = (v_abs - np.min(v_abs)) / (np.max(v_abs) - np.min(v_abs))
            axes[3].plot(ts, v_normed, label='Normalized absolute velocity')
            dt = np.diff(frame_times[[0, np.abs(dt_i)]])
            axes[3].plot(ts[xs] - dt, vs, 'r-x', label='velocity (shifted)')
            axes[3].set_title('normalized motion energy, %s camera, %.0f fps' % (self.side, 60))
            axes[3].set_ylabel('rate of change (a.u.)')
            axes[3].legend()
            plt.tight_layout()
        return 'PASS' if dt_i == 0 else 'FAIL'

    def check_position(self, hist_thresh=0.7, pos_thresh=100, metric=cv2.TM_CCOEFF_NORMED,
                       display=False, test=False, roi=None):
        """Check camera is positioned correctly
        For the template matching zero-normalized cross-correlation (default) should be more
        robust to exposure (which we're not checking here).  The L2 norm (TM_SQDIFF) should
        also work.

        :param hist_thresh: The minimum histogram cross-correlation threshold to pass (0-1).
        :param pos_thresh: The maximum number of pixels off that the template matcher may be off by
        :param metric: The metric to use for template matching.
        :param roi: A tuple of indices for the face template in the for ((y1, y2), (x1, x2))
        """
        if not test and self.data['frame_samples'] is None:
            return 'NOT_SET'

        ROI = {
            'left': ((45, 346), (138, 501)),
            'right': ((14, 174), (430, 618)),
            'body': ((141, 272), (90, 339))
        }
        roi = roi or ROI[self.side]
        (y1, y2), (x1, x2) = roi
        refs = self.load_reference_frames(self.side)

        # Method 1: compareHist
        ref_h = cv2.calcHist([refs[0]], [0], None, [256], [0, 256])
        frames = refs if test else self.data['frame_samples']
        hists = [cv2.calcHist([x], [0], None, [256], [0, 256]) for x in frames]
        corr = [cv2.compareHist(test_h, ref_h, cv2.HISTCMP_CORREL) for test_h in hists]
        hist_correlates = all(x > hist_thresh for x in corr)

        # Method 2:
        # TODO Ensure template fully in frame
        template = refs[0][tuple(slice(*r) for r in roi)]
        top_left = []
        for frame in frames:
            res = cv2.matchTemplate(frame, template, metric)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            top_left.append(min_loc if metric < 2 else max_loc)
            # bottom_right = (top_left[0] + w, top_left[1] + h)
        err = (x1, y1) - np.median(np.array(top_left), axis=0)

        if display:
            plt.figure()
            # Plot frame with template overlay
            img = frames[0]
            ax0 = plt.subplot(221)
            ax0.imshow(img, cmap='gray', vmin=0, vmax=255)
            bounds = (x1 - err[0], x2 - err[0], y2 - err[1], y1 - err[1])
            ax0.imshow(template, cmap='gray', alpha=0.5, extent=bounds)
            xy = (x1 - pos_thresh, y1 - pos_thresh)
            ax0.add_patch(Rectangle(xy, x2 - x1 + (pos_thresh * 2), y2 - y1 +(pos_thresh * 2),
                                    fill=True, facecolor='green', lw=0, alpha=0.2))
            xy = (x1 - err[0], y1 - err[1])
            ax0.add_patch(Rectangle(xy, x2-x1, y2-y1,
                                    edgecolor='pink', fill=False, hatch='//', lw=1))
            ax0.set(xlim=(0, img.shape[1]), ylim=(img.shape[0], 0))
            ax0.set_axis_off()
            # Plot the image histograms
            ax1 = plt.subplot(212)
            ax1.plot(ref_h[5:-1], label='reference frame')
            ax1.plot(np.array(hists).mean(axis=0)[5:-1], label='mean frame')
            ax1.set_xlim([0, 256])
            plt.legend()
            # Plot the correlations for each sample frame
            ax2 = plt.subplot(222)
            ax2.plot(corr, label='hist correlation')
            ax2.axhline(hist_thresh, 0, self.n_samples,
                        linestyle=':', color='r', label='pass threshold')
            ax2.set(xlabel='Sample Frame #', ylabel='Hist correlation')
            plt.legend()
            plt.suptitle('Check position')
            plt.show()
        face_aligned = all(np.abs(err) < pos_thresh)

        return 'PASS' if face_aligned and hist_correlates else 'FAIL'

    def check_focus(self, n=20, threshold=100, roi=None, display=False, test=False):
        """Check video is in focus
        Two methods are used here: Looking at the high frequencies with a DFT and
        applying a Laplacian HPF and looking at the variance.

        Note:
            - Both methods are sensitive to noise (Laplacian is 2nd order filter).
            - The thresholds for the fft may need to be different for the left/right vs body as
              the distribution of frequencies in the image is different (e.g. the holder
              comprises mostly very high frequencies).
            - The image may be overall in focus but the places we care about can still be out of
              focus (namely the face).  For this we'll take 2 ROIs - face and wheel/paws.
        """
        if not test and self.data['frame_samples'] is None:
            return 'NOT_SET'

        ROI = {
            'left': (np.s_[:400, :561], np.s_[500:, 100:800]),  # (face, wheel)
            'right': (np.s_[:196, 397:], np.s_[221:, 255:]),
            'body': (np.s_[143:274, 84:433],)  # body holder
        }
        roi = roi or ROI[self.side]

        if test:
            """In test mode load a reference frame and run it through a normalized box filter with
            increasing kernel size.
            """
            idx = (0,)
            ref = self.load_reference_frames(self.side)[idx]
            img = np.empty((n, *ref.shape), dtype=np.uint8)
            kernal_sz = np.unique(np.linspace(0, 15, n, dtype=int))
            for i, k in enumerate(kernal_sz):
                img[i] = ref if k == 0 else cv2.blur(ref, (k, k))
            if display:
                # Plot blurred images
                f, axes = plt.subplots(1, len(img))
                for ax, ig, k in zip(axes, img, kernal_sz):
                    self.imshow(ig, ax=ax, title='Kernal ({0}, {0})'.format(k or 'None'))
                f.suptitle('Reference frame with box filter')
        else:
            # Sub-sample the frame samples
            idx = np.unique(np.linspace(0, len(self.data['frame_samples']) - 1, n, dtype=int))
            img = self.data['frame_samples'][idx]

        # A measure of the sharpness effectively taking the second derivative of the image

        lpc_var = np.empty((min(n, len(img)), len(roi)))
        for i, frame in enumerate(img[::-1]):
            lpc = cv2.Laplacian(frame, cv2.CV_16S, ksize=1)
            lpc_var[i] = [lpc[mask].var() for mask in roi]

        if display:
            # Plot the first sample image
            f = plt.figure()
            gs = f.add_gridspec(len(roi) + 1, 3)
            f.add_subplot(gs[0:len(roi), 0])
            self.imshow(img[0], title=f'Frame #{self.frame_samples_idx[idx[0]]}')
            # Plot the ROIs with and without filter
            lpc = cv2.Laplacian(img[0], cv2.CV_16S, ksize=1)
            abs_lpc = cv2.convertScaleAbs(lpc)
            for i, r in enumerate(roi):
                f.add_subplot(gs[i, 1])
                self.imshow(img[0][r], title=f'ROI #{i + 1}')
                f.add_subplot(gs[i, 2])
                self.imshow(abs_lpc[r], title=f'ROI #{i + 1} - Lapacian filter')
            f.suptitle('Laplacian blur detection')
            # Plot variance over frames
            ax = f.add_subplot(gs[len(roi), :])
            ln = plt.plot(lpc_var)
            [l.set_label(f'ROI #{i + 1}') for i, l in enumerate(ln)]
            ax.axhline(threshold, 0, n, linestyle=':', color='r', label='lower threshold')
            ax.set(xlabel='Frame sample', ylabel='Variance of the Laplacian')
            plt.tight_layout()
            plt.legend()

        # Second test is to highpass with dft
        h, w = img.shape[1:]
        cX, cY = w // 2, h // 2
        sz = 60  # Seems to be the magic number for high pass
        mask = np.ones((h, w, 2), bool)
        mask[cY - sz:cY + sz, cX - sz:cX + sz] = False
        filt_mean = np.empty(len(img))
        for i, frame in enumerate(img[::-1]):
            dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shift = np.fft.fftshift(dft) * mask  # Shift & remove low frequencies
            f_ishift = np.fft.ifftshift(f_shift)  # Shift back
            filt_frame = cv2.idft(f_ishift)  # Reconstruct
            filt_frame = cv2.magnitude(filt_frame[..., 0], filt_frame[..., 1])
            # Re-normalize to 8-bits to make threshold simpler
            img_back = cv2.normalize(filt_frame, None, alpha=0, beta=256,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            filt_mean[i] = np.mean(img_back)
            if i == len(img) - 1 and display:
                # Plot Fourier transforms
                f, axes = plt.subplots(1, 3)
                self.imshow(img[0], ax=axes[0], title='Original frame')
                dft_shift = np.fft.fftshift(dft)
                magnitude = 20 * np.log(cv2.magnitude(dft_shift[..., 0], dft_shift[..., 1]))
                self.imshow(magnitude, ax=axes[1], title='Magnitude spectrum')
                self.imshow(img_back, ax=axes[2], title='Filtered frame')
                f.suptitle('Discrete Fourier Transform')
                plt.show()
        return 'PASS' if np.all(lpc_var > threshold) and np.all(filt_mean > threshold) else 'FAIL'

    @staticmethod
    def load_reference_frames(side):
        refs = [np.load(str(x)) for x in Path(__file__).parent.glob(f'ref*_{side}.npy')]
        refs = np.c_[refs]
        return refs

    @staticmethod
    def imshow(frame, ax=None, title=None, **kwargs):
        """plt.imshow with some convenient defaults for greyscale frames"""
        h = ax or plt.gca()
        defaults = {
            'cmap': kwargs.pop('cmap', 'gray'),
            'vmin': kwargs.pop('vmin', 0),
            'vmax': kwargs.pop('vmax', 255)
        }
        h.imshow(frame, **defaults, **kwargs)
        h.set(title=title)
        h.set_axis_off()
        return ax


def run_all_qc(session, update=False, cameras=('left', 'right', 'body'), stream=True, **kwargs):
    """Run QC for all cameras
    Run the camera QC for left, right and body cameras.
    :param session: A session path or eid.
    :param update: If True, QC fields are updated on Alyx.
    :param cameras: A list of camera names to perform QC on.
    :return: dict of CameraCQ objects
    """
    qc = {}
    for camera in cameras:
        qc[camera] = CameraQC(session, side=camera, stream=stream,
                              one=kwargs.pop('one', None))
        qc[camera].run(update=update, **kwargs)
    return qc

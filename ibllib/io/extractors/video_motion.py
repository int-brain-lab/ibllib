"""
A module for aligning the wheel motion with the rotary encoder.  Currently used by the camera QC
in order to check timestamp alignment.
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from scipy import signal
import cv2
from itertools import cycle
import matplotlib.animation as animation
import logging
from pathlib import Path

from oneibl.one import ONE, OneOffline
import ibllib.io.video as vidio
from brainbox.core import Bunch
import brainbox.video as video
import brainbox.behavior.wheel as wh
from ibllib.misc.exp_ref import eid2ref
import alf.io as alfio


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class MotionAlignment:
    roi = {
        'left': ((800, 1020), (233, 1096)),
        'right': ((426, 510), (104, 545)),
        'body': ((402, 481), (31, 103))
    }

    def __init__(self, eid, one=None, log=logging.getLogger('ibllib'), **kwargs):
        self.one = one or ONE()
        self.eid = eid
        self.session_path = kwargs.pop('session_path', self.one.path_from_eid(eid))
        if self.one and not isinstance(self.one, OneOffline):
            self.ref = eid2ref(self.eid, as_dict=False, one=self.one)
        else:
            self.ref = None
        self.log = log
        self.trials = self.wheel = self.camera_times = None
        raw_cam_path = self.session_path.joinpath('raw_video_data')
        camera_path = list(raw_cam_path.glob('_iblrig_*Camera.raw.*'))
        self.video_paths = {vidio.label_from_path(x): x for x in camera_path}
        self.data = Bunch()
        self.alignment = Bunch()

    def align_all_trials(self, side='all'):
        """Align all wheel motion for all trials"""
        if self.trials is None:
            self.load_data()
        if side == 'all':
            side = self.video_paths.keys()
        if not isinstance(side, str):
            # Try to iterate over sides
            [self.align_all_trials(s) for s in side]
        if side not in self.video_paths:
            raise ValueError(f'{side} camera video file not found')
        # Align each trial sequentially
        for i in np.arange(self.trials['intervals'].shape[0]):
            self.align_motion(i, display=False)

    @staticmethod
    def set_roi(video_path):
        """Manually set the ROIs for a given set of videos
        TODO Improve docstring
        TODO A method for setting ROIs by label
        """
        frame = vidio.get_video_frame(str(video_path), 0)

        def line_select_callback(eclick, erelease):
            """
            Callback for line selection.

            *eclick* and *erelease* are the press and release events.
            """
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
            return np.array([[x1, x2], [y1, y2]])

        plt.imshow(frame)
        roi = RectangleSelector(plt.gca(), line_select_callback,
                                drawtype='box', useblit=True,
                                button=[1, 3],  # don't use middle button
                                minspanx=5, minspany=5,
                                spancoords='pixels',
                                interactive=True)
        plt.show()
        ((x1, x2, *_), (y1, *_, y2)) = roi.corners
        col = np.arange(round(x1), round(x2), dtype=int)
        row = np.arange(round(y1), round(y2), dtype=int)
        return col, row

    def load_data(self, download=False):
        """
        Load wheel, trial and camera timestamp data
        :return: wheel, trials
        """
        if download:
            self.data.wheel = self.one.load_object(self.eid, 'wheel')
            self.data.trials = self.one.load_object(self.eid, 'trials')
            cam = self.one.load(self.eid, ['camera.times'], dclass_output=True)
            self.data.camera_times = {vidio.label_from_path(url): ts
                                      for ts, url in zip(cam.data, cam.url)}
        else:
            alf_path = self.session_path / 'alf'
            self.data.wheel = alfio.load_object(alf_path, 'wheel')
            self.data.trials = alfio.load_object(alf_path, 'trials')
            self.data.camera_times = {vidio.label_from_path(x): alfio.load_file_content(x)
                                      for x in alf_path.glob('*Camera.times*')}
        assert all(x is not None for x in self.data.values())

    def _set_eid_or_path(self, session_path_or_eid):
        """Parse a given eID or session path
        If a session UUID is given, resolves and stores the local path and vice versa
        :param session_path_or_eid: A session eid or path
        :return:
        """
        self.eid = None
        if alfio.is_uuid_string(str(session_path_or_eid)):
            self.eid = session_path_or_eid
            # Try to set session_path if data is found locally
            self.session_path = self.one.path_from_eid(self.eid)
        elif alfio.is_session_path(session_path_or_eid):
            self.session_path = Path(session_path_or_eid)
            if self.one is not None:
                self.eid = self.one.eid_from_path(self.session_path)
                if not self.eid:
                    self.log.warning('Failed to determine eID from session path')
        else:
            self.log.error('Cannot run alignment: an experiment uuid or session path is required')
            raise ValueError("'session' must be a valid session path or uuid")

    def align_motion(self, period=(-np.inf, np.inf), side='left', sd_thresh=10, display=False):
        # Get data samples within period
        wheel = self.data['wheel']
        self.alignment.label = side
        self.alignment.to_mask = lambda ts: np.logical_and(ts >= period[0], ts <= period[1])
        camera_times = self.data['camera_times'][side]
        cam_mask = self.alignment.to_mask(camera_times)
        frame_numbers, = np.where(cam_mask)

        if frame_numbers.size == 0:
            raise ValueError('No frames during given period')

        # Motion Energy
        camera_path = self.video_paths[side]
        roi = (*[slice(*r) for r in self.roi[side]], 0)
        try:
            # TODO Add function arg to make grayscale
            self.alignment.frames = \
                vidio.get_video_frames_preload(camera_path, frame_numbers, mask=roi)
        except AssertionError:
            self.log.error('Failed to open video')
            return None, None, None
        self.alignment.df, stDev = video.motion_energy(self.alignment.frames, 2)
        self.alignment.period = period  # For plotting

        # Calculate rotary encoder velocity trace
        x = camera_times[cam_mask]
        Fs = 1000
        pos, t = wh.interpolate_position(wheel.timestamps, wheel.position, freq=Fs)
        v, _ = wh.velocity_smoothed(pos, Fs)
        interp_mask = self.alignment.to_mask(t)
        # Convert to normalized speed
        xs = np.unique([find_nearest(t[interp_mask], ts) for ts in x])
        vs = np.abs(v[interp_mask][xs])
        vs = (vs - np.min(vs)) / (np.max(vs) - np.min(vs))

        # FIXME This can be used as a goodness of fit measure
        USE_CV2 = False
        if USE_CV2:
            # convert from numpy format to openCV format
            dfCV = np.float32(self.alignment.df.reshape((-1, 1)))
            reCV = np.float32(vs.reshape((-1, 1)))

            # perform cross correlation
            resultCv = cv2.matchTemplate(dfCV, reCV, cv2.TM_CCORR_NORMED)

            # convert result back to numpy array
            xcorr = np.asarray(resultCv)
        else:
            xcorr = signal.correlate(self.alignment.df, vs)

        # Cross correlate wheel speed trace with the motion energy
        CORRECTION = 2
        self.alignment.c = max(xcorr)
        self.alignment.xcorr = np.argmax(xcorr)
        self.alignment.dt_i = self.alignment.xcorr - xs.size + CORRECTION
        self.log.info(f'{side} camera, adjusted by {self.alignment.dt_i} frames')

        if display:
            # Plot the motion energy
            fig, ax = plt.subplots(2, 1, sharex='all')
            y = np.pad(self.alignment.df, 1, 'edge')
            ax[0].plot(x, y, '-x', label='wheel motion energy')
            thresh = stDev > sd_thresh
            ax[0].vlines(x[np.array(np.pad(thresh, 1, 'constant', constant_values=False))], 0, 1,
                         linewidth=0.5, linestyle=':', label=f'>{sd_thresh} s.d. diff')
            ax[1].plot(t[interp_mask], np.abs(v[interp_mask]))

            # Plot other stuff
            dt = np.diff(camera_times[[0, np.abs(self.alignment.dt_i)]])
            fps = 1 / np.diff(camera_times).mean()
            ax[0].plot(t[interp_mask][xs] - dt, vs, 'r-x', label='velocity (shifted)')
            ax[0].set_title('normalized motion energy, %s camera, %.0f fps' % (side, fps))
            ax[0].set_ylabel('rate of change (a.u.)')
            ax[0].legend()
            ax[1].set_ylabel('wheel speed (rad / s)')
            ax[1].set_xlabel('Time (s)')

            title = f'{self.ref}, from {period[0]:.1f}s - {period[1]:.1f}s'
            fig.suptitle(title, fontsize=16)
            fig.set_size_inches(19.2, 9.89)

        return self.alignment.dt_i, self.alignment.c, self.alignment.df

    def plot_alignment(self, energy=True, save=False):
        if not self.alignment:
            self.log.error('No alignment data, run `align_motion` first')
            return
        # Change backend based on save flag
        backend = matplotlib.get_backend().lower()
        if (save and backend != 'agg') or (not save and backend == 'agg'):
            new_backend = 'Agg' if save else 'Qt5Agg'
            self.log.warning('Switching backend from %s to %s', backend, new_backend)
            matplotlib.use(new_backend)
        from matplotlib import pyplot as plt

        # Main animated plots
        fig, axes = plt.subplots(nrows=2)
        title = f'{self.ref}'  # ', from {period[0]:.1f}s - {period[1]:.1f}s'
        fig.suptitle(title, fontsize=16)

        wheel = self.data['wheel']
        wheel_mask = self.alignment['to_mask'](wheel.timestamps)
        ts = self.data['camera_times'][self.alignment['label']]
        frame_numbers, = np.where(self.alignment['to_mask'](ts))
        if energy:
            self.alignment['frames'] = video.frame_diffs(self.alignment['frames'], 2)
            frame_numbers = frame_numbers[1:-1]
        data = {'frame_ids': frame_numbers}

        def init_plot():
            """
            Plot the wheel data for the current trial
            :return: None
            """
            data['im'] = axes[0].imshow(self.alignment['frames'][0])
            axes[0].axis('off')
            axes[0].set_title(f'adjusted by {self.alignment["dt_i"]} frames')

            # Plot the wheel position
            ax = axes[1]
            ax.clear()
            ax.plot(wheel.timestamps[wheel_mask], wheel.position[wheel_mask], '-x')

            ts_0 = frame_numbers[0]
            data['idx_0'] = ts_0 - self.alignment['dt_i']
            ts_0 = ts[ts_0 + self.alignment['dt_i']]
            data['ln'] = ax.axvline(x=ts_0, color='k')
            ax.set_xlim([ts_0 - (3 / 2), ts_0 + (3 / 2)])
            data['frame_num'] = 0
            mkr = find_nearest(wheel.timestamps[wheel_mask], ts_0)

            data['marker'], = ax.plot(
                wheel.timestamps[wheel_mask][mkr],
                wheel.position[wheel_mask][mkr], 'r-x')
            ax.set_ylabel('Wheel position (rad))')
            ax.set_xlabel('Time (s))')
            return

        def animate(i):
            """
            Callback for figure animation.  Sets image data for current frame and moves pointer
            along axis
            :param i: unused; the current time step of the calling method
            :return: None
            """
            if i < 0:
                data['frame_num'] -= 1
                if data['frame_num'] < 0:
                    data['frame_num'] = len(self.alignment['frames']) - 1
            else:
                data['frame_num'] += 1
                if data['frame_num'] >= len(self.alignment['frames']):
                    data['frame_num'] = 0
            i = data['frame_num']  # NB: This is index for current trial's frame list

            frame = self.alignment['frames'][i]
            t_x = ts[data['idx_0'] + i]
            data['ln'].set_xdata([t_x, t_x])
            axes[1].set_xlim([t_x - (3 / 2), t_x + (3 / 2)])
            data['im'].set_data(frame)

            mkr = find_nearest(wheel.timestamps[wheel_mask], t_x)
            data['marker'].set_data(
                wheel.timestamps[wheel_mask][mkr],
                wheel.position[wheel_mask][mkr]
            )

            return data['im'], data['ln'], data['marker']

        anim = animation.FuncAnimation(fig, animate, init_func=init_plot,
                                       frames=(range(len(self.alignment.df))
                                               if save
                                               else cycle(range(60))),
                                       interval=20, blit=False,
                                       repeat=not save, cache_frame_data=False)
        anim.running = False

        def process_key(event):
            """
            Callback for key presses.
            :param event: a figure key_press_event
            :return: None
            """
            if event.key.isspace():
                if anim.running:
                    anim.event_source.stop()
                else:
                    anim.event_source.start()
                anim.running = ~anim.running
            elif event.key == 'right':
                if anim.running:
                    anim.event_source.stop()
                    anim.running = False
                animate(1)
                fig.canvas.draw()
            elif event.key == 'left':
                if anim.running:
                    anim.event_source.stop()
                    anim.running = False
                animate(-1)
                fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', process_key)

        # init_plot()
        # while True:
        #     animate(0)
        if save:
            filename = '%s_%c.mp4' % (self.ref, self.alignment['label'][0])
            if isinstance(save, (str, Path)):
                filename = Path(save).joinpath(filename)
            self.log.info(f'Saving to {filename}')
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=24, metadata=dict(artist='Miles Wells'), bitrate=1800)
            anim.save(str(filename), writer=writer)
        else:
            plt.show()

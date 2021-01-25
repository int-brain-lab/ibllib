"""
TODO Move into ibllib
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
import re

from oneibl.one import ONE
from brainbox.io.video import get_video_frames_preload, get_video_frame
import brainbox.video.video as video
import brainbox.behavior.wheel as wh
from ibllib.misc.exp_ref import eid2ref
import alf.io as alfio


class MotionAlignment:
    # TODO Make tuple?
    roi = {
        'left': ((800, 1020), (233, 1096)),
        'right': ((426, 510), (104, 545)),
        'body': ((402, 481), (31, 103))
    }
    cam_regex = re.compile(r'(?<=_)[a-z]+(?=Camera.)')

    def __init__(self, eid, one=None, log=logging.getLogger('ibllib')):
        self.one = one or ONE()
        self.eid = eid
        self.session_path = self.one.path_from_eid(eid)
        self.log = log
        self.trials = self.wheel = self.camera_times = None
        raw_cam_path = self.session_path.joinpath('raw_video_data')
        camera_path = list(raw_cam_path.glob('_iblrig_*Camera.raw.*'))
        self.video_paths = {self.cam_regex.search(str(x)).group(): x for x in camera_path}

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

    def set_roi(video_path):
        """Manually set the ROIs for a given set of videos
        TODO Improve docstring
        TODO A method for setting ROIs by side
        """
        frame = get_video_frame(str(video_path), 0)

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
        TODO Assert data present
        TODO load collection
        """
        self.wheel = self.one.load_object(self.eid, 'wheel')
        self.trials = self.one.load_object(self.eid, 'trials')
        cam = self.one.load(eid, ['camera.times'], dclass_output=True)
        self.camera_times = {self.cam_regex.search(url).group(): ts
                             for ts, url in zip(cam.data, cam.url)}
        # cam_ts, = [ts for ts, url in zip(cam_ts.data, cam_ts.url) if side in url]

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
            self.log.error('Cannot run QC: an experiment uuid or session path is required')
            raise ValueError("'session' must be a valid session path or uuid")

    def align_motion(self, trial=None, side='left', save=True, display=False, sd_thresh=10):
        ref = eid2ref(self.eid, as_dict=False, one=self.one)
        backend = matplotlib.get_backend()
        if (save and backend != 'Agg') or (not save and backend == 'Agg'):
            new_backend = 'Agg' if save else 'Qt5Agg'
            self.log.warning('Switching backend from %s to %s', backend, new_backend)
            matplotlib.use(new_backend)
        from matplotlib import pyplot as plt

        if trial is None:
            trial = np.random.randint(len(self.trials.choice))

        def mask(ts):
            intervals = self.trials.intervals
            return np.logical_and(ts >= intervals[trial, 0], ts <= intervals[trial + 1, 1])

        camera_times = self.camera_times[side]
        cam_mask = mask(camera_times)
        wheel_mask = mask(self.wheel.timestamps)
        frame_numbers, = np.where(cam_mask)
        # frames, fps, count = get_video_frames_preload(camera_path, frame_numbers)
        # col = np.arange(233, 1096, dtype=int)  # Predetermined ROI
        # row = np.arange(800, 1020, dtype=int)

        # Motion Energy
        camera_path = self.video_paths[side]
        print(str(camera_path))
        # cap = cv2.VideoCapture(str(camera_path))
        # assert cap.isOpened()
        fame_ids, = np.where(cam_mask)

        roi = (*[slice(*r) for r in self.roi[side]], 0)
        frames = get_video_frames_preload(camera_path, frame_numbers, mask=roi, as_list=True)
        df, stDev = video.motion_energy(frames, 2)
        thresh = stDev > sd_thresh

        x = camera_times[cam_mask]
        Fs = 1000
        # x = x[1::2]
        pos, t = wh.interpolate_position(self.wheel.timestamps, self.wheel.position, freq=Fs)
        v, _ = wh.velocity_smoothed(pos, Fs)
        interp_mask = mask(t)

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        xs = np.unique([find_nearest(t[interp_mask], ts) for ts in x])
        vs = np.abs(v[interp_mask][xs])
        vs = (vs - np.min(vs)) / (np.max(vs) - np.min(vs))

        if display:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(x, df, '-x', label='wheel motion energy')
            ax[0].vlines(x[np.array(thresh)], 0, 1,
                         linewidth=0.5, linestyle=':', label='>%i s.d. diff' % sd_thresh)
            ax[1].plot(t[interp_mask], np.abs(v[interp_mask]))

        # FIXME This can be used as a goodness of fit measure
        USE_CV2 = False
        if USE_CV2:
            # convert from numpy format to openCV format
            dfCV = np.float32(df.reshape((-1, 1)))
            reCV = np.float32(vs.reshape((-1, 1)))

            # perform cross correlation
            resultCv = cv2.matchTemplate(dfCV, reCV, cv2.TM_CCORR_NORMED)

            # convert result back to numpy array
            xcorr = np.asarray(resultCv)
        else:
            xcorr = signal.correlate(df, vs)

        print(np.all(
            np.array([0.04682708, 0.08597485, 0.12770387, 0.16950493, 0.21221944,
                      0.25907589, 0.30770999, 0.35522964, 0.40655199, 0.46094253]) == xcorr))
        c = max(xcorr)
        xcorr = np.argmax(xcorr)
        dt_i = xcorr - xs.size
        self.log.info(f'{side} camera, adjusted by {dt_i} frames')

        if display:
            dt = np.diff(camera_times[[0, np.abs(dt_i)]])
            ax[0].plot(t[interp_mask][xs] - dt, vs, 'r-x', label='velocity (shifted)')
            ax[0].set_title('normalized motion energy, %s camera, %.0f fps' % (side, fps))
            ax[0].set_ylabel('rate of change (a.u.)')
            ax[0].legend()
            ax[1].set_ylabel('Abs wheel velocity (rad / s)')
            ax[1].set_xlabel('Time (s)')

            fig.suptitle('%s, trial %i' % (ref, trial), fontsize=16)
            fig.set_size_inches(19.2, 9.89)
            fig.savefig('%s_%i_%c.png' % (ref, trial, side[0]), dpi=100)

            ###
            fig, axes = plt.subplots(nrows=2)
            fig.suptitle('%s, trial %i' % (ref, trial), fontsize=16)
            data = {}

        def init_plot():
            """
            Plot the wheel data for the current trial
            :return: None
            """
            data['im'] = axes[0].imshow(df[0])
            axes[0].axis('off')
            axes[0].set_title('%s camera, adjusted by %d frames' % (side, dt_i))

            # Plot the wheel position
            ax = axes[1]
            ax.clear()
            ax.plot(self.wheel.timestamps[wheel_mask], self.wheel.position[wheel_mask], '-x')

            ts_0, = np.where(cam_mask)
            data['idx_0'] = ts_0[0] - dt_i
            ts_0 = camera_times[ts_0[0] + dt_i]
            data['ln'] = ax.axvline(x=ts_0, color='k')
            ax.set_xlim([ts_0 - (3 / 2), ts_0 + (3 / 2)])
            data['frame_num'] = 0
            mkr = find_nearest(self.wheel.timestamps[wheel_mask], ts_0)

            data['marker'], = ax.plot(
                self.wheel.timestamps[wheel_mask][mkr],
                self.wheel.position[wheel_mask][mkr], 'r-x')
            ax.set_ylabel('Wheel position (rad))')
            ax.set_xlabel('Time (s))')
            return

        def animate(i):
            """
            Callback for figure animation.  Sets image data for current frame and moves pointer
            along axis
            :param i: unused; the current timestep of the calling method
            :return: None
            """
            if i < 0:
                data['frame_num'] -= 1
                if data['frame_num'] < 0:
                    data['frame_num'] = len(df) - 1
            else:
                data['frame_num'] += 1
                if data['frame_num'] >= len(df):
                    data['frame_num'] = 0
            i = data['frame_num']  # NB: This is index for current trial's frame list

            frame = df[i]
            t_x = camera_times[data['idx_0'] + i]
            data['ln'].set_xdata([t_x, t_x])
            axes[1].set_xlim([t_x - (3 / 2), t_x + (3 / 2)])
            data['im'].set_data(frame)

            mkr = find_nearest(self.wheel.timestamps[wheel_mask], t_x)
            data['marker'].set_data(
                self.wheel.timestamps[wheel_mask][mkr],
                self.wheel.position[wheel_mask][mkr]
            )

            return data['im'], data['ln'], data['marker']
        if display:
            anim = animation.FuncAnimation(fig, animate, init_func=init_plot,
                                           frames=range(len(df)) if save else cycle(range(60)),
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
        if display:
            fig.canvas.mpl_connect('key_press_event', process_key)

        # init_plot()
        # while True:
        #     animate(0)
        if save:
            filename = '%s_%i_%c.mp4' % (ref, trial, side[0])
            self.log.info('Saving to ' + filename)
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=24, metadata=dict(artist='Miles Wells'), bitrate=1800)
            anim.save(filename, writer=writer)
        elif display:
            plt.show()

        return dt_i, c, df


def motion_energy(video_path, n=5):
    cap = cv2.VideoCapture(video_path)  # TODO Add https support
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = 100
    prev_frame = np.array([])
    nrg = np.empty(n_frames - 1)
    for idx in np.array_split(np.arange(n_frames), np.ceil(n_frames / n)):
        frames = get_video_frames_preload(video_path, idx, mask=np.s_[:, :, 0])
        frames = np.r_[prev_frame[np.newaxis, :, :], frames] if prev_frame.size else frames
        nrg[idx[:len(frames)-1]], _ = video.motion_energy(frames, diff=1, normalize=False)
        prev_frame = frames[-1]
    return nrg

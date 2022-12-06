"""DLC QC
This module runs a list of quality control metrics on the dlc traces.

Example - Run DLC QC
    qc = DlcQC(eid, 'left', download_data=True)
    qc.run()
Question:
    We're not extracting the audio based on TTL length.  Is this a problem?
"""
import logging
import warnings
from inspect import getmembers, isfunction

import numpy as np

from ibllib.qc import base
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from one.alf.spec import is_session_path
from iblutil.util import Bunch
from brainbox.behavior.dlc import insert_idx, SAMPLING

_log = logging.getLogger(__name__)


class DlcQC(base.QC):
    """A class for computing camera QC metrics"""

    bbox = {
        'body': {
            'xrange': range(201, 500),
            'yrange': range(81, 330)
        },
        'left': {
            'xrange': range(301, 700),
            'yrange': range(181, 470)
        },
        'right': {
            'xrange': range(301, 600),
            'yrange': range(110, 275)
        },
    }

    dstypes = {
        'left': [
            '_ibl_leftCamera.dlc.*', '_ibl_leftCamera.times.*', '_ibl_leftCamera.features.*', '_ibl_trials.table.*'
        ],
        'right': [
            '_ibl_rightCamera.dlc.*', '_ibl_rightCamera.times.*', '_ibl_rightCamera.features.*', '_ibl_trials.table.*'
        ],
        'body': [
            '_ibl_bodyCamera.dlc.*', '_ibl_bodyCamera.times.*'
        ],
    }

    def __init__(self, session_path_or_eid, side, ignore_checks=['check_pupil_diameter_snr'], **kwargs):
        """
        :param session_path_or_eid: A session eid or path
        :param side: The camera to run QC on
        :param ignore_checks: Checks that won't count towards aggregate QC, but will be run and added to extended QC
        :param log: A logging.Logger instance, if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        """
        # Make sure the type of camera is chosen
        self.side = side
        # When an eid is provided, we will download the required data by default (if necessary)
        download_data = not is_session_path(session_path_or_eid)
        self.download_data = kwargs.pop('download_data', download_data)
        super().__init__(session_path_or_eid, **kwargs)
        self.data = Bunch()

        # checks to be added to extended QC but not taken into account for aggregate QC
        self.ignore_checks = ignore_checks
        # QC outcomes map
        self.metrics = None

    def load_data(self, download_data: bool = None) -> None:
        """Extract the data from data files
        Extracts all the required task data from the data files.

        Data keys:
            - camera_times (float array): camera frame timestamps extracted from frame headers
            - dlc_coords (dict): keys are the points traced by dlc, items are x-y coordinates of
                                 these points over time, those with likelihood <0.9 set to NaN

        :param download_data: if True, any missing raw data is downloaded via ONE.
        """
        if download_data is not None:
            self.download_data = download_data
        if self.one and not self.one.offline:
            self._ensure_required_data()

        alf_path = self.session_path / 'alf'

        # Load times
        self.data['camera_times'] = alfio.load_object(alf_path, f'{self.side}Camera')['times']
        # Load dlc traces
        dlc_df = alfio.load_object(alf_path, f'{self.side}Camera', namespace='ibl')['dlc']
        targets = np.unique(['_'.join(col.split('_')[:-1]) for col in dlc_df.columns])
        # Set values to nan if likelihood is too low
        dlc_coords = {}
        for t in targets:
            idx = dlc_df.loc[dlc_df[f'{t}_likelihood'] < 0.9].index
            dlc_df.loc[idx, [f'{t}_x', f'{t}_y']] = np.nan
            dlc_coords[t] = np.array((dlc_df[f'{t}_x'], dlc_df[f'{t}_y']))
        self.data['dlc_coords'] = dlc_coords

        # load stim on times
        self.data['stimOn_times'] = alfio.load_object(alf_path, 'trials', namespace='ibl')['stimOn_times']

        # load pupil diameters
        if self.side in ['left', 'right']:
            features = alfio.load_object(alf_path, f'{self.side}Camera', namespace='ibl')['features']
            self.data['pupilDiameter_raw'] = features['pupilDiameter_raw']
            self.data['pupilDiameter_smooth'] = features['pupilDiameter_smooth']

    def _ensure_required_data(self):
        """
        Ensures the datasets required for QC are local.  If the download_data attribute is True,
        any missing data are downloaded.  If all the data are not present locally at the end of
        it an exception is raised.
        :return:
        """
        for ds in self.dstypes[self.side]:
            # Check if data available locally
            if not next(self.session_path.rglob(ds), None):
                # If download is allowed, try to download
                if self.download_data is True:
                    assert self.one is not None, 'ONE required to download data'
                    try:
                        self.one.load_dataset(self.eid, ds, download_only=True)
                    except ALFObjectNotFound:
                        raise AssertionError(f'Dataset {ds} not found locally and failed to download')
                else:
                    raise AssertionError(f'Dataset {ds} not found locally and download_data is False')

    def _compute_trial_window_idxs(self):
        """Find start and end times of a window around stimulus onsets in video indices."""
        window_lag = -0.5
        window_len = 2.0
        start_window = self.data['stimOn_times'] + window_lag
        start_idx = insert_idx(self.data['camera_times'], start_window)
        end_idx = np.array(start_idx + int(window_len * SAMPLING[self.side]), dtype='int64')
        return start_idx, end_idx

    def _compute_proportion_nan_in_trial_window(self, body_part):
        """Find proportion of NaN frames for a given body part in trial-based windows."""
        # find timepoints in windows around stimulus onset
        start_idx, end_idx = self._compute_trial_window_idxs()
        # compute fraction of points in windows that are NaN
        dlc_coords = np.concatenate([self.data['dlc_coords'][body_part][0, start_idx[i]:end_idx[i]]
                                     for i in range(len(start_idx))])
        prop_nan = np.sum(np.isnan(dlc_coords)) / dlc_coords.shape[0]
        return prop_nan

    def run(self, update: bool = False, **kwargs) -> (str, dict):
        """
        Run DLC QC checks and return outcome
        :param update: if True, updates the session QC fields on Alyx
        :param download_data: if True, downloads any missing data if required
        :returns: overall outcome as a str, a dict of checks and their outcomes
        """
        _log.info(f'Running DLC QC for {self.side} camera, session {self.eid}')
        namespace = f'dlc{self.side.capitalize()}'
        if all(x is None for x in self.data.values()):
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(DlcQC, is_metric)
        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}

        ignore_metrics = [f'_{namespace}_' + i[6:] for i in self.ignore_checks]
        metrics_to_aggregate = {k: v for k, v in self.metrics.items() if k not in ignore_metrics}
        outcome = self.overall_outcome(metrics_to_aggregate.values())

        if update:
            extended = {
                k: 'NOT_SET' if v is None else v
                for k, v in self.metrics.items()
            }
            self.update_extended_qc(extended)
            self.update(outcome, namespace)
        return outcome, self.metrics

    def check_time_trace_length_match(self):
        '''
        Check that the length of the DLC traces is the same length as the video.
        '''
        dlc_coords = self.data['dlc_coords']
        times = self.data['camera_times']
        for target in dlc_coords.keys():
            if times.shape[0] != dlc_coords[target].shape[1]:
                _log.warning(f'{self.side}Camera length of camera.times does not match '
                             f'length of camera.dlc {target}')
                return 'FAIL'
        return 'PASS'

    def check_trace_all_nan(self):
        '''
        Check that none of the dlc traces, except for the 'tube' traces, are all NaN.
        '''
        dlc_coords = self.data['dlc_coords']
        for target in dlc_coords.keys():
            if 'tube' not in target:
                if all(np.isnan(dlc_coords[target][0])) or all(np.isnan(dlc_coords[target][1])):
                    _log.warning(f'{self.side}Camera dlc trace {target} all NaN')
                    return 'FAIL'
        return 'PASS'

    def check_mean_in_bbox(self):
        '''
        Empirical bounding boxes around average dlc points, averaged across time and points;
        sessions with points out of this box were often faulty in terms of raw videos
        '''

        dlc_coords = self.data['dlc_coords']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_mean = np.nanmean([np.nanmean(dlc_coords[k][0]) for k in dlc_coords.keys()])
            y_mean = np.nanmean([np.nanmean(dlc_coords[k][1]) for k in dlc_coords.keys()])

        xrange = self.bbox[self.side]['xrange']
        yrange = self.bbox[self.side]['yrange']
        if int(x_mean) not in xrange or int(y_mean) not in yrange:
            return 'FAIL'
        else:
            return 'PASS'

    def check_pupil_blocked(self):
        '''
        Check if pupil diameter is nan for more than 60 % of the frames
        (might be blocked by a whisker)
        Check if standard deviation is above a threshold, found for bad sessions
        '''

        if self.side == 'body':
            return 'NOT_SET'

        if np.mean(np.isnan(self.data['pupilDiameter_raw'])) > 0.9:
            _log.warning(f'{self.eid}, {self.side}Camera, pupil diameter too often NaN')
            return 'FAIL'

        thr = 5 if self.side == 'right' else 10
        if np.nanstd(self.data['pupilDiameter_raw']) > thr:
            _log.warning(f'{self.eid}, {self.side}Camera, pupil diameter too unstable')
            return 'FAIL'

        return 'PASS'

    def check_lick_detection(self):
        '''
        Check if both of the two tongue edge points are less than 10 % NaN, indicating that
        wrong points are detected (spout edge, mouth edge)
        '''

        if self.side == 'body':
            return 'NOT_SET'
        dlc_coords = self.data['dlc_coords']
        nan_l = np.mean(np.isnan(dlc_coords['tongue_end_l'][0]))
        nan_r = np.mean(np.isnan(dlc_coords['tongue_end_r'][0]))
        if (nan_l < 0.1) and (nan_r < 0.1):
            return 'FAIL'
        return 'PASS'

    def check_pupil_diameter_snr(self):
        if self.side == 'body':
            return 'NOT_SET'
        thresh = 5 if self.side == 'right' else 10
        if 'pupilDiameter_raw' not in self.data.keys() or 'pupilDiameter_smooth' not in self.data.keys():
            return 'NOT_SET'
        # compute signal to noise ratio between raw and smooth dia
        good_idxs = np.where(~np.isnan(self.data['pupilDiameter_smooth']) & ~np.isnan(self.data['pupilDiameter_raw']))[0]
        snr = (np.var(self.data['pupilDiameter_smooth'][good_idxs]) /
               (np.var(self.data['pupilDiameter_smooth'][good_idxs] - self.data['pupilDiameter_raw'][good_idxs])))
        if snr < thresh:
            return 'FAIL', float(round(snr, 3))
        return 'PASS', float(round(snr, 3))

    def check_paw_close_nan(self):
        if self.side == 'body':
            return 'NOT_SET'
        thresh_fail = 0.20  # prop of NaNs above this threshold means the check fails
        thresh_warning = 0.10  # prop of NaNs above this threshold means the check is a warning
        # compute fraction of points in windows that are NaN
        prop_nan = self._compute_proportion_nan_in_trial_window(body_part='paw_r')
        if prop_nan > thresh_fail:
            return 'FAIL'
        elif prop_nan > thresh_warning:
            return 'WARNING'
        else:
            return 'PASS'

    def check_paw_far_nan(self):
        if self.side == 'body':
            return 'NOT_SET'
        thresh_fail = 0.20  # prop of NaNs above this threshold means the check fails
        thresh_warning = 0.10  # prop of NaNs above this threshold means the check is a warning
        # compute fraction of points in windows that are NaN
        prop_nan = self._compute_proportion_nan_in_trial_window(body_part='paw_l')
        if prop_nan > thresh_fail:
            return 'FAIL'
        elif prop_nan > thresh_warning:
            return 'WARNING'
        else:
            return 'PASS'


def run_all_qc(session, cameras=('left', 'right', 'body'), one=None, **kwargs):
    """Run DLC QC for all cameras
    Run the DLC QC for left, right and body cameras.
    :param session: A session path or eid.
    :param update: If True, QC fields are updated on Alyx.
    :param cameras: A list of camera names to perform QC on.
    :return: dict of DlcQC objects
    """
    qc = {}
    run_args = {k: kwargs.pop(k) for k in ('download_data', 'update') if k in kwargs.keys()}
    for camera in cameras:
        qc[camera] = DlcQC(session, side=camera, one=one, **kwargs)
        qc[camera].run(**run_args)
    return qc

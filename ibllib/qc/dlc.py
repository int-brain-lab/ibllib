"""DLC QC
This module runs a list of quality control metrics on the dlc traces.

Example - Run DLC QC
    qc = DlcQC(eid, download_data=True)
    qc.run()
Question:
    We're not extracting the audio based on TTL length.  Is this a problem?
"""
from ibllib.qc import base
from brainbox.core import Bunch
import alf.io
import numpy as np
import logging
from inspect import getmembers, isfunction
from scipy.stats import zscore

_log = logging.getLogger('ibllib')


class DlcQC(base.QC):
    """A class for computing camera QC metrics"""
    # TODO You can add some constants here, which can be accessed without
    # instantiating the class

    # Datasets to check for (or download if download_data == True)
    dstypes = ['camera.dlc', 'camera.times']

    def __init__(self, session_path_or_eid, video_type, **kwargs):
        """
        :param session_path_or_eid: A session eid or path
        :param log: A logging.Logger instance,
                    if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        TODO Add optional kwargs specific to DLC
        """
        # When an eid is provided, we will download the required data by
        # default (if necessary)
        download_data = not alf.io.is_session_path(session_path_or_eid)
        self.download_data = kwargs.pop('download_data', download_data)
        super().__init__(session_path_or_eid, **kwargs)
        self.data = Bunch()
        # QC outcomes map
        self.metrics = None
        self.outcome = 'NOT_SET'
        self.video_type = video_type

    def load_data(self, download_data: bool = None) -> None:
        """Extract the data from raw data files
        Extracts all the required task data from the raw data files.
​
        :param download_data: if True, any missing raw data
                              is downloaded via ONE.
        """
        if download_data is not None:
            self.download_data = download_data
        if self.one:
            self._ensure_required_data()
        _log.info('Gathering data for QC')

        alf_path = self.session_path / 'alf'

        cam0 = alf.io.load_object(
            alf_path,
            '%sCamera' %
            self.video_type,
            namespace='ibl')

        cam = cam0['dlc']
        points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

        # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
        XYs = {}
        for point in points:
            x = np.ma.masked_where(
                cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
            x = x.filled(np.nan)
            y = np.ma.masked_where(
                cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
            y = y.filled(np.nan)
            XYs[point] = np.array([x, y])

        self.data['camera_times'], self.data['XYs'] = cam0['times'], XYs

    def check_time_trace_length_match(self):
        XYs = self.data['XYs']
        Times = self.data['camera_times']
        t = 0
        for point in XYs:
            for coordinate in [0, 1]:
                if len(Times) < len(XYs[point][coordinate]):
                    _log.error(f'{self.video_type} dlc time<{point} \
                                  lengths mismatch')
                    t += 1

        if t == 0:
            return 'PASS'
        else:
            return 'FAIL'

    def check_if_points_all_nan(self):
        XYs = self.data['XYs']
        t = 0
        for point in XYs:
            if 'tube' in point:
                continue
            for coordinate in [0, 1]:
                if all(np.isnan(XYs[point][coordinate])):
                    t += 1
                    _log.error(f'{self.video_type} {point} all coord {coordinate} nan')

        if t == 0:
            return 'PASS'
        else:
            return 'FAIL'

    def get_mean_positions(self):
        XYs = self.data['XYs']
        mloc = {}  # mean locations
        for point in XYs:
            mloc[point] = [np.nanmean(XYs[point][0]),
                           np.nanmean(XYs[point][1])]
        return mloc

    def check_if_mean_in_box(self):
        '''
        Empirical bounding boxes around average
        dlc points, averaged across time and points;
        sessions with points out of this box were
        often faulty in terms of raw videos
        '''

        mloc = self.get_mean_positions()
        x = np.nanmean([i[0] for i in mloc.values()])
        y = np.nanmean([i[1] for i in mloc.values()])

        if self.video_type == 'left':
            a = any(~np.array([300 < x < 700, 180 < y < 470]))
        if self.video_type == 'right':
            a = any(~np.array([300 < x < 600, 110 < y < 275]))
        if self.video_type == 'body':
            a = any(~np.array([200 < x < 500, 80 < y < 330]))

        if a:
            _log.error(f'{self.video_type} average position is out of box')
            return 'FAIL'
        else:
            return 'PASS'


    def check_whisker_pupil_block(self):
        '''
        count fraction of points that are 2 std 
        away from mean; numbers sensible for 
        pupil points only; 
        '''
        if self.video_type=='body':
            return 'PASS'    
        
        XYs = self.data['XYs']
        
        for point in XYs:
            if 'pupil' not in point:
                continue
            x = zscore(XYs[point][0],nan_policy = 'omit')   
            y = zscore(XYs[point][1],nan_policy = 'omit')
            ps = (x**2 + y**2)**0.5
            outs = len(np.where(ps > 2)[0])
            total = np.sum(~np.isnan(XYs[point][0]))
                 
            if outs/total > 0.12:
                _log.error(f'{self.eid}, {self.video_type}, {point} \
                              too often far from mean')
                return 'FAIL'   
                
        return 'PASS'                

     
    def check_lick_detection(self):
        '''
        check if both of the two tongue edge points are 
        less than 10 % nan, indicating that 
        wrong points are detected (spout edge, mouth edge)
        '''
        
        if self.video_type=='body':
            return 'PASS'      
               
        XYs = self.data['XYs']
        points = ['tongue_end_r','tongue_end_l'] 
        ms = []
        for point in points:    
            ms.append(np.mean(np.isnan(XYs[point][0])))    
        if (ms[0] < 0.1) and (ms[1] < 0.1):    
            return 'FAIL'            
        return 'PASS'  


    def _ensure_required_data(self):
        """
        Ensures the datasets required for QC are local.
        If the download_data attribute is True,
        any missing data are downloaded.  If all the data are
        not present locally at the end of it an exception is raised.
        If the stream attribute is True, the video file is not
        required to be local, however it must be remotely accessible.
        TODO Simply check that the required data is present.
        If download_data is True, it should download the data via ONE.
        Should raise an error if there isn't the required
        data for the session
        :return:
        """
        assert self.one is not None, 'ONE required to download data'
        # Get extractor type
        for dstype in self.dstypes:
            dataset = self.one.datasets_from_type(self.eid, dstype)
            kwargs = {'download_only': True}
            present = (
                (self.one.load_dataset(self.eid, d, **kwargs) for d in dataset)
                if self.download_data
                else (next(self.session_path.rglob(d), None) for d in dataset)
            )
            assert (dataset and all(present)), f'Dataset {dstype} not found'

    def run(self, update: bool = False, **kwargs) -> (str, dict):
        """
        Run video QC checks and return outcome
        :param update: if True, updates the session QC fields on Alyx
        :param download_data: if True, downloads any missing data if required
        :returns: overall outcome as a str, a dict of checks and their outcomes
        TODO Shouldn't need to change this method
        """
        _log.info('Computing QC outcome')
        namespace = f'dlc{self.video_type.capitalize()}'
        if all(x is None for x in self.data.values()):
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(DlcQC, is_metric)
        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}
        all_pass = all(x is None or x == 'PASS' for x in self.metrics.values())
        self.outcome = 'PASS' if all_pass else 'FAIL'
        if update:
            bool_map = {k: None if v is None else v == 'PASS'
                        for k, v in self.metrics.items()}
            self.update_extended_qc(bool_map)
            self.update(self.outcome, namespace)
        return self.outcome, self.metrics


def run_all_qc(session, update=False,
               cameras=('left', 'right', 'body'), **kwargs):
    """Run QC for all cameras
    Run the camera QC for left, right and body cameras.
    :param session: A session path or eid.
    :param update: If True, QC fields are updated on Alyx.
    :param cameras: A list of camera names to perform QC on.
    :return: dict of CameraCQ objects
    """
    qc = {}
    for camera in cameras:
        qc[camera] = DlcQC(session, video_type=camera,
                           one=kwargs.pop('one', None))
        qc[camera].run(update=update, **kwargs)
    return qc

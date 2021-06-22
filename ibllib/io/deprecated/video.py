"""Functions for fetching video frames, meta data and file locations"""
import warnings
import re
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from brainbox.core import Bunch
from oneibl.one import ONE
from oneibl import params

VIDEO_LABELS = ('left', 'right', 'body')


class VideoStreamer:
    """
    Provides a wrapper to stream a video from a password protected HTTP server using opencv
    """

    def __init__(self, url_vid):
        """
        TODO Allow auth as input
        :param url_vid: full url of the video or dataset dictionary as output by alyx rest datasets
        :returns cv2.VideoCapture object
        """
        # pop the data url from the dataset record if the input is a dictionary
        if isinstance(url_vid, dict):
            url_vid = next(fr['data_url'] for fr in url_vid['file_records'] if fr['data_url'])
        self.url = url_vid
        self._par = params.get(silent=True)
        self.cap = cv2.VideoCapture(self._url)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def _url(self):
        username = self._par.HTTP_DATA_SERVER_LOGIN
        password = self._par.HTTP_DATA_SERVER_PWD
        return re.sub(r'(^https?://)', r'\1' + f'{username}:{password}@', self.url)

    def get_frame(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return self.cap.read()


def get_video_meta(video_path, one=None):
    """
    Return a bunch of video information with the fields ('length', 'fps', 'width', 'height',
    'duration', 'size')
    :param video_path: A path to the video.  May be a file path or URL.
    :param one: An instance of ONE
    :return: A Bunch of video mata data
    """
    warnings.warn('Support for oneibl.one will soon be removed, use one.api instead',
                  category=DeprecationWarning)
    is_url = isinstance(video_path, str) and video_path.startswith('http')
    cap = VideoStreamer(video_path).cap if is_url else cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f'Failed to open video file {video_path}'

    # Get basic properties of video
    meta = Bunch()
    meta.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta.fps = int(cap.get(cv2.CAP_PROP_FPS))
    meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta.duration = timedelta(seconds=meta.length / meta.fps) if meta.fps > 0 else 0
    if is_url and one:
        eid = one.eid_from_path(video_path)
        name = re.match(r'.*(_iblrig_[a-z]+Camera\.raw\.)(?:[\w-]{36}\.)?(mp4)$', video_path)
        det, = one.alyx.rest('datasets', 'list', session=eid, name=''.join(name.groups()))
        meta.size = det['file_size']
    elif is_url and not one:
        meta.size = None
    else:
        meta.size = Path(video_path).stat().st_size
    cap.release()
    return meta


def url_from_eid(eid, label=None, one=None):
    """Return the video URL(s) for a given eid

    :param eid: The session id
    :param label: The video label (e.g. 'body') or a tuple thereof
    :param one: An instance of ONE
    :return: The URL string if the label is a string, otherwise a dict of urls with labels as keys
    """
    warnings.warn('Support for oneibl.one will soon be removed, use one.api instead',
                  category=DeprecationWarning)

    valid_labels = ('left', 'right', 'body')
    if not (label is None or np.isin(label, ('left', 'right', 'body')).all()):
        raise ValueError('labels must be one of ("%s")' % '", "'.join(valid_labels))
    one = one or ONE()
    datasets = one.list_datasets(eid, details=True)

    def match(dataset):
        if dataset['dataset_type'] != '_iblrig_Camera.raw':
            return False
        if label:
            name = re.match(r'(?:_iblrig_)([a-z]+)(?=Camera.raw.mp4$)', dataset['name']).group(1)
            return name in label
        else:
            return True
    datasets = [ds for ds in datasets if match(ds)]
    urls = [next(r['data_url'] for r in ds['file_records'] if r['data_url']) for ds in datasets]
    # If one label specified, return the url, otherwise return a dict
    if isinstance(label, str):
        return urls[0]
    urls_dict = {label_from_path(url): url for url in urls}
    return {**dict.fromkeys(label), **urls_dict} if label else urls_dict


def label_from_path(video_name):
    """
    Return the video label, i.e. 'left', 'right' or 'body'
    :param video_name: A file path, URL or file name for the video
    :return: The string label or None if the video doesn't match
    """
    result = re.search(r'(?<=_)([a-z]+)(?=Camera)', str(video_name))
    return result.group() if result else None

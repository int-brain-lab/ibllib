"""Functions for fetching video frames, meta data and file locations"""
import sys
import re
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from brainbox.core import Bunch
from oneibl.stream import VideoStreamer
from oneibl.one import ONE

VIDEO_LABELS = ('left', 'right', 'body')


def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (w, h, 3)
    """
    is_url = isinstance(video_path, str) and video_path.startswith('http')
    cap = VideoStreamer(video_path).cap if is_url else cv2.VideoCapture(str(video_path))
    # 0-based index of the frame to be decoded/captured next.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_image = cap.read()
    cap.release()
    return frame_image


def get_video_frames_preload(vid, frame_numbers=None, mask=Ellipsis, as_list=False,
                             func=lambda x: x, quiet=False):
    """
    Obtain numpy array corresponding to a particular video frame in video.
    Fetching and returning a list is about 33% faster but may be less memory controlled. NB: Any
    gain in speed will be lost if subsequently converted to array.
    :param vid: URL or local path to mp4 file or cv2.VideoCapture instance.
    :param frame_numbers: video frames to be returned. If None, return all frames.
    :param mask: a logical mask or slice to apply to frames
    :param as_list: if true the frames are returned as a list, this is faster but may be less
    memory efficient
    :param func: Function to be applied to each frame. Applied after masking if applicable.
    :param quiet: if true, suppress frame loading progress output.
    :return: numpy array corresponding to frame of interest, or list if as_list is True.
    Default dimensions are (n, w, h, 3) where n = len(frame_numbers)

    Example - Load first 1000 frames, keeping only the first colour channel:
        frames = get_video_frames_preload(vid, range(1000), mask=np.s_[:, :, 0])
    """
    is_cap = not isinstance(vid, (str, Path))
    if is_cap:
        cap = vid
    else:
        is_url = isinstance(vid, str) and vid.startswith('http')
        cap = VideoStreamer(vid).cap if is_url else cv2.VideoCapture(str(vid))
    assert cap.isOpened(), 'Failed to open video'

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_numbers = frame_numbers if frame_numbers is not None else range(frame_count)

    # Setting the index is extremely slow; determine where frame index must be set
    # The first index is always explicitly set.
    to_set = np.insert(np.diff(frame_numbers), 0, 0) != 1
    if as_list:
        frame_images = [None] * len(frame_numbers)
    else:
        ret, frame = cap.read()
        frame_images = np.zeros((len(frame_numbers), *func(frame[mask or ...]).shape), np.uint8)
    for ii, i in enumerate(frame_numbers):
        if not quiet:
            sys.stdout.write(f'\rloading frame {ii}/{len(frame_numbers)}')
            sys.stdout.flush()
        if to_set[ii]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_images[ii] = func(frame[mask or ...])
        else:
            print(f'failed to read frame #{i}')
    if not is_cap:
        cap.release()
    if not quiet:
        sys.stdout.write('\x1b[2K\r')  # Erase current line in stdout
    return frame_images


def get_video_meta(video_path, one=None):
    """
    Return a bunch of video information with the fields ('length', 'fps', 'width', 'height',
    'duration', 'size')
    :param video_path: A path to the video.  May be a file path or URL.
    :param one: An instance of ONE
    :return: A Bunch of video mata data
    """
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
    valid_labels = ('left', 'right', 'body')
    if not (label is None or np.isin(label, ('left', 'right', 'body')).all()):
        raise ValueError('labels must be one of ("%s")' % '", "'.join(valid_labels))
    one = one or ONE()
    datasets = one.list(eid, details=True)

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


def assert_valid_label(label):
    """
    Raises a value error is the provided label is not supported.
    :param label: A video label to verify
    :return: the label in lowercase
    """
    if not isinstance(label, str):
        try:
            return tuple(map(assert_valid_label, label))
        except AttributeError:
            raise ValueError('label must be string or iterable of strings')
    if label.lower() not in VIDEO_LABELS:
        raise ValueError(f"camera must be one of ({', '.join(VIDEO_LABELS)})")
    return label.lower()

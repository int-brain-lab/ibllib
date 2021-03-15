import re

import cv2
import oneibl.params


class VideoStreamer(object):
    """
    Provides a wrapper to stream a video from a password protected HTTP server using opencv
    """

    def __init__(self, url_vid):
        """
        :param url_vid: full url of the video or dataset dictionary as output by alyx rest datasets
        :returns cv2.VideoCapture object
        """
        # pop the data url from the dataset record if the input is a dictionary
        if isinstance(url_vid, dict):
            url_vid = next(fr['data_url'] for fr in url_vid['file_records'] if fr['data_url'])
        self.url = url_vid
        self._par = oneibl.params.get(silent=True)
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

"""
Get video frames and metadata
===================================
Video frames and meta data can be loaded using the brainbox.io.video module, which contains
functions for loading individual or groups of frames efficiently.  The video may be streamed
remotely or loaded from a local file.
"""

import brainbox.io.video as video
from oneibl.one import ONE

# Example 1: loading a single frame
url = ''
frame = video.get_video_frame(url, 1000)

# Example 2: load video meta-data
"""
You can load all the information for a given video.  In order to load the video size from a URL 
an instance of ONE must be provided, otherwise this entry will be blank. An Bunch is returned 
with a number of fields.
"""
meta = video.get_video_meta(url, ONE(silent=True))
for k, v in meta:
    print(f'The video {k} = {v}')

# Example 3: loading multiple frames

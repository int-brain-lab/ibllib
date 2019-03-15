import logging
from pathlib import Path
import re

import numpy as np

SAMPLE_SIZE = 2  # int16
DEFAULT_CHUNK_SIZE = 1e6
logger_ = logging.getLogger('ibllib')


def read_meta_data(md_file):
    """
    Reads the spkike glx metadata file and parse in a dictionary
    Agnostic: does not make any assumption on the keys/content, it just parses key=values
    """
    with open(md_file) as fid:
        md = fid.read()
    d = {}
    for a in md.splitlines():
        k, v = a.split('=')
        # if all numbers, try to interpret the string
        if v and re.fullmatch('[0-9,.]*', v):
            v = [float(val) for val in v.split(',')]
            # scalars should not be nested
            if len(v) == 1:
                v = v[0]
        # tildes in keynames removed
        d[k.replace('~', '')] = v
    return d


class Reader:
    """
    Class for SpikeGLX reading purposes
    Some format description was found looking at the Matlab SDK here
    https://github.com/billkarsh/SpikeGLX/blob/master/MATLAB-SDK/DemoReadSGLXData.m
    """
    def __init__(self, sglx_file):
        self.file_bin = Path(sglx_file)
        self.nbytes = self.file_bin.stat().st_size
        file_meta_data = Path(sglx_file).with_suffix('.meta')
        if file_meta_data.exists():
            self.file_meta_data = file_meta_data
            self.meta = read_meta_data(file_meta_data)
            if self.nc * self.ns * 2 != self.nbytes:
                logger_.warning(str(sglx_file) + " : meta data and filesize do not checkout")
        else:
            self.file_meta_data = None
            self.meta = None

    @property
    def sf(self):
        """ :return: sampling frequency (Hz) """
        if not self.meta:
            return
        keyname = 'niSampRate'
        if self.meta.get('typeThis') == 'imec':
            keyname = 'imSampRate'
        return self.meta.get(keyname)

    @property
    def nc(self):
        """ :return: number of channels """
        if not self.meta:
            return
        return int(self.meta.get('nSavedChans'))

    @property
    def ns(self):
        """ :return: number of samples """
        if not self.meta:
            return
        return self.meta.get('fileTimeSecs') * self.sf

    def read_chunk(self, first_sample=0, last_sample=10000):
        """
        reads all channels from first_sample to last_sample
        """
        byt_offset = int(self.nc * first_sample * SAMPLE_SIZE)
        ns_to_read = last_sample - first_sample + 1
        with open(self.file_bin, 'rb') as fid:
            fid.seek(byt_offset)
            D = np.fromfile(fid, dtype=np.dtype('int16'), count=ns_to_read * self.nc
                            ).reshape((int(ns_to_read), int(self.nc)))
        return D


def read(sglx_file, first_sample=0, last_sample=10000):
    sglxr = Reader(sglx_file)
    D = sglxr.read()
    return D, sglxr.meta

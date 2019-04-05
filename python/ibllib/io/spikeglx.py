import logging
from pathlib import Path
import re

import numpy as np

SAMPLE_SIZE = 2  # int16
DEFAULT_BATCH_SIZE = 1e6
logger_ = logging.getLogger('ibllib')


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
        if not file_meta_data.exists():
            self.file_meta_data = None
            self.meta = None
            self.gain_channels = 1
            logger_.warning(str(sglx_file) + " : no metadata file found. Very limited support")
        else:
            self.file_meta_data = file_meta_data
            self.meta = read_meta_data(file_meta_data)
            if self.nc * self.ns * 2 != self.nbytes:
                logger_.warning(str(sglx_file) + " : meta data and filesize do not checkout")
            self.gain_channels = _gain_channels(self.meta)

    @property
    def type(self):
        """ :return: ap or lf. Useful to index dictionaries """
        if not self.meta:
            return 0
        if self.meta['snsApLfSy'][0] == 0 and self.meta['snsApLfSy'][1] != 0:
            return 'lf'
        elif self.meta['snsApLfSy'][0] != 0 and self.meta['snsApLfSy'][1] == 0:
            return 'ap'

    @property
    def int2volts(self):
        """ :return: Conversion scalar to Volts. Needs to be combined with channel gains """
        if not self.meta:
            return 1
        if self.meta.get('typeThis', None):
            return self.meta.get('imAiRangeMax') / 512
        else:
            return self.meta.get('imAiRangeMax') / 768

    @property
    def fs(self):
        """ :return: sampling frequency (Hz) """
        if not self.meta:
            return 1
        keyname = 'niSampRate'
        if self.meta.get('typeThis') == 'imec':
            keyname = 'imSampRate'
        return self.meta.get(keyname)

    @property
    def nc(self):
        """ :return: number of channels """
        if not self.meta:
            return
        return int(sum(self.meta.get('snsApLfSy')))

    @property
    def ns(self):
        """ :return: number of samples """
        if not self.meta:
            return
        return self.meta.get('fileTimeSecs') * self.fs

    def read_samples(self, first_sample=0, last_sample=10000, sync_trace=False):
        """
        reads all channels from first_sample to last_sample, following numpy slicing convention
        sglx.read_samples(first=0, last=100) would be equivalent to slicing the array D
        D[:,0:100] where the last axis represent time and the first channels.
         :return: numpy array of int16
        """
        byt_offset = int(self.nc * first_sample * SAMPLE_SIZE)
        ns_to_read = last_sample - first_sample
        with open(self.file_bin, 'rb') as fid:
            fid.seek(byt_offset)
            darray = np.fromfile(fid, dtype=np.dtype('int16'), count=ns_to_read * self.nc
                                 ).reshape((int(ns_to_read), int(self.nc)))
        darray = np.float32(darray) / self.gain_channels[self.type] * self.int2volts
        return darray


def read(sglx_file, first_sample=0, last_sample=10000):
    sglxr = Reader(sglx_file)
    D = sglxr.read()
    return D, sglxr.meta


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


def _gain_channels(meta_data):
    """
    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: numpy array with one gain value per channel
    """
    # interprets the gain value from the metadata header
    if 'imroTbl' in meta_data.keys():
        sy_gain = np.ones(int(meta_data['snsApLfSy'][-1]), dtype=np.float32)
        # the sync traces are not included in the gain values, so are included for broadcast ops
        gain = re.findall(r'([0-9]* [0-9]* [0-9]* [0-9]* [0-9]*)', meta_data['imroTbl'])
        out = {'lf': np.hstack((np.array([np.float32(g.split(' ')[-1]) for g in gain]), sy_gain)),
               'ap': np.hstack((np.array([np.float32(g.split(' ')[-2]) for g in gain]), sy_gain))}
    elif 'niMNGain' in meta_data.keys():
        raise NotImplementedError()
    return out

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
            self.gain_channels = _gain_channels_from_meta(self.meta)
            self.memmap = np.memmap(sglx_file, dtype='int16', mode='r', shape=(self.ns, self.nc))

    @property
    def type(self):
        """:return: ap or lf. Useful to index dictionaries """
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
        return int(self.meta.get('fileTimeSecs') * self.fs)

    def read_samples(self, first_sample=0, last_sample=10000):
        """
        reads all channels from first_sample to last_sample, following numpy slicing convention
        sglx.read_samples(first=0, last=100) would be equivalent to slicing the array D
        D[:,0:100] where the last axis represent time and the first channels.

         :param first_sample: first sample to be read, python slice-wise
         :param last_sample:  last sample to be read, python slice-wise
         :return: numpy array of int16
        """
        byt_offset = int(self.nc * first_sample * SAMPLE_SIZE)
        ns_to_read = last_sample - first_sample
        with open(self.file_bin, 'rb') as fid:
            fid.seek(byt_offset)
            darray = np.fromfile(fid, dtype=np.dtype('int16'), count=ns_to_read * self.nc
                                 ).reshape((int(ns_to_read), int(self.nc)))
        # we don't want to apply any gain on the sync trace
        sync_tr_ind = np.where(self.gain_channels[self.type] == 1.)
        gain = 1 / self.gain_channels[self.type] * self.int2volts
        gain[sync_tr_ind] = 1.
        sync = split_sync(darray[:, sync_tr_ind])
        darray = np.float32(darray) * gain
        return darray, sync

    def read_sync(self, slice=slice(0, 10000)):
        """
        Reads only the sync trace at specified samples using slicing syntax

        >>> sync_samples = sr.read_sync(0:10000)
        """
        if not(self.meta and self.meta['acqApLfSy'][2]):
            logger_.warning('Sync trace not labeled in metadata. Assuming last trace')
        return split_sync(self.memmap[slice, -1])


def read(sglx_file, first_sample=0, last_sample=10000):
    """
    Function to read from a spikeglx binary file without instantiating the class.
    Gets the meta-data as well.

    >>> ibllib.io.spikeglx.read('/path/to/file.bin', first_sample=0, last_sample=1000)

    :param sglx_file: full path the the binary file to read
    :param first_sample: first sample to be read, python slice-wise
    :param last_sample: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
    """
    sglxr = Reader(sglx_file)
    D, sync = sglxr.read_samples(first_sample=first_sample, last_sample=last_sample)
    return D, sync, sglxr.meta


def read_meta_data(md_file):
    """
    Reads the spkike glx metadata file and parse in a dictionary
    Agnostic: does not make any assumption on the keys/content, it just parses key=values

    :param md_file: last sample to be read, python slice-wise
    :return: Data array, sync trace, meta-data
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


def _map_channels_from_meta(meta_data):
    """
    Interpret the meta data string to extract an array of channel positions along the shank

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: dictionary of arrays 'shank', 'col', 'row', 'flag', one value per active site
    """
    if 'snsShankMap' in meta_data.keys():
        chmap = re.findall(r'([0-9]*:[0-9]*:[0-9]*:[0-9]*)', meta_data['snsShankMap'])
        # shank#, col#, row#, drawflag
        # (nb: drawflag is one should be drawn and considered spatial average)
        chmap = np.array([np.float32(cm.split(':')) for cm in chmap])
        return {k: chmap[:, v] for (k, v) in {'shank': 0, 'col': 1, 'row': 2, 'flag': 3}.items()}


def _gain_channels_from_meta(meta_data):
    """
    Interpret the meta data string to extract an array of gain values for each channel

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


def split_sync(sync_tr):
    """
    The synchronization channelx are stored as single bits, this will split the int16 original
    channel into 16 single bits channels

    :param sync_tr: numpy vector: samples of synchronisation trace
    :return: int8 numpy array of 16 channels, 1 column per sync trace
    """
    sync_tr = np.int16(np.copy(sync_tr))
    out = np.unpackbits(sync_tr.view(np.uint8)).reshape(sync_tr.size, 16)
    out = np.flip(np.roll(out, 8, axis=1), axis=1)
    return np.int8(out)

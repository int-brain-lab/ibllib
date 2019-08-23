import json
import logging
from pathlib import Path
import re

import numpy as np

from brainbox.core import Bunch
from ibllib.ephys import neuropixel as neuropixel


SAMPLE_SIZE = 2  # int16
DEFAULT_BATCH_SIZE = 1e6
_logger = logging.getLogger('ibllib')


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
            self.channel_conversion_sample2mv = 1
            _logger.warning(str(sglx_file) + " : no metadata file found. Very limited support")
        else:
            self.file_meta_data = file_meta_data
            self.meta = read_meta_data(file_meta_data)
            if self.nc * self.ns * 2 != self.nbytes:
                _logger.warning(str(sglx_file) + " : meta data and filesize do not checkout")
            self.channel_conversion_sample2mv = _conversion_sample2mv_from_meta(self.meta)
            self.memmap = np.memmap(sglx_file, dtype='int16', mode='r', shape=(self.ns, self.nc))

    @property
    def version(self):
        if not self.meta:
            return None
        return _get_neuropixel_version_from_meta(self.meta)

    @property
    def type(self):
        """:return: ap, lf or nidq. Useful to index dictionaries """
        if not self.meta:
            return 0
        return _get_type_from_meta(self.meta)

    @property
    def fs(self):
        """ :return: sampling frequency (Hz) """
        if not self.meta:
            return 1
        return _get_fs_from_meta(self.meta)

    @property
    def nc(self):
        """ :return: number of channels """
        if not self.meta:
            return
        return _get_nchannels_from_meta(self.meta)

    @property
    def ns(self):
        """ :return: number of samples """
        if not self.meta:
            return
        return int(np.round(self.meta.get('fileTimeSecs') * self.fs))

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
        darray = np.float32(darray) * self.channel_conversion_sample2mv[self.type]
        sync = split_sync(darray[:, _get_sync_trace_indices_from_meta(self.meta)])
        return darray, sync

    def read_sync(self, _slice=slice(0, 10000)):
        """
        Reads only the sync trace at specified samples using slicing syntax

        >>> sync_samples = sr.read_sync(0:10000)
        """
        if not self.meta:
            _logger.warning('Sync trace not labeled in metadata. Assuming last trace')
        return split_sync(self.memmap[_slice, _get_sync_trace_indices_from_meta(self.meta)])


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
        if v and re.fullmatch('[0-9,.]*', v) and v.count('.') < 2:
            v = [float(val) for val in v.split(',')]
            # scalars should not be nested
            if len(v) == 1:
                v = v[0]
        # tildes in keynames removed
        d[k.replace('~', '')] = v
    d['neuropixelVersion'] = _get_neuropixel_version_from_meta(d)
    return Bunch(d)


def _get_neuropixel_version_from_meta(md):
    """
    Get neuropixel version tag (3A, 3B1, 3B2) from the metadata dictionary
    """
    if 'typeEnabled' in md.keys():
        return '3A'
    elif 'typeImEnabled' in md.keys() and 'typeNiEnabled' in md.keys():
        if 'imDatPrb_port' in md.keys() and 'imDatPrb_slot' in md.keys():
            return '3B2'
        else:
            return '3B1'


def _get_sync_trace_indices_from_meta(md):
    """
    Returns a list containing indices of the sync traces in the original array
    """
    typ = _get_type_from_meta(md)
    ntr = int(_get_nchannels_from_meta(md))
    if typ == 'nidq':
        nsync = int(md.get('snsMnMaXaDw')[-1])
    elif typ in ['lf', 'ap']:
        nsync = int(md.get('snsApLfSy')[2])
    return list(range(ntr - nsync, ntr))


def _get_nchannels_from_meta(md):
    typ = _get_type_from_meta(md)
    if typ == 'nidq':
        return int(np.round(np.sum(md.get('snsMnMaXaDw'))))
    elif typ in ['lf', 'ap']:
        return int(np.round(sum(md.get('snsApLfSy'))))


def _get_fs_from_meta(md):
    if md.get('typeThis') == 'imec':
        return md.get('imSampRate')
    else:
        return md.get('niSampRate')


def _get_type_from_meta(md):
    """
    Get neuropixel data type (ap, lf or nidq) from metadata
    """
    snsApLfSy = md.get('snsApLfSy', [-1, -1, -1])
    if snsApLfSy[0] == 0 and snsApLfSy[1] != 0:
        return 'lf'
    elif snsApLfSy[0] != 0 and snsApLfSy[1] == 0:
        return 'ap'
    elif snsApLfSy == [-1, -1, -1] and md.get('typeThis', None) == 'nidq':
        return 'nidq'


def _map_channels_from_meta(meta_data):
    """
    Interpret the meta data string to extract an array of channel positions along the shank

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: dictionary of arrays 'shank', 'col', 'row', 'flag', one value per active site
    """
    if 'snsShankMap' in meta_data.keys():
        chmap = re.findall(r'([0-9]*:[0-9]*:[0-9]*:[0-9]*)', meta_data['snsShankMap'])
        # for digital nidq types, the key exists but does not contain any information
        if not chmap:
            return {'shank': None, 'col': None, 'row': None, 'flag': None}
        # shank#, col#, row#, drawflag
        # (nb: drawflag is one should be drawn and considered spatial average)
        chmap = np.array([np.float32(cm.split(':')) for cm in chmap])
        return {k: chmap[:, v] for (k, v) in {'shank': 0, 'col': 1, 'row': 2, 'flag': 3}.items()}


def _conversion_sample2mv_from_meta(meta_data):
    """
    Interpret the meta data string to extract an array of conversion factprs for each channel

    :param meta_data: dictionary output from  spikeglx.read_meta_data
    :return: numpy array with one gain value per channel
    """

    def int2volts(md):
        """ :return: Conversion scalar to Volts. Needs to be combined with channel gains """
        if md.get('typeThis', None) == 'imec':
            return md.get('imAiRangeMax') / 512
        else:
            return md.get('niAiRangeMax') / 32768

    int2volt = int2volts(meta_data)
    # interprets the gain value from the metadata header
    if 'imroTbl' in meta_data.keys():
        sy_gain = np.ones(int(meta_data['snsApLfSy'][-1]), dtype=np.float32)
        # the sync traces are not included in the gain values, so are included for broadcast ops
        gain = re.findall(r'([0-9]* [0-9]* [0-9]* [0-9]* [0-9]*)', meta_data['imroTbl'])
        out = {'lf': np.hstack((np.array([1 / np.float32(g.split(' ')[-1]) for g in gain]) *
                                int2volt, sy_gain)),
               'ap': np.hstack((np.array([1 / np.float32(g.split(' ')[-2]) for g in gain]) *
                                int2volt, sy_gain))}
    elif 'niMNGain' in meta_data.keys():
        gain = np.r_[
            1 / np.ones(int(meta_data['snsMnMaXaDw'][0],)) * meta_data['niMNGain'] * int2volt,
            1 / np.ones(int(meta_data['snsMnMaXaDw'][1],)) * meta_data['niMAGain'] * int2volt,
            np.ones(int(np.sum(meta_data['snsMnMaXaDw'][2:]),))]
        out = {'nidq': gain}
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


def glob_ephys_files(session_path):
    """
    From an arbitrary folder (usually session folder) gets the ap and lf files and labels
    Associated to the subfolders where they are
    the expected folder tree is:
    ├── 3A
    │   ├── imec0
    │   │   ├── sync_testing_g0_t0.imec0.ap.bin
    │   │   └── sync_testing_g0_t0.imec0.lf.bin
    │   └── imec1
    │       ├── sync_testing_g0_t0.imec1.ap.bin
    │       └── sync_testing_g0_t0.imec1.lf.bin
    └── 3B
        ├── sync_testing_g0_t0.nidq.bin
        ├── imec0
        │   ├── sync_testing_g0_t0.imec0.ap.bin
        │   └── sync_testing_g0_t0.imec0.lf.bin
        └── imec1
            ├── sync_testing_g0_t0.imec1.ap.bin
            └── sync_testing_g0_t0.imec1.lf.bin

    :param session_path: folder, string or pathlib.Path
    :returns: a list of dictionaries with keys 'ap': apfile, 'lf': lffile and 'label'
    """
    def get_label(raw_ephys_apfile):
        if raw_ephys_apfile.parts[-2] != 'raw_ephys_data':
            return raw_ephys_apfile.parts[-2]
        else:
            return ''

    ephys_files = []
    for raw_ephys_apfile in Path(session_path).rglob('*.ap.bin'):
        # first get the ap file
        ephys_files.extend([Bunch({'label': None, 'ap': None, 'lf': None, 'path': None})])
        ephys_files[-1].ap = raw_ephys_apfile
        # then get the corresponding lf file if it exists
        lf_file = raw_ephys_apfile.parent / raw_ephys_apfile.name.replace('.ap.', '.lf.')
        if lf_file.exists():
            ephys_files[-1].lf = lf_file
        # finally, the label is the current directory except if it is bare in raw_ephys_data
        ephys_files[-1].label = get_label(raw_ephys_apfile)
        ephys_files[-1].path = raw_ephys_apfile.parent
    # for 3b probes, need also to get the nidq dataset type
    for raw_ephys_nidqfile in Path(session_path).rglob('*.nidq.bin'):
        ephys_files.extend([Bunch({'label': get_label(raw_ephys_nidqfile),
                                   'nidq': raw_ephys_nidqfile,
                                   'path': raw_ephys_nidqfile.parent})])
    return ephys_files


def _mock_spikeglx_file(mock_path, meta_file, ns, nc, sync_depth):
    """
    For testing purposes, create a binary file with sync pulses to test reading and extraction
    """
    tmp_meta_file = Path(mock_path).joinpath(meta_file.name)
    tmp_bin_file = Path(mock_path).joinpath(meta_file.name).with_suffix('.bin')
    md = read_meta_data(meta_file)
    fs = _get_fs_from_meta(md)
    fid_source = open(meta_file)
    fid_target = open(tmp_meta_file, 'w+')
    line = fid_source.readline()
    while line:
        line = fid_source.readline()
        if line.startswith('fileSizeBytes'):
            line = f'fileSizeBytes={ns * nc * 2}\n'
        if line.startswith('fileTimeSecs'):
            line = f'fileTimeSecs={ns / fs}\n'
        fid_target.write(line)
    fid_source.close()
    fid_target.close()
    # each channel as an int of chn + 1
    D = np.tile(np.int16(np.arange(nc) + 1), (ns, 1))
    # the last channel is the sync that we fill with
    sync = np.uint16(2 ** np.float32(np.arange(-1, sync_depth)))
    D[:, -1] = 0
    D[:sync.size, -1] = sync
    with open(tmp_bin_file, 'w+') as fid:
        D.tofile(fid)
    return {'bin_file': tmp_bin_file, 'ns': ns, 'nc': nc, 'sync_depth': sync_depth, 'D': D}


def get_hardware_config(config_file):
    """
    Reads the neuropixel_wirings.json file containing sync mapping and parameters
    :param config_file: folder or json file
    :return: dictionary or None
    """
    config_file = Path(config_file)
    if config_file.is_dir():
        config_file = list(config_file.glob('*.wiring.json'))
        if config_file:
            config_file = config_file[0]
    if not config_file or not config_file.exists():
        _logger.warning(f"No neuropixel *.wiring.json file found in {str(config_file)}")
        return
    with open(config_file) as fid:
        par = json.loads(fid.read())
    return par


def _sync_map_from_hardware_config(hardware_config):
    """
    :param hardware_config: dictonary from json read of neuropixel_wirings.json
    :return: dictionary where key names refer to object and values to sync channel index
    """
    pin_out = neuropixel.SYNC_PIN_OUT[hardware_config['SYSTEM']]
    sync_map = {hardware_config['SYNC_WIRING_DIGITAL'][pin]: pin_out[pin]
                for pin in hardware_config['SYNC_WIRING_DIGITAL']
                if pin_out[pin] is not None}
    return sync_map


def get_sync_map(folder_ephys):
    hc = get_hardware_config(folder_ephys)
    if not hc:
        _logger.warning(f"No channel map for {str(folder_ephys)}")
        return None
    else:
        return _sync_map_from_hardware_config(hc)

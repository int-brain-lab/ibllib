"""Data extraction from fibrephotometry DAQ files.

Below is the expected folder structure for a fibrephotometry session:

    subject/
    ├─ 2021-06-30/
    │  ├─ 001/
    │  │  ├─ raw_photometry_data/
    │  │  │  │  ├─ _neurophotometrics_fpData.raw.pqt
    │  │  │  │  ├─ _neurophotometrics_fpData.channels.csv
    │  │  │  │  ├─ _mcc_DAQdata.raw.tdms

fpData.raw.pqt is a copy of the 'FPdata' file, the output of the Neuophotometrics Bonsai workflow.
fpData.channels.csv is table of frame flags for deciphering LED and GPIO states. The default table,
copied from the Neurophotometrics manual can be found in iblscripts/deploy/fppc/
_mcc_DAQdata.raw.tdms is the DAQ tdms file, containing the pulses from bpod and from the neurophotometrics system
"""
import logging

import pandas as pd
import numpy as np
import scipy.interpolate

import one.alf.io as alfio
from ibllib.io.extractors.base import BaseExtractor
from ibllib.io.raw_daq_loaders import load_channels_tdms, load_raw_daq_tdms
from neurodsp.utils import rises

_logger = logging.getLogger(__name__)

DAQ_CHMAP = {"photometry": 'AI0', 'bpod': 'AI1'}
V_THRESHOLD = 3

"""Available LEDs on the Neurophotometrics FP3002"""
LIGHT_SOURCE_MAP = {
    0: 'None',
    415: 'Violet',
    470: 'Blue',
    560: 'Green'
}


def sync_photometry_to_daq(daq_file, photometry_file, chmap=DAQ_CHMAP, v_threshold=V_THRESHOLD):
    """
    :param daq_file: tdms file
    :param photometry_file:
    :param chmap:
    :param v_threshold:
    :return:
    """
    df_photometry = pd.read_csv(photometry_file)
    daq_frames, tag_daq_frames, fs, vdaq = read_daq_timestamps(daq_file, chmap=chmap, v_threshold=v_threshold)
    nf = np.minimum(tag_daq_frames.size, df_photometry['Input0'].size)
    ipeak = np.argmax(np.correlate(tag_daq_frames[:nf].astype(np.int8), df_photometry['Input0'].values[:nf], mode='full'))
    # if the frame shift is negative, it means that the photometry frames are early
    frame_shift = ipeak - nf + 1

    df = np.median(np.diff(df_photometry['Timestamp']))
    interp = scipy.interpolate.interp1d(daq_frames[:nf] / fs, df_photometry['Timestamp'][:nf])
    drift_ppm = (np.polyfit(daq_frames[:nf] / fs, df_photometry['Timestamp'][:nf], 1)[0] - 1) * 1e6
    _logger.info(f"drift PPM: {drift_ppm}")

    # here is a bunch of safeguards
    assert np.all(np.abs(np.diff(daq_frames) - df * fs) < 1)  # check that there are no missed frames on daq
    assert np.unique(np.diff(df_photometry['FrameCounter'])).size == 1  # checks that there are no missed frames on photo
    assert frame_shift == 0  # it's always the end frames that are missing
    assert drift_ppm < 20
    return interp, drift_ppm


def read_daq_timestamps(daq_file, chmap=DAQ_CHMAP, v_threshold=V_THRESHOLD):
    """
    From a tdms daq file, extracts the photometry frames and their tagging.
    :param daq_file:
    :param chmap:
    :param v_threshold:
    :return:
    """
    channel_names = [c.name for c in load_raw_daq_tdms(daq_file)['Analog'].channels()]
    assert all([v in channel_names for v in chmap.values()]), "Missing channel"
    vdaq, fs = load_channels_tdms(daq_file, chmap=chmap, return_fs=True)
    daq_frames = rises(vdaq['photometry'], step=v_threshold, analog=True)
    tagged_frames = vdaq['bpod'][daq_frames] > v_threshold
    return daq_frames, tagged_frames, fs, vdaq


def check_timestamps(daq_file, photometry_file, tolerance=20, chmap=DAQ_CHMAP, v_threshold=V_THRESHOLD):
    """
    Reads data file and checks that the number of timestamps check out with a tolerance of n_frames
    :param daq_file:
    :param photometry_file:
    :param tolerance: number of acceptable missing frames between the daq and the photometry file
    :param chmap:
    :param v_threshold:
    :return: None
    """
    df_photometry = pd.read_csv(photometry_file)
    daq_frames = read_daq_timestamps(daq_file, chmap=chmap, v_threshold=v_threshold)[0]
    assert (daq_frames.shape[0] - df_photometry.shape[0]) < tolerance
    _logger.info(f"{daq_frames.shape[0] - df_photometry.shape[0]} frames difference, "
                 f"{'/'.join(daq_file.parts[-2:])}: {daq_frames.shape[0]} frames, "
                 f"{'/'.join(photometry_file.parts[-2:])}: {df_photometry.shape[0]}")


class FibrePhotometry(BaseExtractor):
    save_names = ('photometry.signal.npy', 'photometry.photometryLightSource.npy',
                  'photometryLightSource.properties.tsv', 'photometry.times.npy')
    var_names = ('signal', 'lightSource', 'lightSource_properties', 'timestamps')

    def __init__(self, *args, **kwargs):
        """An extractor for all Neurophotometrics fibrephotometry data"""
        super().__init__(*args, **kwargs)

    @staticmethod
    def _channel_meta(light_source_map=None):
        """
        Return table of light source wavelengths and corresponding colour labels.

        Parameters
        ----------
        light_source_map : dict
            An optional map of light source wavelengths (nm) used and their corresponding colour name.

        Returns
        -------
        pandas.DataFrame
            A sorted table of wavelength and colour name.
        """
        light_source_map = light_source_map or LIGHT_SOURCE_MAP
        names = ('wavelength', 'color')
        meta = pd.DataFrame(sorted(light_source_map.items()), columns=names)
        meta.index.rename('channel_id', inplace=True)
        return meta

    def _extract(self, light_source_map=None, **kwargs):
        """

        Parameters
        ----------
        light_source_map : dict
            An optional map of light source wavelengths (nm) used and their corresponding colour name.

        Returns
        -------
        numpy.ndarray
            A 1D array of signal values.
        numpy.ndarray
            A 1D array of ints corresponding to the active light source during a given frame.
        pandas.DataFrame
            A table of light source IDs, their wavelength in nm and corresponding colour name.
        """
        fp_data = alfio.load_object(self.session_path / 'raw_photometry_data', 'fpData')
        ts = self.extract_timestamps(fp_data, **kwargs)

        # Load channels and
        channel_meta_map = self._channel_meta(kwargs.get('light_source_map'))
        channel_map = fp_data['channels'].set_index('Condition')
        # Extract signal columns into 2D array
        signal = fp_data['raw'].filter(like='Region', axis=1).sort_index(axis=1).values
        # Extract channel index
        state = fp_data['raw'].get('LedState', fp_data['raw'].get('Flags', None))
        channel_id = np.empty_like(state)
        for (label, ch), (id, wavelength) in zip(channel_map.items(), channel_meta_map['wavelength'].items()):
            mask = state.isin(ch)
            channel_id[mask] = id
            if wavelength != 0:
                assert str(wavelength) in label

        return signal, channel_id, channel_meta_map, ts

    def extract_timestamps(self, fp_data, **kwargs):
        """Extract the photometry.timestamps array.

        This is dependant on the DAQ and task synchronization protocol.

        Parameters
        ----------
        fp_data : dict
            A Bunch of raw fibrephotometry data, with the keys ('raw', 'channels').

        Returns
        -------
        numpy.ndarray
            An array of timestamps, one per frame.
        """
        raise NotImplementedError  # To subclass

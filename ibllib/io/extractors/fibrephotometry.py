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
from ibllib.io.extractors.training_trials import GoCueTriggerTimes
from neurodsp.utils import rises, sync_timestamps

_logger = logging.getLogger(__name__)

DAQ_CHMAP = {"photometry": 'AI0', 'bpod': 'AI1'}
V_THRESHOLD = 3

"""
Neurophotometrics FP3002 specific information.
The light source map refers to the available LEDs on the system.
The flags refers to the byte encoding of led states in the system.
"""
LIGHT_SOURCE_MAP = {
    'color': ['None', 'Violet', 'Blue', 'Green'],
    'wavelength': [0, 415, 470, 560],
    'name': ['None', 'Isosbestic', 'GCaMP', 'RCaMP'],
}

NEUROPHOTOMETRICS_LED_STATES = {
    'Condition': {
        0: 'No additional signal',
        1: 'Output 0 signal HIGH + Stimulation',
        2: 'Output 0 signal HIGH + Input 0 signal HIGH',
        3: 'Input 0 signal HIGH + Stimulation',
        4: 'Output 0 HIGH + Input 0 HIGH + Stimulation'
    },
    'No LED ON': {0: 0, 1: 48, 2: 528, 3: 544, 4: 560},
    'L415': {0: 1, 1: 49, 2: 529, 3: 545, 4: 561},
    'L470': {0: 2, 1: 50, 2: 530, 3: 546, 4: 562},
    'L560': {0: 4, 1: 52, 2: 532, 3: 548, 4: 564}
}


def sync_photometry_to_daq(vdaq, fs, df_photometry, chmap=DAQ_CHMAP, v_threshold=V_THRESHOLD):
    """
    :param vdaq: dictionary of daq traces.
    :param fs: sampling frequency
    :param df_photometry:
    :param chmap:
    :param v_threshold:
    :return:
    """
    # here we take the flag that is the most common
    daq_frames, tag_daq_frames = read_daq_timestamps(vdaq=vdaq, v_threshold=v_threshold)
    nf = np.minimum(tag_daq_frames.size, df_photometry['Input0'].size)

    # we compute the framecounter for the DAQ, and match the bpod up state frame by frame for different shifts
    # the shift that minimizes the mismatch is usually good
    df = np.median(np.diff(df_photometry['Timestamp']))
    fc = np.cumsum(np.round(np.diff(daq_frames) / fs / df).astype(np.int32)) - 1  # this is a daq frame counter
    fc = fc[fc < (nf - 1)]
    max_shift = 300
    error = np.zeros(max_shift * 2 + 1)
    shifts = np.arange(-max_shift, max_shift + 1)
    for i, shift in enumerate(shifts):
        rolled_fp = np.roll(df_photometry['Input0'].values[fc], shift)
        error[i] = np.sum(np.abs(rolled_fp - tag_daq_frames[:fc.size]))
    # a negative shift means that the DAQ is ahead of the photometry and that the DAQ misses frame at the beginning
    frame_shift = shifts[np.argmax(-error)]
    if np.sign(frame_shift) == -1:
        ifp = fc[np.abs(frame_shift):]
    elif np.sign(frame_shift) == 0:
        ifp = fc
    elif np.sign(frame_shift) == 1:
        ifp = fc[:-np.abs(frame_shift)]
    t_photometry = df_photometry['Timestamp'].values[ifp]
    t_daq = daq_frames[:ifp.size] / fs
    # import matplotlib.pyplot as plt
    # plt.plot(shifts, -error)
    fcn_fp2daq = scipy.interpolate.interp1d(t_photometry, t_daq, fill_value='extrapolate')
    drift_ppm = (np.polyfit(t_daq, t_photometry, 1)[0] - 1) * 1e6
    if drift_ppm > 120:
        _logger.warning(f"drift photometry to DAQ PPM: {drift_ppm}")
    else:
        _logger.info(f"drift photometry to DAQ PPM: {drift_ppm}")
    # here is a bunch of safeguards
    assert np.unique(np.diff(df_photometry['FrameCounter'])).size == 1  # checks that there are no missed frames on photo
    assert np.abs(frame_shift) <= 5  # it's always the end frames that are missing
    assert np.abs(drift_ppm) < 60
    ts_daq = fcn_fp2daq(df_photometry['Timestamp'].values)  # those are the timestamps in daq time
    return ts_daq, fcn_fp2daq, drift_ppm


def read_daq_voltage(daq_file, chmap=DAQ_CHMAP):
    channel_names = [c.name for c in load_raw_daq_tdms(daq_file)['Analog'].channels()]
    assert all([v in channel_names for v in chmap.values()]), "Missing channel"
    vdaq, fs = load_channels_tdms(daq_file, chmap=chmap, return_fs=True)
    vdaq = {k: v - np.median(v) for k, v in vdaq.items()}
    return vdaq, fs


def read_daq_timestamps(vdaq, v_threshold=V_THRESHOLD):
    """
    From a tdms daq file, extracts the photometry frames and their tagging.
    :param vsaq: dictionary of the voltage traces from the DAQ. Each item has a key describing
    the channel as per the channel map, and contains a single voltage trace.
    :param v_threshold:
    :return:
    """
    daq_frames = rises(vdaq['photometry'], step=v_threshold, analog=True)
    if daq_frames.size == 0:
        daq_frames = rises(-vdaq['photometry'], step=v_threshold, analog=True)
        _logger.warning(f'No photometry pulses detected, attempting to reverse voltage and detect again,'
                        f'found {daq_frames.size} in reverse voltage. CHECK YOUR FP WIRING TO THE DAQ !!')
    tagged_frames = vdaq['bpod'][daq_frames] > v_threshold
    return daq_frames, tagged_frames


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
    v, fs = read_daq_voltage(daq_file=daq_file, chmap=chmap)
    daq_frames, _ = read_daq_timestamps(vdaq=v, v_threshold=v_threshold)
    assert (daq_frames.shape[0] - df_photometry.shape[0]) < tolerance
    _logger.info(f"{daq_frames.shape[0] - df_photometry.shape[0]} frames difference, "
                 f"{'/'.join(daq_file.parts[-2:])}: {daq_frames.shape[0]} frames, "
                 f"{'/'.join(photometry_file.parts[-2:])}: {df_photometry.shape[0]}")


class FibrePhotometry(BaseExtractor):
    """
        FibrePhotometry(self.session_path, collection=self.collection)
    """
    save_names = ('photometry.signal.pqt')
    var_names = ('df_out')

    def __init__(self, *args, collection='raw_photometry_data', **kwargs):
        """An extractor for all Neurophotometrics fibrephotometry data"""
        self.collection = collection
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
        meta = pd.DataFrame.from_dict(light_source_map)
        meta.index.rename('channel_id', inplace=True)
        return meta

    def _extract(self, light_source_map=None, collection=None, regions=None, **kwargs):
        """

        Parameters
        ----------
        regions: list of str
            The list of regions to extract. If None extracts all columns containing "Region". Defaults to None.
        light_source_map : dict
            An optional map of light source wavelengths (nm) used and their corresponding colour name.
        collection: str / pathlib.Path
            An optional relative path from the session root folder to find the raw photometry data.
            Defaults to `raw_photometry_data`

        Returns
        -------
        numpy.ndarray
            A 1D array of signal values.
        numpy.ndarray
            A 1D array of ints corresponding to the active light source during a given frame.
        pandas.DataFrame
            A table of intensity for each region, with associated times, wavelengths, names and colors
        """
        collection = collection or self.collection
        fp_data = alfio.load_object(self.session_path / collection, 'fpData')
        ts = self.extract_timestamps(fp_data['raw'], **kwargs)

        # Load channels and
        channel_meta_map = self._channel_meta(kwargs.get('light_source_map'))
        led_states = fp_data.get('channels', pd.DataFrame(NEUROPHOTOMETRICS_LED_STATES))
        led_states = led_states.set_index('Condition')
        # Extract signal columns into 2D array
        regions = regions or [k for k in fp_data['raw'].keys() if 'Region' in k]
        out_df = fp_data['raw'].filter(items=regions, axis=1).sort_index(axis=1)
        out_df['times'] = ts
        out_df['wavelength'] = np.NaN
        out_df['name'] = ''
        out_df['color'] = ''
        # Extract channel index
        states = fp_data['raw'].get('LedState', fp_data['raw'].get('Flags', None))
        for state in states.unique():
            ir, ic = np.where(led_states == state)
            if ic.size == 0:
                continue
            for cn in ['name', 'color', 'wavelength']:
                out_df.loc[states == state, cn] = channel_meta_map.iloc[ic[0]][cn]
        return out_df

    def extract_timestamps(self, fp_data, **kwargs):
        """Extract the photometry.timestamps array.

        This depends on the DAQ and task synchronization protocol.

        Parameters
        ----------
        fp_data : dict
            A Bunch of raw fibrephotometry data, with the keys ('raw', 'channels').

        Returns
        -------
        numpy.ndarray
            An array of timestamps, one per frame.
        """
        daq_file = next(self.session_path.joinpath(self.collection).glob('*.tdms'))
        vdaq, fs = read_daq_voltage(daq_file, chmap=DAQ_CHMAP)
        ts, fcn_daq2_, drift_ppm = sync_photometry_to_daq(
            vdaq=vdaq, fs=fs, df_photometry=fp_data, v_threshold=V_THRESHOLD)
        gc_bpod, _ = GoCueTriggerTimes(session_path=self.session_path).extract(task_collection='raw_behavior_data', save=False)
        gc_daq = rises(vdaq['bpod'])

        fcn_daq2_bpod, drift_ppm, idaq, ibp = sync_timestamps(
            rises(vdaq['bpod']) / fs, gc_bpod, return_indices=True)
        assert drift_ppm < 100, f"Drift between bpod and daq is above 100 ppm: {drift_ppm}"
        assert (gc_daq.size - idaq.size) < 5, "Bpod and daq synchronisation failed as too few" \
                                              "events could be matched"
        ts = fcn_daq2_bpod(ts)
        return ts

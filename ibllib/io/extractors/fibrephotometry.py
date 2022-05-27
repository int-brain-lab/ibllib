"""Data extraction from fibrephotometry DAQ files.

Below is the expected folder structure for a fibrephotometry session:

    subject/
    ├─ 2021-06-30/
    │  ├─ 001/
    │  │  ├─ alf/
    │  │  │  ├─ raw_photometry_data/
    │  │  │  │  ├─ fpData.raw.csv
    │  │  │  │  ├─ fpData.channels.csv

fpData.raw.csv is a copy of the 'FPdata' file, the output of the Neuophotometrics Bonsai workflow.
fpData.channels.csv is table of frame flags for deciphering LED and GPIO states. The default table,
copied from the Neurophotometrics manual can be found in iblscripts/deploy/fppc/
"""
import logging

import pandas as pd
import numpy as np

import one.alf.io as alfio
from ibllib.io.extractors.base import BaseExtractor

_logger = logging.getLogger(__name__)


"""Available LEDs on the Neurophotometrics FP3002"""
LIGHT_SOURCE_MAP = {
    0: 'None',
    415: 'Violet',
    470: 'Blue',
    560: 'Green'
}


class FibrePhotometry(BaseExtractor):
    save_names = ('photometry.signal.npy', 'photometry.photometryLightSource.npy',
                  'photometryLightSource.properties.tsv', 'photometry.times.npy')
    var_names = ('signal', 'lightSource', 'lightSource_properties', 'timestamps')

    def __init__(self, *args, **kwargs):
        """An extractor for all widefield data"""
        super().__init__(*args, **kwargs)

    def _channel_meta(self, light_source_map=None):
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


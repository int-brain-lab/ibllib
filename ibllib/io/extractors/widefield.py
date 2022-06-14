"""Data extraction from widefield binary file"""

import logging
import numpy as np
import shutil
from pathlib import Path
import pandas as pd

import neurodsp as dsp
import ibllib.exceptions as err
import ibllib.io.extractors.base as extractors_base
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync, get_sync_fronts, data_for_keys, FpgaTrials
from ibllib.io.video import get_video_meta

import wfield.cli as wfield_cli
from labcams.io import parse_cam_log
import one.alf.io as alfio
import spikeglx


_logger = logging.getLogger('ibllib')

"""Available LEDs for Widefield Imaging"""
LIGHT_SOURCE_MAP = {
    0: 'None',
    405: 'Violet',
    470: 'Blue',
}

DEFAULT_WIRING_MAP = {
    5: 470,
    6: 405
}

CHMAPS = {'nidq':
                {'left_camera': 0,
                 'right_camera': 1,
                 'body_camera': 2,
                 'frame_trigger': 3,
                 'frame2ttl': 4,
                 'rotary_encoder_0': 5,
                 'rotary_encoder_1': 6,
                 'audio': 7,
                 'bpod': 16}
          }


def load_channel_map(session_path):
    """
    Gets default channel map for the version/binary file type combination
    :param ef: ibllib.io.spikeglx.glob_ephys_file dictionary with field 'ap' or 'nidq'
    :return: channel map dictionary
    """

    default_chmap = CHMAPS['nidq']

    # Try to load channel map from file
    chmap = spikeglx.get_sync_map(session_path.joinpath('raw_widefield_data'))
    # If chmap provided but not with all keys, fill up with default values
    if not chmap:
        return default_chmap
    else:
        if data_for_keys(default_chmap.keys(), chmap):
            return chmap
        else:
            _logger.warning("Keys missing from provided channel map, "
                            "setting missing keys from default channel map")
            return {**default_chmap, **chmap}


def load_sync(session_path):
    # TODO should this also extract?
    sync = alfio.load_object(session_path.joinpath('raw_widefield_data'), 'sync', namespace='spikeglx', short_keys=True)

    return sync


def get_sync_and_chan_map(session_path):
    sync = load_sync(session_path)
    chmap = load_channel_map(session_path)

    return sync, chmap


def widefield_extract_all(session_path, save=True):
    """
    For the IBL ephys task, reads ephys binary file and extract:
        -   sync
        -   wheel
        -   behaviour
        -   video time stamps
    :param session_path: '/path/to/subject/yyyy-mm-dd/001'
    :param save: Bool, defaults to False
    :return: outputs, files
    """
    extractor_type = extractors_base.get_session_extractor_type(session_path)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    sync, chmap = get_sync_and_chan_map(session_path)
    base = [FpgaTrials]
    outputs, files = extractors_base.run_extractor_classes(
        base, session_path=session_path, save=save, sync=sync, chmap=chmap)
    return outputs, files


class Widefield(extractors_base.BaseExtractor):
    save_names = (None, None, None, 'widefieldChannels.frameAverage.npy', 'widefieldU.images.npy', 'widefieldSVT.uncorrected.npy',
                  None, None, 'widefieldSVT.haemoCorrected.npy', 'widefield.times.npy', 'widefield.widefieldLightSource.npy',
                  'widefieldLightSource.properties.csv')
    raw_names = ('motioncorrect_2_540_640_uint16.bin', 'motion_correction_shifts.npy', 'motion_correction_rotation.npy',
                 'frames_average.npy', 'U.npy', 'SVT.npy', 'rcoeffs.npy', 'T.npy', 'SVTcorr.npy', 'timestamps.npy', 'led.npy',
                 'led_properties.csv')
    var_names = ()

    def __init__(self, *args, **kwargs):
        """An extractor for all widefield data"""
        super().__init__(*args, **kwargs)
        self.data_path = self.session_path.joinpath('raw_widefield_data')

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

    def _channel_wiring(self):
        try:
            wiring = pd.read_csv(self.data_path.joinpath('widefieldChannels.wiring.csv'))
        except FileNotFoundError:
            _logger.warning('LED wiring map not found, using default')
            wiring = pd.DataFrame(DEFAULT_WIRING_MAP.items(), columns=('LED', 'wavelength'))

        return wiring

    def _extract(self, extract_timestamps=True, save=False, **kwargs):
        """
        NB: kwargs should be loaded from meta file
        Parameters
        ----------
        n_channels
        dtype
        shape
        kwargs

        Returns
        -------

        """
        self.preprocess(**kwargs)
        if extract_timestamps:
            _ = self.sync_timestamps(save=save)

        return None

    def _save(self, data=None, path_out=None):

        if not path_out:
            path_out = self.session_path.joinpath(self.default_path)
        path_out.mkdir(exist_ok=True, parents=True)

        new_files = []
        if not self.data_path.exists():
            _logger.warning(f'Path does not exist: {self.data_path}')
            return new_files

        for before, after in zip(self.raw_names, self.save_names):
            if after is None:
                continue
            else:
                try:
                    file_orig = next(self.data_path.glob(before))
                    file_new = path_out.joinpath(after)
                    shutil.move(file_orig, file_new)
                    new_files.append(file_new)
                except StopIteration:
                    _logger.warning(f'File not found: {before}')

        return new_files

    def preprocess(self, fs=30, functional_channel=0, nbaseline_frames=30, k=200):

        # MOTION CORRECTION
        wfield_cli._motion(str(self.data_path))
        # COMPUTE AVERAGE FOR BASELINE
        wfield_cli._baseline(str(self.data_path), nbaseline_frames)
        # DATA REDUCTION
        wfield_cli._decompose(str(self.data_path), k=k)
        # HAEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = wfield_cli.load_stack(str(self.data_path))
        if dat.shape[1] == 2:
            del dat
            wfield_cli._hemocorrect(str(self.data_path), fs=fs, functional_channel=functional_channel)

    def remove_files(self, file_prefix='motion'):
        motion_files = self.data_path.glob(f'{file_prefix}*')
        for file in motion_files:
            _logger.info(f'Removing {file}')
            file.unlink()

    def sync_timestamps(self, bin_exists=False, save=False, save_paths=None, **kwargs):

        # TODO move this to personal project and put in the actual sync here
        if save and save_paths:
            assert len(save_paths) == 3, 'Must provide save_path as list with 3 paths'
            Path(save_paths[0]).parent.mkdir(parents=True, exist_ok=True)
            Path(save_paths[1]).parent.mkdir(parents=True, exist_ok=True)
            Path(save_paths[2]).parent.mkdir(parents=True, exist_ok=True)

        filepath = next(self.data_path.glob('*.camlog'))

        fpga_sync, chmap = get_sync_and_chan_map(self.session_path)
        fpga_led = get_sync_fronts(fpga_sync, chmap['frame_trigger'])
        logdata, led, sync, ncomm = parse_cam_log(filepath, readTeensy=True)

        # Should we allow this?
        # Case where led greater than video frames


        # We can have more

        # Check that the no. of syncs from bpod and teensy match
        assert len(led) == len(fpga_led), 'Number of detected frames on fpga and camlog do not match'

        # if they are not the same length what do we do, do we extrapolate the times that we don't have?

        # convert to seconds
        fcn, drift, iteensy, ifpga = dsp.utils.sync_timestamps(led.timestamp.values / 1e3, fpga_led['times'], return_indices=True)

        _logger.debug(f'Widefield-FPGA clock drift: {drift} ppm')
        assert led.frame.is_monotonic_increasing
        video_path = next(self.data_path.glob('widefield.raw*.mov'))
        video_meta = get_video_meta(video_path)

        diff = len(led) - video_meta.length
        if diff < 0:
            raise ValueError('More frames than timestamps detected')
        if diff > 2:
            raise ValueError('Timestamps and frames differ by more than 2')

        led = led[0:video_meta.length]

        widefield_times = fcn(led.timestamp.values / 1e3)


        # Find led times that are outside of the sync pulses
        # led_times = np.copy(led.timestamp.values)
        # pre_times = led_times < np.min(sync.timestamp)
        # post_times = led_times > np.max(sync.timestamp)
        # led_times[pre_times] = np.nan
        # led_times[post_times] = np.nan
        #
        # # Interpolate frames that lie within sync pulses timeframe
        # widefield_times = fcn(led_times / 1e3)
        # kp = ~np.isnan(widefield_times)
        # # Extrapolate times that lie outside sync pulses timeframe (i.e before or after)
        # pol = np.polyfit(led_times[kp] / 1e3, widefield_times[kp], 1)
        # extrap_vals = np.polyval(pol, led.timestamp.values / 1e3)
        # widefield_times[~kp] = extrap_vals[~kp]

        assert np.all(np.diff(widefield_times) > 0)

        # Now extract the LED channels and meta data
        # Load channel meta and wiring map
        channel_meta_map = self._channel_meta(kwargs.get('light_source_map'))
        channel_wiring = self._channel_wiring()
        channel_id = np.empty_like(led.led.values)

        for _, d in channel_wiring.iterrows():
            mask = led.led.values == d['LED']
            if np.sum(mask) == 0:
                raise err.WidefieldWiringException
            channel_id[mask] = channel_meta_map.get(channel_meta_map['wavelength'] == d['wavelength']).index[0]

        if save:
            save_time = save_paths[0] if save_paths else self.data_path.joinpath('timestamps.npy')
            save_led = save_paths[1] if save_paths else self.data_path.joinpath('led.npy')
            save_meta = save_paths[2] if save_paths else self.data_path.joinpath('led_properties.csv')
            save_paths = [save_time, save_led, save_meta]
            np.save(save_time, widefield_times)
            np.save(save_led, channel_id)
            channel_meta_map.to_csv(save_meta)

            return save_paths
        else:
            return widefield_times, channel_id, channel_meta_map

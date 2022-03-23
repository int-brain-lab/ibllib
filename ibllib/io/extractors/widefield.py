"""Data extraction from widefield binary file"""
from collections import OrderedDict
import logging
from pathlib import Path, PureWindowsPath
import uuid

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import parse_version
# from wfield.decomposition import approximate_svd
# from wfield.plots import plot_summary_motion_correction
# from wfield.registration import motion_correct
import wfield.cli as wfield_cli

from wfield import decomposition, plots, registration, utils, io as wfield_io
from labcams.io import parse_cam_log

import one.alf.io as alfio
from iblutil.util import Bunch
import ibllib.dsp as dsp
import ibllib.exceptions as err
from ibllib.io.raw_data_loaders import load_widefield_mmap
from ibllib.io.extractors import biased_trials, training_trials
from ibllib.io.extractors.base import BaseExtractor
from ibllib.dsp.utils import sync_timestamps
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync, get_sync_fronts
from ibllib.io.video import get_video_meta

_logger = logging.getLogger('ibllib')
FILENAME_MAP = {
    'motioncorrect_*.bin': 'widefield.raw_motionCorrected.bin',
    'motion_correction_shifts.npy': 'widefield.motionCorrentionShifts.npy',
    'motion_correction_rotation.npy': 'widefield.motionCorrentionRotation.npy',
    # 'frames_average.npy': 'frameAverage.widefield.npy',
    'U.npy': 'widefieldU.images.npy',
    'SVT.npy': 'widefieldSVT.uncorrected.npy',
    # 'rcoeffs.npy': '',
    # 'T.npy': 'widefieldT.uncorrected.npy',
    'SVTcorr.npy': 'widefieldSVT.haemoCorrected.npy'
}


class Widefield(BaseExtractor):
    save_names = ('_ibl_trials.feedbackType.npy',)
    raw_names = ('motioncorrect_2_540_640_uint16.bin', 'motion_correction_shifts.npy', 'motion_correction_rotation.npy',
                 'frames_average.npy', 'U.npy', 'SVT.npy', 'rcoeffs.npy', 'T.npy', 'SVTcorr.npy')
    var_names = ('feedbackType',)

    def __init__(self, *args, **kwargs):
        """An extractor for all widefield data"""
        super().__init__(*args, **kwargs)

    def _extract(self, **kwargs):
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
        ##########################################################
        # dat = load_widefield_mmap(self.session_path, dtype=dtype, shape=shape, mode='r+')

        # return [out[k] for k in out] + [wheel['timestamps'], wheel['position'],
        #                                 moves['intervals'], moves['peakAmplitude']]

    def _symlink_files(self):
        ...

    def preprocess(self, fs=30, functional_channel=0, nbaseline_frames=30, k=200):
        data_path = self.session_path.joinpath('raw_widefield_data')

        # MOTION CORRECTION
        wfield_cli._motion(str(data_path))
        # COMPUTE AVERAGE FOR BASELINE
        wfield_cli._baseline(str(data_path), nbaseline_frames)
        # DATA REDUCTION
        wfield_cli._decompose(str(data_path), k=k)
        # HAEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = wfield_cli.load_stack(str(data_path))
        if dat.shape[1] == 2:
            del dat
            wfield_cli._hemocorrect(str(data_path), fs=fs, functional_channel=functional_channel)

    def rename_files(self, session_folder) -> bool:
        """
        Rename the raw widefield data for a given session.

        Parameters
        ----------
        session_folder : str, pathlib.Path
            A session path containing widefield data.

        Returns
        -------
        success : bool
            True if all files were successfully renamed.
        TODO Double-check filenames and call this function
        """
        session_path = Path(session_folder).joinpath('raw_widefield_data')
        if not session_path.exists():
            _logger.warning(f'Path does not exist: {session_path}')
            return False
        renames = (
            ('dorsal_cortex_landmarks.json', 'widefieldLandmarks.dorsalCortex.json'),
            ('*.dat', 'widefield.raw.dat'),
            ('*.camlog', 'widefieldEvents.raw.camlog')
        )
        success = True
        for before, after in renames:
            try:
                filename = next(session_path.glob(before))
                filename.rename(after)
                # TODO Save nchannels and frame size from filename?
            except StopIteration:
                _logger.warning(f'File not found: {before}')
                success = False
        return success

    def sync_timestamps(self, bin_exists=False):
        filepath = next(self.session_path.joinpath('raw_widefield_data').glob('*.camlog'))
        fpga_sync, chmap = get_main_probe_sync(self.session_path, bin_exists=bin_exists)
        bpod = get_sync_fronts(fpga_sync, chmap['bpod'])
        logdata, led, sync, ncomm = parse_cam_log(filepath, readTeensy=True)
        if bpod.times.size == 0:
            raise err.SyncBpodFpgaException('No Bpod event found in FPGA. No behaviour extraction. '
                                            'Check channel maps.')

        fcn, drift, iteensy, ifpga = dsp.utils.sync_timestamps(sync.timestamp.values, bpod['times'], return_indices=True)
        _logger.debug(f'Widefield-FPGA clock drift: {drift} ppm')
        assert led.frame.is_monotonic_increasing  # FIXME Need to check whether to use logdata instead
        video_path = next(self.session_path.joinpath('raw_widefield_data').glob('widefield.raw*.mov'))
        video_meta = get_video_meta(video_path)
        assert video_meta.length == len(led)  # FIXME This fails
        widefield_times = fcn(led.timestamp.values)
        return widefield_times  # FIXME Need to sort frame mismatch
    # TODO Add QC check for

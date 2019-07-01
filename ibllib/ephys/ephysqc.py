"""
Quality control of Neuropixel electrophysiology data.
"""
from pathlib import Path
import logging

import numpy as np
from scipy import signal

from ibllib.io import spikeglx
import ibllib.dsp as dsp
from ibllib.misc import print_progress

logger_ = logging.getLogger('ibllib')

RMS_WIN_LENGTH_SECS = 3
WELCH_WIN_LENGTH_SAMPLES = 1024


def rmsmap(fbin):
    """
    Computes RMS map in time domain and spectra for each channel of Neuropixel probe

    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :type fbin: str or pathlib.Path
    :return: a dictionary with amplitudes in channeltime space, channelfrequency space, time
     and frequency scales
    """
    if not isinstance(fbin, spikeglx.Reader):
        sglx = spikeglx.Reader(fbin)
    rms_win_length_samples = 2 ** np.ceil(np.log2(sglx.fs * RMS_WIN_LENGTH_SECS))
    # the window generator will generates window indices
    wingen = dsp.WindowGenerator(ns=sglx.ns, nswin=rms_win_length_samples, overlap=0)
    # pre-allocate output dictionary of numpy arrays
    win = {'TRMS': np.zeros((wingen.nwin, sglx.nc)),
           'nsamples': np.zeros((wingen.nwin,)),
           'fscale': dsp.fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / sglx.fs, one_sided=True),
           'tscale': wingen.tscale(fs=sglx.fs)}
    win['spectral_density'] = np.zeros((len(win['fscale']), sglx.nc))
    # loop through the whole session
    for first, last in wingen.firstlast:
        D = sglx.read_samples(first_sample=first, last_sample=last)[0].transpose()
        # remove low frequency noise below 1 Hz
        D = dsp.hp(D, 1 / sglx.fs, [0, 1])
        iw = wingen.iw
        win['TRMS'][iw, :] = dsp.rms(D)
        win['nsamples'][iw] = D.shape[1]
        # the last window may be smaller than what is needed for welch
        if last - first < WELCH_WIN_LENGTH_SAMPLES:
            continue
        # compute a smoothed spectrum using welch method
        _, w = signal.welch(D, fs=sglx.fs, window='hanning', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                            detrend='constant', return_onesided=True, scaling='density', axis=-1)
        win['spectral_density'] += w.T
        # print at least every 20 windows
        if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
            print_progress(iw, wingen.nwin)
    return win


def extract_rmsmap(fbin, out_folder=None, force=False):
    """
    Wrapper for rmsmap that outputs _ibl_ephysRmsMap and _ibl_ephysSpectra ALF files

    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :param folder_alf: folder in which to store output ALF files. Creates/Uses an ALF folder at
     the same level as the `fbin` file provided by default
    :param force: do not re-extract if all ALF files already exist
    :return: None
    """
    logger_.info(str(fbin))
    sglx = spikeglx.Reader(fbin)
    # check if output ALF files exist already:
    if out_folder is None:
        out_folder = Path(fbin).parent / ('qc_ephys_' + Path(fbin).name.split('.')[0])
    else:
        out_folder = Path(out_folder)

    files = {'rms': out_folder / ('_ibl_ephysRmsMap_' + sglx.type + '.rms..npy'),
             'times': out_folder / ('_ibl_ephysRmsMap_' + sglx.type + '.times.npy'),
             'power': out_folder / ('_ibl_ephysSpectra_' + sglx.type + '.power.npy'),
             'frequencies': out_folder / ('_ibl_ephysSpectra_' + sglx.type + '.frequencies.npy')}
    # if they do and the option Force is set to false, do not recompute and exit
    if all([files[f].exists() for f in files]) and not force:
        logger_.warning('Output exists. Skipping ' + str(fbin) + ' Use force option to override')
        return
    # crunch numbers
    rms = rmsmap(fbin)
    # output ALF files, single precision
    if not out_folder.exists():
        out_folder.mkdir()
    np.save(file=files['rms'], arr=rms['TRMS'].astype(np.single))
    np.save(file=files['times'], arr=rms['tscale'].astype(np.single))
    np.save(file=files['power'], arr=rms['spectral_density'].astype(np.single))
    np.save(file=files['frequencies'], arr=rms['fscale'].astype(np.single))

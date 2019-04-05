from pathlib import Path

import numpy as np
from scipy import signal

from ibllib.io import spikeglx
import ibllib.dsp as dsp
from ibllib.misc import print_progress


RMS_WIN_LENGTH_SECS = 3
WELCH_WIN_LENGTH_SAMPLES = 1024


def rmsmap(fbin):
    """
    Computes RMS map in time domain and spectra for each channel of Neuropixel probe
    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :return: win: a dictionary with a time-RMS per channel map, spectra per channel, time
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
    for first, last in wingen.slices:
        D = sglx.read_samples(first_sample=first, last_sample=last).transpose()
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
        if (iw % 20) == 0:
            print_progress(iw, wingen.nwin)
    return win


def extract_rmsmap(fbin, folder_alf=None):
    """
    wrapper for rmsmap that outputs _ibl_ephysRmsMap and _ibl_ephysSpectra ALF files
    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :return: None
    """
    print(str(fbin))
    sglx = spikeglx.Reader(fbin)
    rms = rmsmap(fbin)

    if folder_alf is None:
        folder_alf = Path(fbin).parent / 'alf'
    if not folder_alf.exists():
        folder_alf.mkdir()

    np.save(file=folder_alf / ('_ibl_ephysRmsMap.' + sglx.type + '.rms.npy'),
            arr=rms['TRMS'].astype(np.single))
    np.save(file=folder_alf / ('_ibl_ephysRmsMap.' + sglx.type + '.times.npy'),
            arr=rms['tscale'].astype(np.single))
    np.save(file=folder_alf / ('_ibl_ephysSpectra' + sglx.type + '.power.npy'),
            arr=rms['TRMS'].astype(np.single))
    np.save(folder_alf / ('_ibl_ephysSpectra' + sglx.type + '.frequencies.npy'),
            arr=rms['TRMS'].astype(np.single))
    return

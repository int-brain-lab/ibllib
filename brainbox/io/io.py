import time
import numpy as np
# (Previously required `os.path` to get file info before memmapping)
# import os.path as op
from ibllib.io import spikeglx


def extract_waveforms(ephys_file, ts, ch, t=2.0, sr=30000, n_ch_probe=385, dtype='int16',
                      offset=0, car=True):
    '''
    Extracts spike waveforms from binary ephys data file, after (optionally)
    common-average-referencing (CAR) spatial noise.

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    ts : ndarray_like
        The timestamps (in s) of the spikes.
    ch : ndarray_like
        The channels on which to extract the waveforms.
    t : numeric (optional)
        The time (in ms) of each returned waveform.
    sr : int (optional)
        The sampling rate (in hz) that the ephys data was acquired at.
    n_ch_probe : int (optional)
        The number of channels of the recording.
    dtype: str (optional)
        The datatype represented by the bytes in `ephys_file`.
    offset: int (optional)
        The offset (in bytes) from the start of `ephys_file`.
    car: bool (optional)
        A flag to perform CAR before extracting waveforms.

    Returns
    -------
    waveforms : ndarray
        An array of shape (#spikes, #samples, #channels) containing the waveforms.

    Examples
    --------
    1) Extract all the waveforms for unit1 with and without CAR.
        >>> import numpy as np
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        (*Note, if there is no 'alf' directory, make 'alf' directory from 'ks2' output directory):
        >>> e_spks.ks2_to_alf(path_to_ks_out, path_to_alf_out)
        # Get a clusters bunch and a units bunch from a spikes bunch from an alf directory.
        >>> clstrs_b = aio.load_object(path_to_alf_out, 'clusters')
        >>> spks_b = aio.load_object(path_to_alf_out, 'spikes')
        >>> units_b = bb.processing.get_units_bunch(spks, ['times'])
        # Get the timestamps and 20 channels around the max amp channel for unit1, and extract the
        # two sets of waveforms.
        >>> ts = units_b['times']['1']
        >>> max_ch = max_ch = clstrs_b['channels'][1]
        >>> if max_ch < 10:  # take only channels greater than `max_ch`.
        >>>     ch = np.arange(max_ch, max_ch + 20)
        >>> elif (max_ch + 10) > 385:  # take only channels less than `max_ch`.
        >>>     ch = np.arange(max_ch - 20, max_ch)
        >>> else:  # take `n_c_ch` around `max_ch`.
        >>>     ch = np.arange(max_ch - 10, max_ch + 10)
        >>> wf = bb.io.extract_waveforms(path_to_ephys_file, ts, ch, car=False)
        >>> wf_car = bb.io.extract_waveforms(path_to_ephys_file, ts, ch, car=True)
    '''

    # (Previously memmaped the file manually, but now use `spikeglx.Reader`)
    # item_bytes = np.dtype(dtype).itemsize
    # n_samples = (op.getsize(ephys_file) - offset) // (item_bytes * n_ch_probe)
    # file_m = np.memmap(ephys_file, shape=(n_samples, n_ch_probe), dtype=dtype, mode='r')

    # Get memmapped array of `ephys_file`
    s_reader = spikeglx.Reader(ephys_file)
    file_m = s_reader.data  # the memmapped array
    n_wf_samples = np.int(sr / 1000 * (t / 2))  # number of samples to return on each side of a ts
    ts_samples = np.array(ts * sr).astype(int)  # the samples corresponding to `ts`
    t_sample_first = ts_samples[0] - n_wf_samples

    # Exception handling for impossible channels
    ch = np.asarray(ch)
    ch = ch.reshape((ch.size, 1)) if ch.size == 1 else ch
    if np.any(ch < 0) or np.any(ch > n_ch_probe):
        raise Exception('At least one specified channel number is impossible. The minimum channel'
                        ' number was {}, and the maximum channel number was {}. Check specified'
                        ' channel numbers and try again.'.format(np.min(ch), np.max(ch)))

    if car:  # compute spatial noise in chunks
        # see https://github.com/int-brain-lab/iblenv/issues/5
        raise NotImplementedError("CAR option is not available")

    # Initialize `waveforms`, extract waveforms from `file_m`, and CAR.
    waveforms = np.zeros((len(ts), 2 * n_wf_samples, ch.size))
    # Give time estimate for extracting waveforms.
    t0 = time.perf_counter()
    for i in range(5):
        waveforms[i, :, :] = \
            file_m[i * n_wf_samples * 2 + t_sample_first:
                   i * n_wf_samples * 2 + t_sample_first + n_wf_samples * 2, ch].reshape(
                       (n_wf_samples * 2, ch.size))
    dt = time.perf_counter() - t0
    print('Performing waveform extraction. Estimated time is {:.2f} mins. ({})'
          .format(dt * len(ts) / 60 / 5, time.ctime()))
    for spk, _ in enumerate(ts):  # extract waveforms
        spk_ts_sample = ts_samples[spk]
        spk_samples = np.arange(spk_ts_sample - n_wf_samples, spk_ts_sample + n_wf_samples)
        # have to reshape to add an axis to broadcast `file_m` into `waveforms`
        waveforms[spk, :, :] = \
            file_m[spk_samples[0]:spk_samples[-1] + 1, ch].reshape((spk_samples.size, ch.size))
    print('Done. ({})'.format(time.ctime()))

    return waveforms

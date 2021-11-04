import shutil
import logging
from pathlib import Path
import time

import numpy as np

from ibllib.io import spikeglx

_logger = logging.getLogger('ibllib')


def extract_waveforms(ephys_file, ts, ch, t=2.0, sr=30000, n_ch_probe=385, car=True):
    """
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
    """

    # Get memmapped array of `ephys_file`
    with spikeglx.Reader(ephys_file) as s_reader:
        file_m = s_reader.data  # the memmapped array
        n_wf_samples = int(sr / 1000 * (t / 2))  # number of samples to return on each side of a ts
        ts_samples = np.array(ts * sr).astype(int)  # the samples corresponding to `ts`
        t_sample_first = ts_samples[0] - n_wf_samples

        # Exception handling for impossible channels
        ch = np.asarray(ch)
        ch = ch.reshape((ch.size, 1)) if ch.size == 1 else ch
        if np.any(ch < 0) or np.any(ch > n_ch_probe):
            raise Exception('At least one specified channel number is impossible. '
                            f'The minimum channel number was {np.min(ch)}, '
                            f'and the maximum channel number was {np.max(ch)}. '
                            'Check specified channel numbers and try again.')

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


def stream(pid, t0, nsecs=1, one=None, cache_folder=None, remove_cached=False, typ='ap'):
    """
    NB: returned Reader object must be closed after use
    :param pid: Probe UUID
    :param t0: time of the first sample
    :param nsecs: duration of the streamed data
    :param one: An instance of ONE
    :param cache_folder:
    :param typ: 'ap' or 'lf'
    :return: sr, dsets, t0
    """
    CHUNK_DURATION_SECS = 1  # the mtscomp chunk duration. Right now it's a constant
    if nsecs > 10:
        ValueError(f'Streamer works only with 10 or less seconds, set nsecs to lesss than {nsecs}')
    assert one
    assert typ in ['lf', 'ap']
    t0 = np.floor(t0 / CHUNK_DURATION_SECS) * CHUNK_DURATION_SECS
    if cache_folder is None:
        samples_folder = Path(one.alyx._par.CACHE_DIR).joinpath('cache', typ)

    eid, pname = one.pid2eid(pid)
    cbin_rec = one.list_datasets(eid, collection=f"*{pname}", filename=f'*{typ}.*bin', details=True)
    ch_rec = one.list_datasets(eid, collection=f"*{pname}", filename=f'*{typ}.ch', details=True)
    meta_rec = one.list_datasets(eid, collection=f"*{pname}", filename=f'*{typ}.meta', details=True)
    ch_file = one._download_datasets(ch_rec)[0]
    one._download_datasets(meta_rec)[0]

    first_chunk = int(t0 / CHUNK_DURATION_SECS)
    last_chunk = int((t0 + nsecs) / CHUNK_DURATION_SECS) - 1

    samples_folder.mkdir(exist_ok=True, parents=True)
    sr = spikeglx.download_raw_partial(
        one=one,
        url_cbin=one.record2url(cbin_rec)[0],
        url_ch=ch_file,
        first_chunk=first_chunk,
        last_chunk=last_chunk,
        cache_dir=samples_folder)

    if remove_cached:
        probe_cache = samples_folder.joinpath(one.eid2path(eid).relative_to(one.alyx._par.CACHE_DIR),
                                              'raw_ephys_data', f'{pname}')
        shutil.rmtree(probe_cache, ignore_errors=True)

    return sr, t0

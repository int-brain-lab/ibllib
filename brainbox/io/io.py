import numpy as np
import os.path as op


def extract_waveforms(ephys_file, ts, ch, t=2.0, sr=30000, n_ch_probe=385, dtype='int16', offset=0, \
                      car=True):
    '''
    Extracts spike waveforms from binary ephys data file, after (optionally) 
    common-average-referencing.
    
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
    n_ch : int (optional)
        The number of channels of the recording.
    dtype: str (optional)
        The datatype represented by the bytes in `ephys_file`.
    offset: int (optional)
        The offset (in bytes) from the start of `ephys_file`.
    car: bool (optional)
        A flag for whether or not to perform common-average-referencing before extracting waveforms

    Returns
    -------
    waveforms : ndarray 
        An array of shape (#spikes, #samples, #channels) containing the waveforms.
    
    Examples
    --------
    '''
    
    # Get memmapped array of `ephys_file`
    item_bytes = np.dtype(dtype).itemsize
    n_samples = (op.getsize(ephys_file) - offset) // (item_bytes * n_ch_probe)  
    file_m = np.memmap(ephys_file, shape=(n_samples, n_ch_probe), dtype=dtype, mode='r')
    n_wf_samples = np.int(sr/1000/(t/2))  # number of samples to return on each side of each ts
    ts_samples = np.array(ts*sr).astype(int)  # the samples corresponding to `ts`
    
    if car:  # compute temporal and spatial noise
        t_sample_first = ts_samples[0] - n_wf_samples
        t_sample_last = ts_samples[-1] + n_wf_samples
        noise_t = np.median(file_m[np.ix_(np.arange(t_sample_first, t_sample_last), ch)], axis=1)
        noise_s = np.median(file_m[np.ix_(np.arange(t_sample_first, t_sample_last), ch)], axis=0)
    
    # Initialize `waveforms` and then extract waveforms from `file_m`.
    waveforms = np.zeros((len(ts), 2*n_wf_samples, len(ch)))
    for spk in range(len(ts)):
        spk_ts_sample = ts_samples[spk]
        spk_samples = np.arange(spk_ts_sample - n_wf_samples, spk_ts_sample + n_wf_samples)
        waveforms[spk, :, :] = file_m[np.ix_(spk_samples, ch)]
        if car:  # subtract temporal noise
            waveforms[spk, :, :]-=noise_t[spk_samples - t_sample_first, None]
    if car:  # subtract spatial noise  
        waveforms-=noise_s[None, None, :]
    
    return waveforms
        
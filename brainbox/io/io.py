import scipy.io as sio
import scipy.signal as sp
import numpy as np
import os.path as op
import phylib.traces.waveform as phy_wave
import phylib.io.model as phy_model
import alf.io as aio
import brainbox as bb

def extract_waveforms(ephys_file, ts, ch, t=2.0, sr=30000, n_ch=385, dtype='int16', offset=0):
    '''
    Extracts spike waveforms from binary ephys data file.
    
    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data. 
    ts : ndarray_like 
        The timestamps (in s) of the spikes.
    ch : ndarray_like
        The channels on which to extract the waveforms.
    t : numeric 
        The time (in ms) of each returned waveform.
    sr : int
        The sampling rate (in hz) that the ephys data was acquired at.
    n_ch : int
        The number of channels of the recording.
    dtype: str
        The datatype represented by the bytes in `ephys_file`.
    offset: int
        The offset (in bytes) from the start of `ephys_file`.
    -------
    waveforms : ndarray 
        An array of shape (#spikes, #samples, #channels) containing the waveforms.
    
    Examples
    --------
    '''
    
    # Get memmapped array of `ephys_file`
    item_bytes = np.dtype(dtype).itemsize
    n_samples = (op.getsize(ephys_file) - offset) // (item_bytes * n_ch)  
    file_m = np.memmap(str(ephys_file), shape=(n_samples, n_ch), dtype=dtype, mode='r')
    wf_samples = np.int(sr/1000/(t/2))  # number of samples to return on each side of each ts
    ts_samples = np.array(ts*sr).astype(int)  # the samples corresponding to `ts`
    
    # Initialize `waveforms` and then extract waveforms from `file_m`
    waveforms = np.zeros((len(ts), 2*wf_samples, len(ch)))
    for spk in range(len(ts)):
        s = ts_samples[spk]
        s = np.arange(s-wf_samples,s+wf_samples)
        waveforms[spk, :, :] = file_m[np.ix_(s, ch)]
    
    return waveforms
        
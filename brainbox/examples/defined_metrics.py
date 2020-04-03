# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:01:32 2020

@author: Steinmetz Lab User
"""



import time
from pathlib import Path
import numpy as np
import alf.io as aio
import matplotlib.pyplot as plt
from max_acceptable_isi_viol_2 import max_acceptable_cont_2
import brainbox as bb
from phylib.stats import correlograms
import pandas as pd

def FP_RP(ts):
    binSize=0.25 #in ms
    b= np.arange(0,10.25,binSize)/1000 + 1e-6 #bins in seconds
    bTestIdx = [5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
    bTest = [b[i] for i in bTestIdx]

    thresh = 0.2
    acceptThresh=0.1
    if(len(ts)>0 and ts[-1]>ts[0]):
        recDur = (ts[-1]-ts[0])
        fr = len(ts)/recDur
        # print(fr_source)
        mfunc =np.vectorize(max_acceptable_cont_2)
        m = mfunc(fr,bTest,recDur,fr*acceptThresh,thresh)
        c0 = correlograms(ts,np.zeros(len(ts),dtype='int8'),cluster_ids=[0],bin_size=binSize/1000,sample_rate=20000,window_size=.05,symmetrize=False)
        cumsumc0 = np.cumsum(c0[0,0,:])
        res = cumsumc0[bTestIdx]
        didpass = int(np.any(np.less_equal(res,m)))
        #OR
        didpass2 = didpass
        # if res(np.where(m==-1)[0])==0:
        #     didpass2 = 1
        # print(didpass[uidx])
    else: 
        didpass=0
        # didpass2 = 0

    return didpass
    
    
def noise_cutoff(amps,quartile_length=.25,std_cutoff = 2):
        nbins = 500
        bins_list= np.linspace(0, np.max(amps), nbins)
        n,bins,patches = plt.hist(amps,bins = bins_list ,facecolor = 'blue',alpha = 0.5)
        # plt.xticks(np.arange(10,40,5))
        # plt.yticks(np.arange(0,1000,200))
        # plt.xlim(10,37)
        # plt.ylim(0, 200)
        # plt.show()
               
        dx = np.diff(n)
        
        idx_nz = np.nonzero(dx) #indices of nonzeros
        length_nonzeros = idx_nz[0][-1]-idx_nz[0][0] #length of the entire stretch, from first nonzero to last nonzero
        high_quartile = 1-quartile_length
        high_quartile_start_ind = int(np.ceil(high_quartile*(length_nonzeros)))
        high_quartile_end_ind = idx_nz[0][-1]
        mean_high_quartile = np.mean(dx[high_quartile_start_ind:high_quartile_end_ind])
        std_high_quartile = np.std(dx[high_quartile_start_ind:high_quartile_end_ind])
        
        first_low_quartile = dx[idx_nz[0][0]]
        #statistical test? can ask whether this is within 2 std's of high quartile dx's
        within_2stds = first_low_quartile<mean_high_quartile + std_cutoff*std_high_quartile or first_low_quartile<mean_high_quartile - std_cutoff*std_high_quartile
        cutoff = 0 if within_2stds else 1
        return cutoff 
    
    
def peak_to_peak_amp(ephys_file, ts, ch, t=2.0, sr=30000, n_ch_probe=385, dtype='int16', offset=0,
                   car=True):
    
    
    # def ptp_over_noise(ephys_file, ts, ch, t=2.0, sr=30000, n_ch_probe=385, dtype='int16', offset=0,
    #                car=True):
    '''
    For specified channels, for specified timestamps, computes the mean (peak-to-peak amplitudes /
    the MADs of the background noise).

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    ts : ndarray_like
        The timestamps (in s) of the spikes.
    ch : ndarray_like
        The channels on which to extract the waveforms.
    t : numeric (optional)
        The time (in ms) of the waveforms to extract to compute the ptp.
    sr : int (optional)
        The sampling rate (in hz) that the ephys data was acquired at.
    n_ch_probe : int (optional)
        The number of channels of the recording.
    dtype: str (optional)
        The datatype represented by the bytes in `ephys_file`.
    offset: int (optional)
        The offset (in bytes) from the start of `ephys_file`.
    car: bool (optional)
        A flag to perform common-average-referencing before extracting waveforms.

    Returns
    -------
    ptp_sigma : ndarray
        An array containing the mean ptp_over_noise values for the specified `ts` and `ch`.

    Examples
    --------
    1) Compute ptp_over_noise for all spikes on 20 channels around the channel of max amplitude
    for unit 1.
        >>> ts = units_b['times']['1']
        >>> max_ch = max_ch = clstrs_b['channels'][1]
        >>> if max_ch < 10:  # take only channels greater than `max_ch`.
        >>>     ch = np.arange(max_ch, max_ch + 20)
        >>> elif (max_ch + 10) > 385:  # take only channels less than `max_ch`.
        >>>     ch = np.arange(max_ch - 20, max_ch)
        >>> else:  # take `n_c_ch` around `max_ch`.
        >>>     ch = np.arange(max_ch - 10, max_ch + 10)
        >>> p = bb.metrics.ptp_over_noise(ephys_file, ts, ch)
    '''

    # Ensure `ch` is ndarray
    ch = np.asarray(ch)
    ch = ch.reshape((ch.size, 1)) if ch.size == 1 else ch

    # Get waveforms.
    wf = bb.io.extract_waveforms(ephys_file, ts, ch, t=t, sr=sr, n_ch_probe=n_ch_probe,
                                 dtype=dtype, offset=offset, car=car)

    # Initialize `mean_ptp` based on `ch`, and compute mean ptp of all spikes for each ch.
    mean_ptp = np.zeros((ch.size,))
    for cur_ch in range(ch.size,):
        mean_ptp[cur_ch] = np.mean(np.max(wf[:, :, cur_ch], axis=1) -
                                   np.min(wf[:, :, cur_ch], axis=1))

    # # Compute MAD for `ch` in chunks.
    # s_reader = spikeglx.Reader(ephys_file)
    # file_m = s_reader.data  # the memmapped array
    # n_chunk_samples = 5e6  # number of samples per chunk
    # n_chunks = np.ceil(file_m.shape[0] / n_chunk_samples).astype('int')
    # # Get samples that make up each chunk. e.g. `chunk_sample[1] - chunk_sample[0]` are the
    # # samples that make up the first chunk.
    # chunk_sample = np.arange(0, file_m.shape[0], n_chunk_samples, dtype=int)
    # chunk_sample = np.append(chunk_sample, file_m.shape[0])
    # # Give time estimate for computing MAD.
    # t0 = time.perf_counter()
    # stats.median_absolute_deviation(file_m[chunk_sample[0]:chunk_sample[1], ch], axis=0)
    # dt = time.perf_counter() - t0
    # print('Performing MAD computation. Estimated time is {:.2f} mins.'
    #       ' ({})'.format(dt * n_chunks / 60, time.ctime()))
    # # Compute MAD for each chunk, then take the median MAD of all chunks.
    # mad_chunks = np.zeros((n_chunks, ch.size), dtype=np.int16)
    # for chunk in range(n_chunks):
    #     mad_chunks[chunk, :] = stats.median_absolute_deviation(
    #         file_m[chunk_sample[chunk]:chunk_sample[chunk + 1], ch], axis=0, scale=1)
    # print('Done. ({})'.format(time.ctime()))

    # # Return `mean_ptp` over `mad`
    # mad = np.median(mad_chunks, axis=0)
    # ptp_sigma = mean_ptp / mad
    ptp_amp = np.mean(mean_ptp)
    return ptp_sigma

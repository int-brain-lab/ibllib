# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:01:32 2020

@author: Noam Roth
"""



import time
from pathlib import Path
import numpy as np
import alf.io as aio
import matplotlib.pyplot as plt
from brainbox.examples.max_acceptable_isi_viol_2 import max_acceptable_cont_2
import brainbox as bb
from phylib.stats import correlograms
import pandas as pd
from ibllib.io import spikeglx

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
    
    
def noise_cutoff(amps,quartile_length=.25):
        nbins = 100
        end_low=5
        if(len(amps)>1):
            bins_list= np.linspace(0, np.max(amps), nbins)
            n,bins = np.histogram(amps,bins = bins_list) 
            dx = np.diff(n) 
            idx_nz = np.nonzero(dx) #indices of nonzeros
            length_nonzeros = idx_nz[0][-1]-idx_nz[0][0] #length of the entire stretch, from first nonzero to last nonzero
            high_quartile = 1-quartile_length
            # high_quartile_start_ind = int(np.ceil(high_quartile*(length_nonzeros)))+idx_nz[0][0]
            # high_quartile_end_ind = idx_nz[0][-1]
            # mean_high_quartile = np.mean(n[high_quartile_start_ind:high_quartile_end_ind])
            # std_high_quartile = np.std(n[high_quartile_start_ind:high_quartile_end_ind])
            
            idx_peak = np.argmax(n)
            length_top_half = idx_nz[0][-1]-idx_peak
            high_quartile = 1-(2*quartile_length)
            
            high_quartile_start_ind = int(np.ceil(high_quartile*length_top_half + idx_peak))
            xx=idx_nz[0][idx_nz[0]>high_quartile_start_ind]
            if len(n[xx])>0:
                mean_high_quartile = np.mean(n[xx])
                std_high_quartile = np.std(n[xx])
                            
                
                first_low_quartile = np.mean(n[idx_nz[0][1:end_low]])
                # within_2stds = first_low_quartile<mean_high_quartile + std_cutoff*std_high_quartile or first_low_quartile<mean_high_quartile - std_cutoff*std_high_quartile
                # cutoff = 0 if within_2stds else 1
                if std_high_quartile>0:
                    cutoff=(first_low_quartile-mean_high_quartile)/std_high_quartile
                else:
                    cutoff=np.float64(np.nan)
            else:
                cutoff=np.float64(np.nan)
        else:
            cutoff=np.float64(np.nan)
        return cutoff 
    
    
def peak_to_peak_amp(ephys_file, samp_inds, nsamps):
    
    #read raw ephys file
    sr = spikeglx.Reader(ephys_file)
    
    #take a subset (nsamps) of the spike samples
    samples = np.random.choice(samp_inds,nsamps)
    #initialize arrays
    amps=np.zeros(len(samples))
    wfs=np.zeros((384,len(samples)))
    wfs_baseline=np.zeros((384,len(samples)))
    cnt=0
    for i in samples:        
        wf = sr.data[int(i)]
        wf_baseline = wf[:-1]-np.median(wf[:-1]) #subtract median baseline
        # plt.plot(wf_baseline)
        wfs[:,cnt] = wf[:-1]
        wfs_baseline[:,cnt] = wf_baseline
        amps[cnt] = np.max(wf_baseline)-np.min(wf_baseline) 
        cnt+=1
    amps = np.max(wfs_baseline,axis=0)-np.min(wfs_baseline,axis=0)
    mean_amp = np.mean(amps)


    return mean_amp

# def ptp_snr(ephys_file,samp_inds,nsamps)

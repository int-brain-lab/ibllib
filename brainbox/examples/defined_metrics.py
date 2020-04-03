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
    
    
def peak_to_peak_amp():

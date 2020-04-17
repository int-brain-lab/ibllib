# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:47:20 2020

@author: Steinmetz Lab User
"""

#Plot example neurons for SAB report

import time
import os
from oneibl.one import ONE
from pathlib import Path
import numpy as np
import alf.io as aio
import matplotlib.pyplot as plt
from max_acceptable_isi_viol_2 import max_acceptable_cont_2
import brainbox as bb
from phylib.stats import correlograms
import pandas as pd
from defined_metrics import FP_RP, noise_cutoff, peak_to_peak_amp
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ks_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab\test_alf_dir1'
alf_probe_dir = ks_dir
spks_b = aio.load_object(alf_probe_dir, 'spikes')  
units_b = bb.processing.get_units_bunch(spks_b)
# units_bamps = bb.processing.get_units_bunch(amps_b)
units = list(units_b.clusters.keys())

n_units = np.max(spks_b.clusters) + 1

    
uidx=0
label=np.empty([len(units)])
RefPViol = np.empty([len(units)])
NoiseCutoff = np.empty([len(units)])
# for unit in units:
#     ts = units_b['times'][unit]
#     amps = units_b['amps'][unit]

#     RefPViol[int(unit)] = FP_RP(ts)
""" 
First example unit is number 171, from Hopkins. This one shows refractory
 period violations for a fixed 2ms window (but passes during the RP viol )
"""
unit = '171'
ts = units_b['times'][unit]
# amps = units_b['amps'][unit] #not working?!

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
    c0 = correlograms(ts,np.zeros(len(ts),dtype='int8'),cluster_ids=[0],\
                      bin_size=binSize/1000,sample_rate=20000,window_size=.02,symmetrize=True) 
    plt.bar(np.arange(-10,10.1,.25),c0[0,0],width = 0.25,alpha=1,color = '.5')
    plt.vlines(2, 0, 500,linewidth=.75)
    plt.vlines(-2, 0, 500,linewidth=.75)
    plt.xticks(np.arange(-10,11,2))
    plt.xlim((-10,10))
    plt.ylabel('Number of spikes')
    plt.xlabel('Time (ms)')
    plt.savefig("RP_example_171.pdf", transparent=True,dpi=288)
    plt.show()
    cumsumc0 = np.cumsum(c0[0,0,:])
    res = cumsumc0[bTestIdx]
    didpass = int(np.any(np.less_equal(res,m)))


"""
Second example unit is unit 0 from  WindowsPath('C:/Users/Steinmetz Lab User/Downloads/FlatIron/mainenlab/Subjects/ZM_2104/2019-09-19/001/alf/probe_right')
"""



plt.rcParams.update({'font.size': 22})
xaxes = ['Amplitude','Amplitude','Amplitude']
yaxes = ['Number of units ','Number of units','Number of units']
titles = ['Amplitude cutoff = 91.1 ','Amplitude cutoff = 3.9','Amplitude cutoff = 0.0'] 
data=[units_b['amps']['0'], units_b['amps']['14'],units_b['amps']['33']]
f,a = plt.subplots(3,1,figsize=(10,20))
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(data[idx],50)
    ax.set_title(titles[idx])
    ax.set_xlabel(xaxes[idx])
    ax.set_ylabel(yaxes[idx])
    # ax.set_xticks(np.linspace(min(data[idx]), max(data[idx])+1,num=3))
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
f.tight_layout(pad=2.0)
# dflux.hist('rate', bins=100, ax=axes[0])
# dflux2.hist('rate', bins=100, ax=axes[1])


unit='0'
amps2=units_b['amps'][unit]
ax=axes[0]
n,bins,patches=ax.hist(amps2,50)
n,bins,patches=plt.hist(amps2,50,ax=axes[0])
unit='14'
amps2=units_b['amps'][unit]
n,bins,patches=plt.hist(amps2,50,ax=axes[1])
unit='33'
amps2=units_b['amps'][unit]
n,bins,patches=plt.hist(amps2,50,ax=axes[2])

#getting real amplitudes failed...

# #first load amps_b (not done here, but needs to be done!)
# unit = '0'
# xx=amps_b[unit] #all amplitudes for a unit
# nbins = 50
# bins_list= np.linspace(np.min(xx[1,:]), np.max(xx[1,:]), nbins)
# n,bins,patches = plt.hist(xx[:,1],bins = bins_list ,facecolor = 'blue',alpha = 0.5)
# #plt.xticks(np.arange(10,40,5))
# #plt.yticks(np.arange(0,1000,200))
# plt.xlim(np.min(xx[1,:]),np.max(xx[1,:]))
# #plt.ylim(0, 200)

# plt.show()
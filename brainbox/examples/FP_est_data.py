# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:25:01 2020

@author: Steinmetz Lab User

apply firing rate sliding window refractory period analysis to sample data
"""

from pathlib import Path
import numpy as np
import alf.io as aio
import matplotlib.pyplot as plt
from max_acceptable_isi_viol_2 import max_acceptable_cont_2
import brainbox as bb
from phylib.stats import correlograms
import pandas as pd

#def run_noise_cutoff():
alf_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab\test_path_alf'
ks_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab'

binSize=0.0005

# b = np.arange(1e-6,0.0055,binSize)
# b=[1,1.5,2,2.5,3,3.5,4,5,7,10]
b = np.arange(0,.0125,binSize)+1e-6
bTestIdx = [2, 3, 4, 5, 6, 8, 10, 15,  20]
bTest = [b[i] for i in bTestIdx]

thresh = 0.1
spks_b = aio.load_object(alf_dir, 'spikes')
clstrs_b = aio.load_object(alf_dir, 'clusters')
units_b = bb.processing.get_units_bunch(spks_b)
units = list(units_b.amps.keys())
n_units = np.max(spks_b.clusters) + 1


uidx=0
didpass=np.empty([len(units)])
for unit in units:
    #unit = units[685] #635 max spike rate #681 1.43
    ts = units_b['times'][unit]
    if(len(ts)>0):
        recDur = (ts[-1]-ts[0])
        fr_source = len(ts)/recDur
        print(fr_source)
        mfunc =np.vectorize(max_acceptable_cont_2)
        m = mfunc(fr_source,bTest,recDur,fr_source/10,thresh)
        c0 = correlograms(ts,np.zeros(len(ts),dtype='int8'),cluster_ids=[0],bin_size=binSize,sample_rate=20000,window_size=.05,symmetrize=False)
        res = np.cumsum(c0[0,0,bTestIdx])
        didpass[uidx] = int(np.any(np.less_equal(res,m)))
        print(didpass[uidx])
        uidx+=1
    else: 
        didpass[uidx]=0
        
        print(didpass[uidx])
        uidx+=1
        
try:
    fpest = pd.DataFrame(didpass)
    fpest.to_csv(Path(ks_dir, 'fpest.tsv'),
                            sep='\t', header=['fpest'])
except Exception as err:
    print("Could not save 'fpest' to .tsv. Details: \n {}".format(err))


#counterexample 578
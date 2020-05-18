# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:15:28 2020

@author: Steinmetz Lab User
"""

#compare labels

import pandas as pd
from pathlib import Path
import numpy as np

ks_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab\'

#load metrics labels
labelsdf = pd.read_csv(Path(ks_dir,'label.tsv'),sep='\t')
l = labelsdf['label']

noisecutoff = pd.read_csv(Path(ks_dir,'NoiseCutoff.tsv'),sep='\t')
nc = noisecutoff['NoiseCutoff']

rfpv = pd.read_csv(Path(ks_dir,'RefPViol.tsv'),sep='\t')
rfp = rfpv['RefPViol']

#load old ks2 (or by-hand?) labels
ks2df = pd.read_csv(Path(ks_dir,'cluster_group.tsv'),sep='\t')
k = ks2df['group']
kind = ks2df['cluster_id']


xg=kind[np.where(k=='good')[0]] #second column of this is cluster ids that are good
kg=xg.tolist()
kg[0] #sanity check, this should be 13, the first cluster id that is good.
xmua = kind[np.where(k=='mua')[0]] # or k== 'noise')]
xnoise = kind[np.where(k=='noise')[0]]
kmua=xmua.tolist()
knoise = xnoise.tolist()
xbad = kmua+knoise

l1 = np.where(l==1)[0]
l0 = np.where(l==0)[0]

interg1 = np.intersect1d(l1,kg) #these are the units that both labeled GOOD
print("number of both good = ", len(interg1))
interg0 = np.intersect1d(l0,kg)
print("number of manual good but metric bad = ", len(interg0))


interb1 = np.intersect1d(l1,xbad)
print("number of manual bad (noise and mua) but metric good = ", len(interb1))

interb0 = np.intersect1d(l0,xbad)
print("number of both bad = ", len(interb0))



#if mismatch, why?
rfpval = rfpv['RefPViol'].tolist()
np.array(rfpval)[interg0.astype(int)]

np.array(nc)[interg0.astype(int)]




if(0):
    #check specific units
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
    alf_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab\test_path_alf'
    alf_probe_dir=alf_dir
    spks_b = aio.load_object(alf_probe_dir, 'spikes')  
    units_b = bb.processing.get_units_bunch(spks_b)
    units = list(units_b.amps.keys())
    
    unit = units[219]
    ts = units_b['times'][unit]
    amps = units_b['amps'][unit]
    
    #test a unit on noise cutoff
    nbins = 100
    end_low=1
    quartile_length=.25
    std_cutoff = 7
    # if(len(amps)>1):
    bins_list= np.linspace(0, np.max(amps), nbins)
    n,bins = np.histogram(amps,bins = bins_list)
    #sanity check plot 
    plt.hist(amps,bins = bins_list)
    plt.ylabel('Number of units')
    plt.xlabel('Amplitudes (uV) --FIX UNITS!!')
    dx = np.diff(n) 
    idx_nz = np.nonzero(dx) #indices of nonzeros
    length_nonzeros = idx_nz[0][-1]-idx_nz[0][0] #length of the entire stretch, from first nonzero to last nonzero
    idx_peak = np.argmax(n)
    length_top_half = idx_nz[0][-1]-idx_peak
    high_quartile = 1-quartile_length
    
    high_quartile_start_ind2 = int(np.ceil(.5*length_top_half + idx_peak))
    xx=idx_nz[0][idx_nz[0]>high_quartile_start_ind2]
    first_idx_after = xx[0]
    mean_touse = np.mean(n[xx])
    std_touse = np.std(n[xx])
    
    
    high_quartile_start_ind = int(np.ceil(high_quartile*(length_nonzeros)))+idx_nz[0][0]
    high_quartile_end_ind = idx_nz[0][-1]
    mean_high_quartile = np.mean(n[high_quartile_start_ind:high_quartile_end_ind])
    std_high_quartile = np.std(n[high_quartile_start_ind:high_quartile_end_ind])
    first_low_quartile = (n[idx_nz[0][1]])
    
    cutoff=(first_low_quartile-mean_high_quartile)/std_high_quartile
    cutoff2 = (first_low_quartile-mean_touse)/std_touse
    print(cutoff)
    print(cutoff2)
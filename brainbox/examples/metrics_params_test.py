# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:17:24 2020

@author: Noam Roth

test the amplitude metric
"""

import time
import os
from oneibl.one import ONE
from pathlib import Path
import numpy as np
import alf.io as aio
import matplotlib.pyplot as plt
import brainbox as bb
from brainbox.examples import max_acceptable_isi_viol_2, defined_metrics
from brainbox.examples.max_acceptable_isi_viol_2 import max_acceptable_cont_2
from phylib.stats import correlograms
import pandas as pd
from brainbox.metrics import metrics
from brainbox.examples.defined_metrics import FP_RP, noise_cutoff, peak_to_peak_amp

#example session
eid = '5cf2b2b7-1a88-40cd-adfc-f4a031ff7412'
probe_name = 'probe_right'


one=ONE()
ses_path=one.path_from_eid(eid)    
alf_probe_dir = os.path.join(ses_path, 'alf', probe_name)
ks_dir = alf_probe_dir
spks_b = aio.load_object(alf_probe_dir, 'spikes')  
units_b = bb.processing.get_units_bunch(spks_b)
units = list(units_b.amps.keys())
lengths_samples = [len(v) for k, v in units_b.samples.items()]
units_nonzeros=[i for i,d in enumerate(lengths_samples) if d>0]
n_units = len(units_nonzeros) #only compute metrics for units with no samples


#for cases where raw data is available locally: 
ephys_file_dir = os.path.join(ses_path, 'raw_ephys_data', probe_name)
ephys_file = os.path.join(ses_path, 'raw_ephys_data', probe_name,'_iblrig_ephysData.raw_g0_t0.imec.ap.cbin')
#create params.py file
params_file = os.path.join(ks_dir,'params.py')
if os.path.exists(ephys_file) and not os.path.exists(params_file):
    f = open(params_file,"w+")
    f.write('dat_path = ' + 'r"' + ephys_file + '"\n' +  '''n_channels_dat = 385
    dtype = 'int16'
    offset = 0
    sample_rate = 30000
    hp_filtered = False
    uidx=0''' )
    f.close()
            
# Initialize metrics
cum_amp_drift = np.full((n_units,), np.nan)
cum_depth_drift = np.full((n_units,), np.nan)
cv_amp = np.full((n_units,), np.nan)
cv_fr = np.full((n_units,), np.nan)
frac_isi_viol = np.full((n_units,), np.nan)
frac_missing_spks = np.full((n_units,), np.nan)
fp_est = np.full((n_units,), np.nan)
pres_ratio = np.full((n_units,), np.nan)
pres_ratio_std = np.full((n_units,), np.nan)
ptp_sigma = np.full((n_units,), np.nan)

units_missing_metrics = set()
label=np.empty([len(units)])
RefPViol = np.empty([len(units)])
NoiseCutoff = np.empty([len(units)])
MeanAmpTrue = np.empty([len(units)])

for idx,unit in enumerate(units_nonzeros):
    if unit == units_nonzeros[0]:
        t0 = time.perf_counter()  # used for computation time estimate
    
    print('computing metrics for unit ' + str(unit) + '...' )

    #load relevant data for unit
    ts = units_b['times'][str(unit)]
    amps = units_b['amps'][str(unit)]
    samples = units_b['samples'][str(unit)]
    depths = units_b['depths'][str(unit)]
    

    quartile_lengthvec=[.05, .1, .2, .25, .3, .5]
    i=0
    nbins = 100
    end_lowvec = [ 1, 5, 10]
    cutoff_plot=np.zeros([len(quartile_lengthvec),len(end_lowvec)])
    for quartile_length in quartile_lengthvec:
        j=0
        for end_low in end_lowvec:
            param1 = quartile_length
            param2 = end_low
            if(len(amps)>1):
                bins_list= np.linspace(0, np.max(amps), nbins)
                n,bins = np.histogram(amps,bins = bins_list) 
                dx = np.diff(n) 
                idx_nz = np.nonzero(dx) #indices of nonzeros
                length_nonzeros = idx_nz[0][-1]-idx_nz[0][0] #length of the entire stretch, from first nonzero to last nonzero
                high_quartile = 1-quartile_length
                idx_peak = np.argmax(n)
                length_top_half = idx_nz[0][-1]-idx_peak
                high_quartile = 1-(2*quartile_length)
                
                high_quartile_start_ind = int(np.ceil(high_quartile*length_top_half + idx_peak))
                xx=idx_nz[0][idx_nz[0]>high_quartile_start_ind]
                if len(n[xx])>0:
                    mean_high_quartile = np.mean(n[xx])
                    std_high_quartile = np.std(n[xx])                            
                    first_low_quartile = np.mean(n[idx_nz[0][1:end_low+1]])
                    if std_high_quartile>0:
                        cutoff=(first_low_quartile-mean_high_quartile)/std_high_quartile
                    else:
                        cutoff=np.float64(np.nan)
                else:
                    cutoff=np.float64(np.nan)
            else:
                cutoff=np.float64(np.nan)
            
            cutoff_plot[i,j] = cutoff
            j+=1
        i+=1
                

fig,ax =plt.subplots()
xx = ax.imshow(cutoff_plot)
cbar = fig.colorbar(xx)

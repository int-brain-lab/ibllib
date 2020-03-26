# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:12:30 2020

@author: Noam Roth
"""

#in v1_cert environment


import time
from pathlib import Path
import numpy as np
import pandas as pd
import brainbox as bb
import alf.io as aio
import matplotlib.pyplot as plt


#def run_noise_cutoff():
alf_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab\test_path_alf'
ks_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab'


spks_b = aio.load_object(alf_dir, 'spikes')
clstrs_b = aio.load_object(alf_dir, 'clusters')
units_b = bb.processing.get_units_bunch(spks_b)
units = list(units_b.amps.keys())
n_units = np.max(spks_b.clusters) + 1

noise_cutoff_vec = np.full((n_units,), np.nan)

for unit in units:
    if unit == units[0]:
        t0 = time.perf_counter()  # used for computation time estimate
    
    try: 
        # Need timestamps, amps, depths
        ts = units_b['times'][unit]
        amps = units_b['amps'][unit]
        depths = units_b['depths'][unit]
        
    #    noise_cutoff_vec[int(unit)] = noise_cutoff(amps)
     
    
    #df_cutoff = pd.DataFrame(noise_cutoff_vec)
    #
    #df_cutoff.to_csv(Path(ks_dir, 'noise_cutoff.tsv'),
    #                                sep='\t', header=['noise_cutoff'])
    #    
        
        
        
    #def noise_cutoff(amps,quartile_length=.25,std_cutoff = 2):
        quartile_length=.2
        std_cutoff = 2
        nbins = 500
        bins_list= np.linspace(0, np.max(amps), nbins)
        n,bins,patches = plt.hist(amps,bins = bins_list ,facecolor = 'blue',alpha = 0.5)
        plt.xticks(np.arange(10,40,5))
        plt.yticks(np.arange(0,1000,200))
        plt.xlim(10,37)
        plt.ylim(0, 200)
        plt.show()
        
        
        dx = np.diff(n)
        
        #assumption: it will always be cut off on the low side, not the high side, because of spike sorting.
         
        
        #issue with the code below is that it includes lots of zeros. so let's start with the first nonzero on both sides.
        #        mean_high_quartile= np.mean(dx[int(np.ceil(.75*len(dx))):len(dx)-1])
        #        std_high_quartile = np.std(dx[int(np.ceil(.75*len(dx))):len(dx)-1])
        #        mean_low_quartile = np.mean(dx[0:int(np.ceil(.25*len(dx)))])
        
        
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
        #return cutoff 
        noise_cutoff_vec[int(unit)] = cutoff
    except Exception as err:
        print("Failed to compute 'cum_drift(amps)' for unit {}. Details: \n {}"
                  .format(unit, err))
        noise_cutoff_vec[int(unit)] = np.nan

df_cutoff = pd.DataFrame(noise_cutoff_vec)

df_cutoff.to_csv(Path(ks_dir, 'noise_cutoff.tsv'),
                                sep='\t', header=['noise_cutoff'])
 
 
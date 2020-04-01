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
from max_acceptable_isi_viol import max_acceptable_cont

#def run_noise_cutoff():
alf_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab\test_path_alf'
ks_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab'


spks_b = aio.load_object(alf_dir, 'spikes')
clstrs_b = aio.load_object(alf_dir, 'clusters')
units_b = bb.processing.get_units_bunch(spks_b)
units = list(units_b.amps.keys())
n_units = np.max(spks_b.clusters) + 1

unit = units[635] #635 max spike rate #681 1.43
ts = units_b['times'][unit]
fr_source = len(ts)/(ts[-1]-ts[0])
print(fr_source)
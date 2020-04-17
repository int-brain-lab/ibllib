# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:26:47 2020

@author: Steinmetz Lab User


Code to generate, for N example units, plots of wfs, aCGs, amplitude histogram, psth, and table of metrics
"""

# %% imports
import time
from pathlib import Path
import numpy as np
import pandas as pd
import brainbox as bb
import alf.io as aio

# %% #first generate metrics and print results in a table

#example:
ks_dir = 
path_to_alf_out
from gen_phy_metrics import gen_metrics
import ibllib.ephys.spikes as e_spks
# (*Note, if there is no 'alf' directory, make 'alf' directory from 'ks2' output directory):
e_spks.ks2_to_alf(path_to_ks_out, path_to_alf_out)
# # Load the alf spikes bunch and clusters bunch, and get a units bunch.
spks_b = aio.load_object(path_to_alf_out, 'spikes')
clstrs_b = aio.load_object(path_to_alf_out, 'clusters')
units_b = bb.processing.get_units_bunch(spks_b)  # may take a few mins to compute

# gen_metrics(alf_dir,ks_dir)

p_vals_b, variances_b = bb.metrics.unit_stability(units_b)
# Plot histograms of variances color-coded by depth of channel of max amplitudes
fig = bb.plot.feat_vars(units_b, feat_name='amps')
# Get all unit IDs which have amps variance > 50
var_vals = np.array(tuple(variances_b['amps'].values()))
bad_units = np.where(var_vals > 50)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:31:58 2020

@author: guido
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ibllib import atlas
from brainbox.atlas import plot_atlas

# Coordinates of slices in mm
ML = -0.5
AP = 1
DV = -2

# Generate some mock data
ba = atlas.AllenAtlas(25)
regions = np.random.choice(ba.regions.acronym, size=500, replace=False)  # pick 500 random regions
values = np.random.uniform(-1, 1, 500)  # generate 500 random values

# Plot atlas 
f, axs = plt.subplots(1, 3, figsize=(40, 10))
plot_atlas(regions, values, ML, AP, DV, color_palette='RdBu_r', minmax=[-1, 1], axs=axs)

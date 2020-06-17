"""
The International Brain Laboratory
Anne Urai, CSHL, 2020-03-17
"""

import pandas as pd
import numpy as np
import datajoint as dj
from ibl_pipeline import subject, acquisition
ephys = dj.create_virtual_module('ephys', 'ibl_ephys')
# The tables relevant are ProbeInsertion, ProbeTrajectory, DefaultCluster, AlignedTrialSpikes

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns
import os
figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ==================================== #
# FIND ALL PROBES IN THE REPEATED SITE
# ==================================== #

# which mice have this repeated site logged in DJ?
tmp = ephys.ProbeTrajectory * subject.Subject * \
      (subject.SubjectLab & 'lab_name = "churchlandlab"')

probes_rs = (ephys.ProbeTrajectory & 'insertion_data_source = "Micro-manipulator"'
             & 'x BETWEEN -2500 AND -2000' & 'y BETWEEN -2500 AND -1500'
             & 'theta BETWEEN 14 AND 16' & 'depth BETWEEN 3500 AND 4500' & 'phi BETWEEN 179 AND 181')

# ==================================== #
# GET NEURONS ALONG THOSE PROBES
# ==================================== #

clust = ephys.DefaultCluster * ephys.DefaultCluster.Metrics * probes_rs * subject.Subject
clust = clust.proj('cluster_amp', 'cluster_depth', 'firing_rate', 'subject_nickname', 'metrics',
                   days_old_at_ephys='DATEDIFF(session_start_time, subject_birth_date)')
clusts = clust.fetch(format='frame').reset_index()

# put metrics into df columns from the blob
for kix, k in enumerate(clusts['metrics'][0].keys()):
    tmp_var = []
    for id, c in clusts.iterrows():
        if k in c['metrics'].keys():
            tmp = c['metrics'][k]
        else:
            tmp = np.nan
        tmp_var.append(tmp)
    clusts[k] = tmp_var

print(clusts.describe())

# cluster_amp: now in KS2 units, convert to uV
# see https://int-brain-lab.slack.com/archives/CK9QY8C82/p1584623970061900
clusts['cluster_amp'] = clusts['cluster_amp'] * 1/2.3 * 10**6

# ==================================== #
# PLOT
# ==================================== #

g = sns.FacetGrid(data=clusts, col='subject_nickname', col_wrap=4, hue='ks2_label',
                  palette=dict(good="seagreen", mua="orange"))
g.map(sns.scatterplot, "firing_rate", "cluster_depth", alpha=0.5).add_legend()
g.set_titles('{col_name}')
g.set_xlabels('Firing rate (spks/s)')
g.set_ylabels('Depth')
plt.tight_layout()
g.savefig(os.path.join(figpath, 'neurons_rsi_firingrate.pdf'))

g2 = sns.FacetGrid(data=clusts, col='subject_nickname', col_wrap=4, hue='ks2_label',
                   palette=dict(good="seagreen", mua="orange"))
g2.map(sns.scatterplot, "cluster_amp", "cluster_depth", alpha=0.5)
g2.set_titles('{col_name}')
g2.set_xlabels('Amplitude (a.u.)')
g2.set_ylabels('Depth')
plt.tight_layout()
g2.savefig(os.path.join(figpath, 'neurons_rsi_cluster_amp.pdf'))

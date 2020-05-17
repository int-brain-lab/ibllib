# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:30:28 2020

@author: Noam Roth

example script to load raw amplitudes

"""


import numpy as np
from oneibl.one import ONE
import alf.io
import brainbox.io.one as bb
from ibllib.io import spikeglx

one = ONE()
eid = one.search(subject='KS022', date='2019-12-10')[0]
eid = one.search(subject ='ZM_2104', date='2019-09-19')[0]
probe = 'probe_right'

# Get paths to the required dataset_types. If required dataset_types are not already downloaded,
# download them.
dtypes = [
        'clusters.amps',
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'clusters.peakToTrough',
        'clusters.uuids',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'spikes.amps',
        'spikes.clusters',
        'spikes.depths',
        'spikes.samples',
        'spikes.templates',
        'spikes.times',
        'ephysData.raw.meta',
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        '_iblrig_RFMapStim.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_codeFiles.raw'
        ]
d_paths = one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)

#%%
ks_dir = d_paths[1]
# spikes = alf.io.load_object(ks_dir,'spikes.times')
ephys_file = Path(r'C:/Users/Steinmetz Lab User/Downloads/FlatIron/mainenlab/Subjects/ZM_2104/2019-09-19/001/raw_ephys_data/probe_right/_iblrig_ephysData.raw_g0_t0.imec.ap.cbin')
# ephys_file = Path(r'C:/Users/Steinmetz Lab User/Downloads/FlatIron/cortexlab/Subjects/KS022/2019-12-10/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin')
spikes, clusters = bb.load_spike_sorting(eid)



#%%
samples = spikes[0]['samples']
clusters = spikes[0]['clusters']
#now take cluster 0
cluster_id = 1
inds = np.where(clusters==cluster_id)[0]
samp_inds = samples[inds]


sr = spikeglx.Reader(ephys_file)

#take a subset of n of these
n = 5
xx = np.random.choice(samp_inds,n)# samp_inds[range(n)]

wfs=np.zeros((384,len(xx)))
wfs_baseline=np.zeros((384,len(xx)))
cnt=0
for i in xx:
    
    wf = sr.data[int(i)]
    wf_baseline = wf[:-1]-np.median(wf[:-1])
    # plt.plot(wf[:-1])
    plt.plot(wf_baseline)
    wfs[:,cnt] = wf[:-1]
    wfs_baseline[:,cnt] = wf_baseline
    cnt+=1
amps = np.max(wfs_baseline,axis=0)-np.min(wfs_baseline,axis=0)
mean_amp = np.mean(amps)
# return mean_amp



#%%
from ibllib.plots import Density, Traces
d = Density(templates.waveforms[it, :, ordre].transpose(), fs=30000)
t = Traces(templates.waveforms[it, :, ordre].transpose(), fs=30000, ax=d.ax)

sr[i0, i1] 
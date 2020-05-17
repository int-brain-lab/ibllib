# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:38:20 2020

@author: Noam Roth

Code to load an eid, get the true amplitudes in order to  (for now:)
1) test two ways of getting waveforms
2) plot waveforms
3) compute mean amplitude and SNR for a neuron

"""

from oneibl.one import ONE
from ibllib.ephys.ephysqc import phy_model_from_ks2_path
from pathlib import Path
import numpy as np

# eid = 'aad23144-0e52-4eac-80c5-c4ee2decb198'
# probe_name = 'probe01'

# ks2_path = r'C:\Users\Steinmetz Lab User\Downloads\FlatIron\churchlandlab\Subjects\CSHL049\2020-01-08\001\alf\probe00'




# xy = one.load(eid, dataset_types=['spikes.times','templates.channels'])

one = ONE()

eid = one.search(dataset_types=['spikes.times','spikes.amps','templates.waveforms'])


#%%

## Loads some spike data

#was having some issue where there were two probes? 

my_data = one.load(eid[1], dataset_types=['spikes.times', 'spikes.amps','templates.waveforms','templates_ind'],dclass_output=True)
st = my_data.data[0]
sa = my_data.data[1]
tw = my_data.data[2]



#%%
data_types = ['templates.waveforms.npy', #nTemplates x nTimesPoints x nChannels (templates.npy?)
'whitening_mat_inv.npy', #nChannels x nChannels
# 'channel_positions.npy', #gives us y coords, which is the second column of this array
'spikes.templates.npy', #which template each spike came from (nSpikes)
'spikes.amps.npy'] #the amount by which the template was scaled to extract each spike (nSpikes) (amplitudes?)

eid = one.search(dataset_types=data_types)

my_data = one.load(eid[0], dataset_types=data_types,dclass_output=True)

#%%
#example dataset which has all of these:
ks_dir = r'C:\Users\Steinmetz Lab User\Downloads\FlatIron\churchlandlab\Subjects\CSHL049\2020-01-08\001\alf\probe00'
# ks_dir = r'C:\Users\Steinmetz Lab User\Downloads\FlatIron\cortexlab\Subjects\KS022\2019-12-10\001\alf\probe00'

tw = np.load(Path(ks_dir + r'\templates.waveforms.npy'))
twc = np.load(Path(ks_dir + r'\templates.waveformsChannels.npy'))
wmi = np.load(Path(ks_dir + r'\whitening_mat_inv.npy'))
stemp = np.load(Path(ks_dir + r'\spikes.templates.npy'))
samps =  np.load(Path(ks_dir + r'\spikes.amps.npy'))
sc = np.load(Path(ks_dir + r'\spikes.clusters.npy'))

#%%

#unwhiten the templates
tempsUnW = np.zeros_like(tw) 
for t in range(len(tw)):
    tempsUnW[t,:,:] = tw[t,:,:]*wmi[twc[t,:],twc[t,:]] #issue that some channels are all zeros? because only includes top 32 channels

tempChanAmps=[]
tempChanAmps = np.max(tempsUnW,axis=1)-np.min(tempsUnW,axis=1) #these amplitudes are already sorted: max chan first, up to 32nd highest amp

scaling_factor = 2.34375e-06
tempAmpsUnscaled = tempChanAmps[:,0] #size 303 (number of templates)
#assign all spikes the amplitude of their template multiplied by their scaling amplitudes 
spikeAmps = tempAmpsUnscaled[stemp]*samps/scaling_factor







# % take the average of all spike amps to get actual template amps (since
# % tempScalingAmps are equal mean for all templates)
tids = np.unique(stemp)
tempAmps = np.zeros(len(tempChanAmps))
spikeAmpOrig = np.zeros(len(tempChanAmps))
for t in tids:
    tempAmps[t] = np.mean(spikeAmps[stemp==t])
    spikeAmpOrig[t] = np.mean(samps[stemp==t])
    
    
    
    #%%
ta =  #template average for each unique template
ta = clusterAverage(spikeTemplates+1, spikeAmps);
tids = unique(spikeTemplates);
tempAmps(tids+1) = ta; % because ta only has entries for templates that had at least one spike
tempAmps = tempAmps'; % for consistency, make first dimension template number


#%% 

# coords = readNPY(fullfile(ksDir, 'channel_positions.npy'));
# ycoords = coords(:,2); xcoords = coords(:,1);


function [spikeAmps, spikeDepths, templateDepths, tempAmps, tempsUnW, templateDuration, waveforms] = templatePositionsAmplitudes(temps, winv, ycoords, spikeTemplates, tempScalingAmps)
% function [spikeAmps, spikeDepths, templateDepths, tempAmps, tempsUnW, templateDuration, waveforms] = templatePositionsAmplitudes(temps, winv, ycoords, spikeTemplates, tempScalingAmps)
%
% Compute some basic things about spikes and templates
%
% outputs: 
% - spikeAmps is length nSpikes vector with amplitude in unwhitened space
% of every spike
% - spikeDepths is the position along the probe of every spike (according
% to the position of the template it was extracted with)
% - templateDepths is the position along the probe of every template
% - templateAmps is the amplitude of each template
% - tempsUnW are the unwhitened templates
% - templateDuration is the trough-to-peak time (in samples)
% - waveforms: returns the waveform from the max-amplitude channel
%
% inputs: 
% - temps, the templates (nTemplates x nTimePoints x nChannels)
% - winv, the whitening matrix (nCh x nCh)
% - ycoords, the coordinates of the channels (nCh x 1)
% - spikeTemplates, which template each spike came from (nSpikes x 1)
% - tempScalingAmps, the amount by which the template was scaled to extract
% each spike (nSpikes x 1)

% unwhiten all the templates
tempsUnW = zeros(size(temps));
for t = 1:size(temps,1)
    tempsUnW(t,:,:) = squeeze(temps(t,:,:))*winv;
end

% compute the biggest absolute value within each template (obsolete)
% absTemps = abs(tempsUnW);
% tempAmps = max(max(absTemps,[],3),[],2);

% The amplitude on each channel is the positive peak minus the negative
tempChanAmps = squeeze(max(tempsUnW,[],2))-squeeze(min(tempsUnW,[],2));

% The template amplitude is the amplitude of its largest channel (but see
% below for true tempAmps)
tempAmpsUnscaled = max(tempChanAmps,[],2);

% need to zero-out the potentially-many low values on distant channels ...
threshVals = tempAmpsUnscaled*0.3; 
tempChanAmps(bsxfun(@lt, tempChanAmps, threshVals)) = 0;

% ... in order to compute the depth as a center of mass
templateDepths = sum(bsxfun(@times,tempChanAmps,ycoords'),2)./sum(tempChanAmps,2);

% assign all spikes the amplitude of their template multiplied by their
% scaling amplitudes (templates are zero-indexed)
spikeAmps = tempAmpsUnscaled(spikeTemplates+1).*tempScalingAmps;


% take the average of all spike amps to get actual template amps (since
% tempScalingAmps are equal mean for all templates)
ta = clusterAverage(spikeTemplates+1, spikeAmps);
tids = unique(spikeTemplates);
tempAmps(tids+1) = ta; % because ta only has entries for templates that had at least one spike
tempAmps = tempAmps'; % for consistency, make first dimension template number

% Each spike's depth is the depth of its template
spikeDepths = templateDepths(spikeTemplates+1);

% Get channel with largest amplitude, take that as the waveform
[~,max_site] = max(max(abs(temps),[],2),[],3);
templates_max = nan(size(temps,1),size(temps,2));
for curr_template = 1:size(temps,1)
    templates_max(curr_template,:) = ...
        temps(curr_template,:,max_site(curr_template));
end
waveforms = templates_max;

% Get trough-to-peak time for each template
[~,waveform_trough] = min(templates_max,[],2);
[~,templateDuration] = arrayfun(@(x) ...
    max(templates_max(x,waveform_trough(x):end),[],2), ...
    transpose(1:size(templates_max,1)));





© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
#%%

#check empirical question of whether wmi matters:
ks_dir = r'E:\Hopkins_CortexLab\test_path_loadks1'
wmi = np.load(Path(ks_dir + r'\whitening_mat_inv.npy'))

stemp = np.load(Path(ks_dir + r'\spike_templates.npy'))
samps =  np.load(Path(ks_dir + r'\amplitudes.npy'))
sc = np.load(Path(ks_dir + r'\spike_clusters.npy'))



ks_dir = r'E:\Hopkins_CortexLab\test_path_loadks2'
temps = np.load(Path(ks_dir + r'\templates.npy'))


#%% 
# % - temps, the templates (nTemplates x nTimePoints x nChannels)
# % - winv, the whitening matrix (nCh x nCh)
# % - ycoords, the coordinates of the channels (nCh x 1)
# % - spikeTemplates, which template each spike came from (nSpikes x 1)
# % - tempScalingAmps, the amount by which the template was scaled to extract
# % each spike (nSpikes x 1)


#unwhiten the templates
tempsUnW = np.zeros_like(tw) 
tempsControl = np.zeros_like(tw)
for t in range(len(tw)):
    tempsUnW[t,:,:] = tw[t,:,:]*wmi[twc[t,:],twc[t,:]] #issue that some channels are all zeros? because only includes top 32 channels
    tempsControl[t,:,:] = tw[t,:,:]*wmi_identity[twc[t,:],twc[t,:]]
tempChanAmps=[]
tempChanAmps = np.max(tempsUnW,axis=1)-np.min(tempsUnW,axis=1) #these amplitudes are already sorted: max chan first, up to 32nd highest amp

tempChanAmps_control=[]
tempChanAmps_control = np.max(tempsControl,axis=1)-np.min(tempsControl,axis=1) #these amplitudes are already sorted: max chan first, up to 32nd highest amp


scaling_factor = 2.34375e-06

tempAmpsUnscaled = tempChanAmps[:,0] #size 303 (number of templates)
#assign all spikes the amplitude of their template multiplied by their scaling amplitudes 
spikeAmps = tempAmpsUnscaled[stemp]*samps/scaling_factor


tempAmpsUnscaled_control = tempChanAmps_control[:,0] #size 303 (number of templates)
#assign all spikes the amplitude of their template multiplied by their scaling amplitudes 
spikeAmps_control = tempAmpsUnscaled_control[stemp]*samps/scaling_factor


# %%
plt.scatter(tempAmpsUnscaled,tempAmpsUnscaled_control)
plt.ylim((-1e-5,1e-5))
plt.xlim((-1e-5,1e-5))
plt.plot([-1e-5, 1e-5], [-1e-6, 1e-6], 'k-', color = 'r')

#%%




# % take the average of all spike amps to get actual template amps (since
# % tempScalingAmps are equal mean for all templates)
tids = np.unique(stemp)
tempAmps = np.zeros(len(tempChanAmps))
spikeAmpOrig = np.zeros(len(tempChanAmps))
for t in tids:
    tempAmps[t] = np.mean(spikeAmps[stemp==t])
    spikeAmpOrig[t] = np.mean(samps[stemp==t])
    
#load wmi
#load other stuff
#do it with wmi
#do it without wmi

#plot mean amp per cluster w/wmi and without (Scatter)




#%%
rms_amps, rms_times = (alf.io.load_object(ephys_path, '_iblqc_ephysTimeRms' +
                                   format)).values()


#%% 
st, sa, tw = one.load(eid[1], dataset_types=['spikes.times', 'spikes.amps','templates.waveforms','templates_ind'])
## Plot spikes density accross the whole experiment with/without running average
bc = np.bincount(np.int64(st[:,0]-np.min(st[:,0])))
t = np.linspace(0,np.double(bc.size)/86400, bc.size)

ts = pd.Series(bc, t)
plt.plot(t*86400, bc)
plt.plot(t*86400, np.convolve(np.ones(30)/30, bc, 'same'))
#%%
        templates_chs = self.templates_channels
        templates_wfs = self.sparse_templates.data[np.arange(self.n_templates), :, templates_chs]
        templates_wfs_unw = templates_wfs.T * self.wmi[templates_chs, templates_chs]
        templates_amps = np.abs(
            np.max(templates_wfs_unw, axis=0) - np.min(templates_wfs_unw, axis=0))

        # scale the spike amplitude values by the template amplitude values
        amplitudes_v = np.zeros_like(self.amplitudes)
        for t in range(self.n_templates):
            idxs = self.get_template_spikes(t)
            amplitudes_v[idxs] = self.amplitudes[idxs] * templates_amps[t]

m = phy_model_from_ks2_path(Path(ks2_path))
true_amps = m.get_amplitudes_true()


# phy_spikes_subset.waveforms.npy
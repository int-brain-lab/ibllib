# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:21:29 2020

@author: Noam Roth
"""


import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import brainbox as bb


#simulate the main neuron with a number of different firing rates and refractory periods
FR_vec = np.arange(0,20,.25) #spikes/s
RP_vec = np.arange(0.0001,.006,.0001) # seconds
cont_thresh = .1 #percent of the neuron's firing rate that you would allow for contamination


# a vector of possible observed contamination spikes
obs_cont_count_vec = np.arange(0,50,1)

#simulate the contaminating neuron
True_cont_rate = np.arange(0,5,.01) #true contamination rate, spikes/s

total_time_window = 3600 #time in the recording session (s)
FP_thresh=.1
#for each true contamination rate, compute the probability of counts

#initialize variables 
max_acceptable_count = np.empty((len(FR_vec),len(RP_vec)))
frind = 0
for fr in FR_vec: 
    acceptable_cont_rate = cont_thresh*fr #you will accept this many contaminating spikes in the isi
    rpind = 0 #initialize
    for rp in RP_vec:
        obs_cont_countind = 0 #initialize
        cumul_prob_array = np.empty(len(obs_cont_count_vec)) #initialize cumulative probability array
        for obs_cont_count in obs_cont_count_vec:
            
            #simulate the rate of the contaminating neuron via a poisson process
            poiss_cont = poisson.pmf(obs_cont_count,True_cont_rate*total_time_window*rp)
            #now find where the true contamination rate crosses the acceptable contamination rate
            #and sum across all of those likelihoods
            int_after_acceptable = np.sum(poiss_cont[np.where(True_cont_rate>acceptable_cont_rate)])
            #sum of the total likelihoods (to normalize to total prob of 1)
            int_total = np.sum(poiss_cont)
            #compute cumulative probability of there being more than the acceptable number of cont spikes
            cumul_prob = int_after_acceptable/int_total 
            cumul_prob_array[obs_cont_countind] = cumul_prob
            obs_cont_countind += 1
        #sanity check plot 1
#            if fr ==4 and rp == RP_vec[150] and obs_cont_count == 10:
#                plot1 = poiss_cont #poisson.pmf(obs_cont_count,True_cont_rate*total_time_window*rp)
#                plt.plot(True_cont_rate,plot1)
#                plt.ylabel('Probability of seeing count = 10 (%)')
#                plt.xlabel('True contamination rate (spks/s)')
#                plt.xticks(np.arange(True_cont_rate[0],True_cont_rate[-1],.5))
#                plt.show()
#                print(cumul_prob)
#        #sanity check plot 2        
#        if fr == 4 and rp == RP_vec[150] :
#            plt.plot(obs_cont_count_vec,cumul_prob_array*100)
#            plt.ylabel('Probability that the observed cont  is above the allowed (%) ')
#            plt.xlabel('Observed contamination count')
#            plt.show()
            
        if(np.where(cumul_prob_array>FP_thresh)[0].size == 0):
            max_acceptable_count[frind][rpind] = -1 #if there are no places where the probability is below the acc
        else:       
            max_acceptable_ind = np.where(cumul_prob_array>FP_thresh)[0][0]
            max_acceptable_count[frind][rpind] = obs_cont_count_vec[max_acceptable_ind]
#            plt.xticks(np.arange(True_cont_rate[0],True_cont_rate[-1],.5)) 
        rpind +=1
    frind +=1
    
plt.imshow(max_acceptable_count,origin='lower')
cb = plt.colorbar()
cb.set_label('number of allowed contaminations)')
plt.ylabel('Firing rate (spks/s)')
plt.xlabel('Refractory period length (ms)')
plt.yticks(np.arange(0,len(FR_vec)+1,5),FR_vec[(np.arange(1,len(FR_vec)+1,5))-1])
plt.xticks(np.arange(0,len(RP_vec+1),5), np.round(RP_vec[np.arange(0,len(RP_vec+1),5)]*1000,decimals=2))
plt.show()




#################
#testing examples
#################
from matplotlib.pyplot import acorr
import random

true_rp = 4 #ms
true_fr = 10
totalT = 3600 #1 hr in seconds
dt = 1/10000 #0.1 ms in seconds
nbins = np.floor(totalT/dt)
train=[]
for i in range(int(nbins)):
    if  random.random() < true_fr*dt:
        train.append(1)
    else:
        train.append(0)

trainA = np.array(train)
spkTs = np.where(trainA==1)[0] #spike times in ms
isis = np.diff(spkTs)
isi_viols = np.where(isis<true_rp)
#get rid of isi violations to make this a real "biological" neuron
trainA[spkTs[isi_viols]] = 0

#sanity check, isi_viols2 should be an empty array
spkTs2 = np.where(trainA==1)[0] #spike times in ms
isis2 = np.diff(spkTs2)
isi_viols2 = np.where(isis2<rp)


#now add a contaminating neuron. 
fr_cont = 1
train_cont=[]
for i in range(int(nbins)):
    if  random.random() < fr_cont*dt:
        train_cont.append(1)
    else:
        train_cont.append(0)

#should this one be biological too? get rid of isi viols

train_observed = trainA+train_cont
spkTs_observed = np.where(train_observed==1)[0] #spike times in ms
isis_observed = np.diff(spkTs_observed)




#now do the FR/RP analysis on this observed neuron
FR_observed = sum(train_observed)/totalT


fr_idx = (np.abs(FR_vec - FR_observed)).argmin()
cont_acceptable = max_acceptable_count[fr_idx]
cont_observed = []
for rp in RP_vec:
    isi_viols_observed = np.where(isis_observed<(rp*1000))[0]
    cont_observed.append(len(isi_viols_observed))
    
plt.plot(RP_vec*1000,cont_acceptable,'b--', RP_vec*1000,cont_observed,'r--')
plt.xlabel('Refractory period length (ms)')
plt.xticks(np.arange(0,RP_vec[-1]*1000,1))
#, np.round(RP_vec[np.arange(0,len(RP_vec+1),5)]*1000,decimals=2))
plt.show()



#examples that work:
#true fr is 10, fr_cont=0, true rp = 2, crosses at 2. true rp = 4, crosses at 4. etc.
#same but with fr_cont=1, crosses earlier!



#example issues:
#same example as above but with fr_cont=.1, stays close to acceptable until true_rp. BUT, crosses earlier!
#why step functions? 
#fr_cont = 0 has a refractory period of 5?!


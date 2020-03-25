# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:28:59 2020

@author: Steinmetz Lab User
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

unit = units[635] #635
ts = units_b['times'][unit]
fr_source = len(ts)/(ts[-1]-ts[0])
print(fr_source)

isis = np.diff(ts)
isis_sampled = random.choices(isis,k=len(ts))
ts_sampled = np.cumsum(isis_sampled)
total_time = ts_sampled[-1]
#sanity check, fr_sampled should approximate fr_source
fr_sampled = len(ts_sampled)/(total_time)

#remove isi_violations
true_rp =.004
isi_viols = np.where(np.array(isis_sampled)<true_rp)
n_isi_viols = len(isi_viols[0])
#get rid of isi violations to make this a real "biological" neuron
    
timestamps_sampled_real = np.delete(ts_sampled,isi_viols[0])

train_real = np.zeros(int(np.ceil(total_time*10000))+1)
timestamps_micros = [element * 10000 for element in timestamps_sampled_real]
train_real[np.ceil(timestamps_micros).astype(int)]=1

#now add contaminating neuron
nbins = np.ceil(total_time*10000+1)
dt = 1/10000

percent_cont = np.arange(0,15,2.5)

cont_observed = np.empty([len(percent_cont),len(RP_vec)])


rpidx=0
for rp in RP_vec:
    cont_acceptable[rpidx] = max_acceptable_cont(FR_observed,rp,total_time,FR_observed*.1,.1)
    rpidx+=1

pidx=0
for p in percent_cont:
    isis_cont = random.choices(isis,k=(int(len(ts)*p/100)))
    ts_cont = np.cumsum(isis_cont)
    fr_cont = len(ts_cont)/total_time
    print(fr_cont)
    train_cont = np.zeros(int(np.ceil(total_time*10000))+1)
    timestamps_micros = [element * 10000 for element in ts_cont]
    train_cont[np.ceil(timestamps_micros).astype(int)]=1

#    
#    fr_cont = (p/100)*fr_sampled
#    train_cont=[]
#    for i in range(int(nbins)):
#        if  random.random() < fr_cont*dt:
#            train_cont.append(1)
#        else:
#            train_cont.append(0)
#sanity check: should equal fr_cont
#print(sum(train_cont)/(len(train_cont)/10000))

#should this one be biological too? get rid of isi viols

    train_observed = train_real+train_cont
    spkTs_observed = np.where(train_observed==1)[0] #spike times in ms
    isis_observed = np.diff(spkTs_observed)
    
    FR_observed = sum(train_observed)/(len(train_real)*dt)


    fr_idx = (np.abs(FR_vec - FR_observed)).argmin()
    rpidx=0
    for rp in RP_vec:
        cont_acceptable[rpidx] = max_acceptable_cont(FR_observed,rp,total_time,FR_observed*.1,.1)

        isi_viols_observed = np.where(isis_observed<(rp*10000))[0]
        print()
        cont_observed[pidx][rpidx] = len(isi_viols_observed) #.append(len(isi_viols_observed))
        rpidx+=1
    pidx+=1
    
plt.plot(RP_vec*1000,cont_acceptable,'k-', 
         RP_vec*1000,cont_observed[0],'r--',
         RP_vec*1000,cont_observed[1],'b--',
         RP_vec*1000,cont_observed[2],'y--',
         RP_vec*1000,cont_observed[3],'c--')
         
plt.xlabel('Refractory period length (ms)')
plt.xticks(np.arange(0,RP_vec[-1]*1000,1))
#, np.round(RP_vec[np.arange(0,len(RP_vec+1),5)]*1000,decimals=2))
plt.show()




plt.plot(RP_vec*1000,cont_acceptable,'k-', RP_vec*1000,cont_observed[0],'c--')
plt.plot(RP_vec*1000,cont_acceptable,'k-', 
         RP_vec*1000,cont_observed[0],'r--',
         RP_vec*1000,cont_observed[1],'b--',
         RP_vec*1000,cont_observed[2],'y--',
         RP_vec*1000,cont_observed[3],'c--',
         RP_vec*1000,cont_observed[4],'r--',
         RP_vec*1000,cont_observed[5],'b--')
#         RP_vec*1000,cont_observed[6],'y--',
#         RP_vec*1000,cont_observed[7],'c--')
#         RP_vec*1000,cont_observed[8],'r--',
#         RP_vec*1000,cont_observed[9],'b--',
#         RP_vec*1000,cont_observed[10],'y--',
#         RP_vec*1000,cont_observed[11],'c--',)
plt.ylim(0,25)
plt.show()


#train_real=[]
#for i in np.arange(0,total_time,.0001):
#    if timestamps_sampled_real[idx]==i:
#        train_real.append(1)
#    else:
#        train_real.append(0)
#    idx+=1


#sanity check, isi_viols2 should be an empty array
#spkTs2 = np.where(trainA==1)[0] #spike times in ms
#isis2 = np.diff(spkTs2)
#isi_viols2 = np.where(isis2<rp)

#examples that work:
#true fr is 10, fr_cont=0, true rp = 2, crosses at 2. true rp = 4, crosses at 4. etc.
#same but with fr_cont=1, crosses earlier!



#example issues:
#same example as above but with fr_cont=.1, stays close to acceptable until true_rp. BUT, crosses earlier!
#why step functions? 
#fr_cont = 0 has a refractory period of 5?!


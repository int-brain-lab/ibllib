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

unit = units[681] #635 max spike rate 
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

percent_cont = np.arange(10,11,1)# np.arange(0,15,2.5)


nbt=50

cont_observed = np.empty([nbt,len(percent_cont),len(RP_vec)])

cont_acceptable=np.empty([len(RP_vec)])
rpidx=0
for rp in RP_vec:
    cont_acceptable[rpidx] = max_acceptable_cont(fr_sampled,rp,total_time,fr_sampled*.1,.1)
    rpidx+=1


for bt in range(nbt):
    pidx=0
    for p in percent_cont:
        #generate a contaminating neuron whos fr is p% of the main neuron
        isis_cont = random.choices(isis,k=(int(len(ts)*p/100)))
        ts_cont = np.cumsum(isis_cont)
        fr_cont = len(ts_cont)/total_time
        train_cont = np.zeros(int(np.ceil(total_time*10000))+1)
        timestamps_micros = [element * 10000 for element in ts_cont]
        train_cont[np.ceil(timestamps_micros).astype(int)]=1
        train_observed = train_real+train_cont
        spkTs_observed = np.where(train_observed==1)[0] #spike times in ms
        isis_observed = np.diff(spkTs_observed)
        
        rpidx=0
        for rp in RP_vec:
    #        cont_acceptable[rpidx] = max_acceptable_cont(FR_observed,rp,total_time,FR_observed*.1,.1)
    
            isi_viols_observed = np.where(isis_observed<(rp*10000))[0]
            cont_observed[bt][pidx][rpidx] = len(isi_viols_observed) #.append(len(isi_viols_observed))
            rpidx+=1
        pidx+=1
  

#plot distributions at one p_cont to check that it is poisson, with the correct mean
rp_choice = 0.0031
l = fr_sampled*.1*2*total_time*rp_choice
x = np.arange(0,20,1)
y = poisson.pmf(x, l)
plt.hist(cont_observed[:,0,np.where(RP_vec==rp_choice)[0]],density=True)
plt.plot(y)
#plt.hist(isis_observed/10000,50,density=True)
plt.ylim(0,.4)
plt.show()


#plot
plt.plot(RP_vec*1000,cont_acceptable,'k-', label='acceptable')
plt.plot(RP_vec*1000,cont_observed[0][0],'g--',label='10% contamination')
plt.legend(loc='upper left', shadow=False, fontsize='medium')
plt.xlabel('Refractory period length (ms)')
plt.ylabel('Number ISI violations')
plt.xticks(np.arange(0,RP_vec[-1]*1000,1))
plt.ylim(0,50)
plt.show()
  


  
#plt.plot(RP_vec*1000,cont_acceptable,'k-', label='acceptable')
#plt.plot(RP_vec*1000,cont_observed[0],'r--',label='0% contamination')
#plt.plot(RP_vec*1000,cont_observed[1],'b--', label='2.5% contamination')
#plt.plot(RP_vec*1000,cont_observed[2],'y--', label='5% contamination')
#plt.plot(RP_vec*1000,cont_observed[3],'c--', label='7.5% contamination')
#plt.plot( RP_vec*1000,cont_observed[4],'g--', label='10% contamination')
#plt.plot( RP_vec*1000,cont_observed[5],'m--', label='12.5% contamination')
#plt.legend(loc='upper left', shadow=False, fontsize='medium')
#plt.xlabel('Refractory period length (ms)')
#plt.ylabel('Number ISI violations')
#plt.xticks(np.arange(0,RP_vec[-1]*1000,1))
#plt.ylim(0,25)
#plt.show()
#
#
##check which percent contaminations pass
#for i in range(0,len(percent_cont)):
#    list3 = [1 for item1, item2 in zip(cont_observed[i], cont_acceptable) if item1 < item2]
#    if len(list3)>0:
#        print(percent_cont[i])
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:21:12 2020

@author: Steinmetz Lab User
"""

rec_duration=3600
thresh=.1

FR_vec=np.arange(0,6,.25)
RP_vec=np.arange(0.0001,0.006,.0005)
max_acc=np.empty([len(FR_vec),len(RP_vec)])

fridx=0
for FR in FR_vec:
    acceptableCont=.1*FR
    rpidx=0    
    for RP in RP_vec:
        max_acc[fridx][rpidx]=max_acceptable_cont(FR, RP, rec_duration,acceptableCont, thresh )
        rpidx+=1
    fridx+=1
    
    
plt.imshow(max_acc,origin='lower')
cb = plt.colorbar()
cb.set_label('number of allowed contaminations)')
plt.ylabel('Firing rate (spks/s)')
plt.xlabel('Refractory period length (ms)')
plt.yticks(np.arange(0,len(FR_vec)+1,5),FR_vec[(np.arange(1,len(FR_vec)+1,5))-1])
plt.xticks(np.arange(0,len(RP_vec+1),5), np.round(RP_vec[np.arange(0,len(RP_vec+1),5)]*1000,decimals=2))
plt.show()
        
        
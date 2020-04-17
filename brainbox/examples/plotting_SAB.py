# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:30:37 2020

@author: Steinmetz Lab User
"""

#plotting example units for SAB report

c0 = correlograms(ts,np.zeros(len(ts),dtype='int8'),cluster_ids=[0],bin_size=binSize/1000,sample_rate=20000,window_size=.02,symmetrize=True)
plt.bar(np.arange(-10,10.1,.25)+.04,c0[0,0],width = .25,facecolor = 'k',alpha=0.5)
plt.ylabel('Number of spikes (0.25 ms bins)')
plt.xlabel('Time (ms)')
plt.axvline(x=-2,color='k',lw=1)
plt.axvline(x=2,color='k',lw=1)
plt.xticks([-10,-5,0,5,10])
plt.show()
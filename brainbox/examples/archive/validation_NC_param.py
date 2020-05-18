# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:54:20 2020

@author: Steinmetz Lab User
"""

ks_dir = r'C:\Users\Steinmetz Lab User\Documents\Lab\SpikeSortingOutput\Hopkins_CortexLab\test_alf_dir1'
#not used: ks_dir = r'C:\Users\Steinmetz Lab User\Downloads\FlatIron\mainenlab\Subjects\ZM_2407\2019-11-05\003\alf\probe_00'
ks_dir = r'C:\Users\Steinmetz Lab User\Downloads\FlatIron\cortexlab\Subjects\KS022\2019-12-10\001\alf\probe00'
noisecutoff = pd.read_csv(Path(ks_dir,'NoiseCutoff.tsv'),sep='\t')
nc = noisecutoff['NoiseCutoff']


#for Hopkins, load old ks2 (or by-hand?) labels
ks2df = pd.read_csv(Path(ks_dir,'cluster_group.tsv'),sep='\t')
k = ks2df['group']
kind = ks2df['cluster_id']
xg=kind[np.where(k=='good')[0]] #second column of this is cluster ids that are good
kg=xg.tolist()

#for Hopkins
man_ind = kg[kg.index(732):kg.index(1129)+1]
man_val = [1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,\
                                 0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,0,\
                                 0,1,0,0, 1,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,\
                                 1,1,1,0,1,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,1,1,0,\
                                 1,1,0,1,1,1,1,1]
#for KS022
man_ind = np.arange(0,107,1)#kg[kg.index(0):kg.index(100)+1]
man_val = [1,1,0,0,0,0,0,1,1,0,1,0,1,0,1,0,0,0,0,0,np.nan,0,0,np.nan,0,1,np.nan,0,1,1,0,1,\
           1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,1,1,0,0,\
               1,0,0,0,1,0,0,0,0,1,np.nan,np.nan,0,0,0,0,0,0,0,0,1,0,1,np.nan,np.nan,0,0,0,0,\
                   0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,1,0]
    
    
    
ind_g = [i for i, x in enumerate(man_val) if x == 1]
gg = [man_ind[i] for i in ind_g]
nc_g = [nc[i] for i in gg]


ind_b = [i for i, x in enumerate(man_val) if x == 0]
bb = [man_ind[i] for i in ind_b]
nc_b = [nc[i] for i in bb]


bins_list =np.arange(-10,300,2.5) 
plt.hist(nc_g,bins = bins_list,color='b',alpha=.5)
plt.hist(nc_b,bins = bins_list,color='r',alpha=.5)
# plt.xscale('log')
plt.xlim((-10,300))
plt.show()


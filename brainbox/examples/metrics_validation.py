# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:15:28 2020

@author: Steinmetz Lab User
"""

#compare labels

import pandas as pd


#load metrics labels
labelsdf = pd.read_csv(Path(ks_dir,'label.tsv'),sep='\t')
l = labelsdf['label']

noisecutoff = pd.read_csv(Path(ks_dir,'NoiseCutoff.tsv'),sep='\t')
nc = noisecutoff['NoiseCutoff']

rfpv = pd.read_csv(Path(ks_dir,'RefPViol.tsv'),sep='\t')
rfp = rfpv['RefPViol']

#load old ks2 (or by-hand?) labels
ks2df = pd.read_csv(Path(ks_dir,'cluster_group.tsv'),sep='\t')
k = ks2df['group']
kind = ks2df['cluster_id']


# gg=np.intersect1d(np.where(k=='good'),np.where(l==1))
# gb = np.intersect1d(np.where(k=='good'),np.where(l==0))
# bg= np.intersect1d(np.where(k!='good'),np.where(l==1))
# bb=np.intersect1d(np.where(k!='good'),np.where(l==0))



# lind_kgood = l[np.where(k=='good')[0]]
# lind_kbad = l[np.where(k=='bad')[0]]

# gg =lind


xg=kind[np.where(k=='good')[0]] #second column of this is cluster ids that are good
kg=xg.tolist()
kg[0] #sanity check, this should be 13, the first cluster id that is good.
xmua = kind[np.where(k=='mua')[0]] # or k== 'noise')]
xnoise = kind[np.where(k=='noise')[0]]
kmua=xmua.tolist()
knoise = xnoise.tolist()
xbad = kmua+knoise

l1 = np.where(l==1)[0]
l0 = np.where(l==0)[0]

interg1 = np.intersect1d(l1,kg) #these are the units that both labeled GOOD
print("number of both good = ", len(interg1))
interg0 = np.intersect1d(l0,kg)
print("number of ks2 good but label bad = ", len(interg0))


interb1 = np.intersect1d(l1,xbad)
print("number of ks2 bad (noise and mua) but label good = ", len(interb1))

interb0 = np.intersect1d(l0,xbad)
print("number of both bad = ", len(interb0))



#if mismatch, why?
rfpval = rfpv['RefPViol'].tolist()
rfpval[interg0]
np.array(rfpval)[interg0.astype(int)]

np.array(nc)[interg0.astype(int)]

#there is something going on -- why are tehre 1209 cluster_ids and only 959 labels?
#last cluster id for which tehre is a ks2/nick label is 954 (this is ind 542. )
#then ind 543-647 correspond to 967 to 1209. those are missing!


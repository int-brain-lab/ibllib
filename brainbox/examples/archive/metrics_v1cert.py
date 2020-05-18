# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:36:49 2020

@author: Steinmetz Lab User
"""

#run metrics on all v1 certificaiton sessions
table_path = r"C:\Users\Steinmetz Lab User\int-brain-lab"

from oneibl.one import ONE
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
one = ONE()
#load metrics labels
eids=[]
probe_names=[]
data_path=[]
cert_sessions = pd.read_csv(Path(table_path,'summary_table_V1Certification.csv'),sep=',')


dtypes = ['clusters.amps',
          'clusters.channels',
          'clusters.depths',
          'clusters.metrics',
          'clusters.peakToTrough',
          'clusters.probes',
          'clusters.uuids',
          'clusters.waveforms',
          'clusters.waveformsChannels',
          'spikes.amps',
          'spikes.clusters',
          'spikes.depths',
          'spikes.samples',
          'spikes.templates',
          'spikes.times']
for ind in cert_sessions.index:
    lab = cert_sessions['lab'][ind]
    subject = cert_sessions['subject'][ind].split('/')[0]
    date = cert_sessions['subject'][ind].split('/')[1]
    probe_name = cert_sessions['probe_name'][ind]
    eid,sinfo = one.search(datasets='spikes.times',task_protocol='certification',details=True,lab=lab,subject=subject,date_range=[date, date])
    eids.append(eid[0])
    probe_names.append(probe_name)
    one.load(eid, dataset_types=dtypes,download_only=True)
    # print(lab, subject, date, probe_name)  #sanity check





# for eid, info in zip(eids, sinfo):
#     print(info)
#     if info['subject'] == 'cer-5':
#         file_paths = one.load(eid, download_only=True)


# eids = ['5cf2b2b7-1a88-40cd-adfc-f4a031ff7412',  'a3df91c8-52a6-4afa-957b-3479a7d0897c' ]
# probe_names = ['probe_right', 'probe00']




from metrics_new import gen_metrics_labels
import time

data = pd.DataFrame([])
for eid,probe_name,ses in zip(eids,probe_names,cert_sessions.index):
    start = time.time()
    lab = cert_sessions['lab'][ses]
    subject = cert_sessions['subject'][ses].split('/')[0]
    date = cert_sessions['subject'][ses].split('/')[1]
    try:
        numpass, numpassRP, numpassAC, ntot = gen_metrics_labels(eid,probe_name)
            
        try:
            data = data.append(pd.DataFrame({'lab': lab, 'subject': subject, 'date': date,'probe_name': probe_name,'numpass': numpass,'numpassRP': numpassRP, 'numpassAC': numpassAC, 'numtot': numtot}, index=[0]), ignore_index=True)
            # data['lab'][ind].replace({'wittenlab': lab}, inplace=True)
            # data['subject'][ind].replace({'lic3': subject}, inplace=True)
            # data['date'][ind].replace({'2019-08-27': date}, inplace=True)
        except:
            print('cannot add data to df')


    except FileNotFoundError:
        print('File not found')
    except:
        print('some other error!')
    print(time.time() - start)

        
        
data.to_csv(Path(table_path, 'v1_cert_metrics.tsv'), sep='\t')
#now plot this data
v1_metrics = pd.read_csv(Path(table_path,'v1_cert_metrics.tsv'), sep='\t',header=0)
numpass_ses=[]
for ind in v1_metrics.index:
    numpass = v1_metrics['numpass'][ind]
    numpass_ses.append(numpass)
    lab = v1_metrics['lab'][ind]
    subject = v1_metrics['subject'][ind]
    date = v1_metrics['subject'][ind]

v1_metrics.plot.barh(x='subject', y='numpass', rot=0, width = .5,figsize=(10,10))
plt.axvline(x=20)
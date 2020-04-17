# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:18:23 2020

@author: Steinmetz Lab User
"""

#pull out amplitudes


#test eid, probe_name
eid = '49a65ab1-890a-41c6-aef3-5f746affb010'
probe_name = 'probe01'



    eid,sinfo = one.search(datasets='_phy_spikes_subset.waveforms',task_protocol='certification',details=True,lab=lab,subject=subject,date_range=[date, date])

_phy_spikes_subset.waveforms.npy 


dtypes = ['_phy_spikes_subset.waveforms']
one.load(eid, dataset_types=dtypes,download_only=True)


from ibllib.ephys.ephysqc import phy_model_from_ks2_path
from pathlib import Path
m = phy_model_from_ks2_path(Path(r"C:\Users\Steinmetz Lab User\Downloads\FlatIron\wittenlab\Subjects\lic3\2019-08-27\002\alf\probe00"))
true_amps = m.get_amplitudes_true()
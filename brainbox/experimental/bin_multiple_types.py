from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from oneibl.one import ONE
from ibllib.misc import bincount2D
import ibllib.io.alf as ioalf
import ibllib.plots as iblplt
import rastermap
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma
plt.ion()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# get the data from flatiron and the current folder
one = ONE()
eid = one.search(subject='ZM_1150', date='2019-05-07', number=1)
D = one.load(eid[0], clobber=False, download_only=True)
session_path = Path(D.local_path[0]).parent

# load objects
spikes = ioalf.load_object(session_path, 'spikes')
trials = ioalf.load_object(session_path, '_ibl_trials')
wheel = ioalf.load_object(session_path, '_ibl_wheel')

def bin_types(spikes,trials,wheel):

 T_BIN = 0.2 #[sec]

 #TO GET MEAN: bincount2D(..., weight=positions) / bincount2D(..., weight=None)
 
 reward_times = trials['feedback_times'][trials['feedbackType'] == 1]
 trial_start_times = trials['intervals'][:, 0]
 #trial_end_times = trials['intervals'][:, 1] #not working as there are nans

 # compute raster map as a function of cluster number
 R1, times1, _ = bincount2D(spikes['times'], spikes['clusters'], T_BIN, weights=spikes['amps']) 
 R2, times2, _ = bincount2D(reward_times, np.array([0]*len(reward_times)), T_BIN) 
 R3, times3, _ = bincount2D(trial_start_times, np.array([0]*len(trial_start_times)), T_BIN) 
 R4, times4, _ = bincount2D(wheel['times'], np.array([0]*len(wheel['times'])), T_BIN, weights=wheel['position'])
 R5, times5, _ = bincount2D(wheel['times'], np.array([0]*len(wheel['times'])), T_BIN, weights=wheel['velocity'])
 #R6, times6, _ = bincount2D(trial_end_times, np.array([0]*len(trial_end_times)), T_BIN)  

 start =max([x for x in [times1[0],times2[0],times3[0],times4[0],times5[0]]])
 stop = min([x for x in [times1[-1],times2[-1],times3[-1],times4[-1],times5[-1]]])
 time_points=np.linspace(start,stop,int((stop-start)/T_BIN))

 binned_data={} 
 binned_data['wheel_position']=np.interp(time_points,wheel['times'],wheel['position'])
 binned_data['wheel_velocity']=np.interp(time_points,wheel['times'],wheel['velocity'])
 binned_data['summed_spike_amps']=R1[:,find_nearest(times1,start):find_nearest(times1,stop)]
 binned_data['reward_event']=R2[0,find_nearest(times2,start):find_nearest(times2,stop)]
 binned_data['trial_start_event']=R3[0,find_nearest(times3,start):find_nearest(times3,stop)]
 #binned_data['trial_end_event']=R6[0,find_nearest(times6,start):find_nearest(times6,stop)]

 #np.vstack([R1,R2,R3,R4])

 return binned_data

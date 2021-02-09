from oneibl.one import ONE
from ibllib.io.video import get_video_meta, get_video_frames_preload
import numpy as np
from pathlib import Path
import cv2
import alf.io
import matplotlib
from random import sample
import time


def stream_save_labeled_frames(eid, video_type):

    startTime = time.time()
    '''
    For a given eid and camera type, stream
    sample frames, print DLC labels on them
    and save
    '''

    # eid = '5522ac4b-0e41-4c53-836a-aaa17e82b9eb'
    # video_type = 'left'

    n_frames = 5  # sample 5 random frames

    save_images_folder = '/home/mic/DLC_QC/example_frames/'
    one = ONE()
    info = '_'.join(np.array(str(one.path_from_eid(eid)).split('/'))[[5,7,8]])
    print(info, video_type)

    r = one.list(eid, 'dataset_types')

    dtypes_DLC = ['_ibl_rightCamera.times.npy',
                  '_ibl_leftCamera.times.npy',
                  '_ibl_bodyCamera.times.npy',
                  '_iblrig_leftCamera.raw.mp4',
                  '_iblrig_rightCamera.raw.mp4',
                  '_iblrig_bodyCamera.raw.mp4',
                  '_ibl_leftCamera.dlc.pqt',
                  '_ibl_rightCamera.dlc.pqt',
                  '_ibl_bodyCamera.dlc.pqt']

    dtype_names = [x['name'] for x in r]

    assert all([i in dtype_names for i in dtypes_DLC]
               ), 'For this eid, not all data available'


    D = one.load(eid,
                 dataset_types=['camera.times', 'camera.dlc'],
                 dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'

    cam0 = alf.io.load_object(
        alf_path,
        '%sCamera' %
        video_type,
        namespace='ibl')

    Times = cam0['times']

    cam = cam0['dlc']
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])


 
    if video_type != 'body':
        d = list(points)
        d.remove('tube_top')
        d.remove('tube_bottom')
        points = np.array(d)    
    
    # stream frames
    recs = [x for x in r if f'{video_type}Camera.raw.mp4' in x['name']][0]['file_records']
    video_path = [x['data_url'] for x in recs if x['data_url'] is not None][0]
    vid_meta = get_video_meta(video_path)

    frame_idx = sample(range(vid_meta['length']), n_frames)
    print('frame indices:', frame_idx) 
    frames = get_video_frames_preload(video_path,
                                      frame_idx,
                                      mask=np.s_[:, :, 0])
    size = [vid_meta['width'], vid_meta['height']]
    #return XYs, frames    
    
    x0 = 0
    x1 = size[0]
    y0 = 0
    y1 = size[1]
    if video_type == 'left':
        dot_s = 10  # [px] for painting DLC dots
    else:
        dot_s = 5

    # writing stuff on frames
    font = cv2.FONT_HERSHEY_SIMPLEX

    if video_type == 'left':
        bottomLeftCornerOfText = (20, 1000)
        fontScale = 4
    else:
        bottomLeftCornerOfText = (10, 500)
        fontScale = 2

    lineType = 2

    # assign a color to each DLC point (now: all points red)
    cmap = matplotlib.cm.get_cmap('Set1')
    CR = np.arange(len(points)) / len(points)

    block = np.ones((2 * dot_s, 2 * dot_s, 3))

    k = 0
    for frame in frames:

        gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


        # print session info
        fontColor = (255, 255, 255)
        cv2.putText(gray,
                    info,
                    bottomLeftCornerOfText,
                    font,
                    fontScale / 4,
                    fontColor,
                    lineType)
                    
        # print time                    
        Time = round(Times[frame_idx[k]], 3)
        a, b = bottomLeftCornerOfText
        bottomLeftCornerOfText0 = (int(a * 10 + b / 2), b)



        a, b = bottomLeftCornerOfText
        bottomLeftCornerOfText0 = (int(a * 10 + b / 2), b)
        cv2.putText(gray,
                    '  time: ' + str(Time),
                    bottomLeftCornerOfText0,
                    font,
                    fontScale / 2,
                    fontColor,
                    lineType)



        # print DLC dots
        ll = 0
        for point in points:

            # Put point color legend
            fontColor = (np.array([cmap(CR[ll])]) * 255)[0][:3]
            a, b = bottomLeftCornerOfText
            if video_type == 'right':
                bottomLeftCornerOfText2 = (a, a * 2 * (1 + ll))
            else:
                bottomLeftCornerOfText2 = (b, a * 2 * (1 + ll))
            fontScale2 = fontScale / 4
            cv2.putText(gray, point,
                        bottomLeftCornerOfText2,
                        font,
                        fontScale2,
                        fontColor,
                        lineType)

            X0 = XYs[point][0][frame_idx[k]]
            Y0 = XYs[point][1][frame_idx[k]]

            X = Y0
            Y = X0

            #print(point,X,Y)
            if not np.isnan(X) and not np.isnan(Y):
                try:
                    col = (np.array([cmap(CR[ll])]) * 255)[0][:3]
                    # col = np.array([0, 0, 255]) # all points red
                    X = X.astype(int)
                    Y = Y.astype(int)

                    uu = block * col
                    gray[X - dot_s:X + dot_s, Y - dot_s:Y + dot_s] = uu

                except Exception as e:
                    print('frame', frame_idx[k])
                    print(e)
            ll += 1

        gray = gray[y0:y1, x0:x1]
        # cv2.imshow('frame', gray)
        cv2.imwrite(f'{save_images_folder}{eid}_frame_{frame_idx[k]}.png',
                    gray)
        cv2.waitKey(1)
        k += 1

    print(f'{n_frames} frames done in', np.round(time.time() - startTime)) numpy as np
from oneibl.one import ONE
from pathlib import Path
from brainbox.io.one import load_channel_locations 
from brainbox.processing import bincount2D
from collections import Counter
import alf.io
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy
import pandas as pd
from ibllib.atlas import regions_from_allen_csv
import random
#from mpl_toolkits.mplot3d import Axes3D
import ibllib.atlas as atlas
from sklearn.decomposition import PCA
from scipy.stats import zscore

plt.ion()
T_BIN = 0.04 # was 0.04 (0.01 - 0.3)


# Rylan's sessions: 
#'5522ac4b-0e41-4c53-836a-aaa17e82b9eb' 
#'e349a2e7-50a3-47ca-bc45-20d1899854ec'



def get_sessions0():
    one = ONE()
    traj_traced = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         django='probe_insertion__session__project__name__'
                                'icontains,ibl_neuropixel_brainwide_01,'
                                'probe_insertion__session__qc__lt,50,'
                                'probe_insertion__session__extended_qc__behavior,1,'
                                'probe_insertion__json__extended_qc__tracing_exists,True,'
                                '~probe_insertion__session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_response_feedback_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_response_stimFreeze_delays__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_n_trial_events__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_reward_volumes__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_reward_volume_set__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
                                '~probe_insertion__session__extended_qc___task_audio_pre_trial__lt,0.9')
    eids = [[x['session']['id'],x['probe_name']] for x in traj_traced]
    return eids
    
    
def get_sessions1():
    '''
    updated alignement
    ''' 
    one = ONE()
    sessions = one.alyx.rest('insertions', 'list', django='json__extended_qc__alignment_resolved,True')
    return [[x['session_info']['id'],x['name']] for x in sessions] 


def get_sessions2():

    '''
    combined sessions, passing behavioral qc
    and (or --> change to union) histology aligned
    '''
    s0 = get_sessions0()
    s1 = get_sessions1()
    s0_ = set(['_'.join(x) for x in s0])
    s1_ = set(['_'.join(x) for x in s1])
    r=list(s0_.intersection(s1_))
    return [x.split('_') for x in r]



def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
 
 
def get_acronyms_per_insertion(eid, probe):

    T_BIN = 0.5
    one = ONE()
    dataset_types = ['spikes.times',
                     'spikes.clusters',
                     'clusters.channels']                     
                     
    D = one.load(eid, dataset_types = dataset_types, dclass_output=True)
    local_path = one.path_from_eid(eid)  
    alf_path = local_path / 'alf'     
    probe_path = alf_path / probe                    
                     
    # bin spikes
    spikes = alf.io.load_object(probe_path, 'spikes')
    R, times, Clusters = bincount2D(
        spikes['times'], spikes['clusters'], T_BIN)                     
                     
                     
    # Get for each cluster the location acronym
    clusters = alf.io.load_object(probe_path, 'clusters')        
    cluster_chans = clusters['channels'][Clusters]
    els = load_channel_locations(eid, one=one)   
    acronyms = els[probe]['acronym'][cluster_chans]

    return acronyms 
 
               
    
    
def get_full_stim_intervals(eid):

    '''
    getting trial numbers, stim present interval
    '''
     
    one = ONE()
    trials = one.load_object(eid, 'trials')
    d = {} # dictionary, trial number and still interval    
    
    for tr in range(len(trials['intervals'])):

        b = trials['goCue_times'][tr] - 1 
        c = trials['feedback_times'][tr]
        ch = trials['choice'][tr]
        pl = trials['probabilityLeft'][tr]
        
        
        if np.isnan(c):
            #print(f'feedback time is nan for trial {tr} and eid {eid}')
            continue        
              
        if c-b>5: # discard too long trials
            continue     
                    
        if np.isnan(trials['contrastLeft'][tr]):
            cont = trials['contrastRight'][tr]            
            side = 0
        else:   
            cont = trials['contrastLeft'][tr]         
            side = 1              
                
        d[tr] = [b,c-b, cont, side, ch, pl]
    print(f"cut {len(d)} of {len(trials['intervals'])} full trials segments")
    return d    
    
    
  
   
def get_PCA(D,acronyms,reg):
    
    '''
    For each region, reduce to 3 dims
    '''    
    

    obs, neurons = D.shape
    c = np.zeros((3,obs))
    if reg == 'whole_probe':
        data = D
        pca = PCA()
        pca.fit(data)
        for i in range(3):        
            c[i] = zscore(np.matmul(data,pca.components_[i]),nan_policy='omit')         
    

    
    else:    
        if reg == 'VIS':
            a2 = []
            for a in acronyms:
                if 'VIS' in a:
                    a2.append('VIS')
                else:
                    a2.append(a)   
            acronyms = np.array(a2)       
        
        m_ask = acronyms == reg
        data = D[:,m_ask]
        print(reg, np.shape(data))
        pca = PCA()
        pca.fit(data)
        for i in range(3):        
            c[i] = zscore(np.matmul(data,pca.components_[i]),nan_policy='omit')

       
    return c 
  


def bin_plot_PCA(eid,probe, reg):

    Full = True
    reaction_time_type = 'full'
    
    one = ONE()
    #probe = 'probe01'
    
    dataset_types = ['spikes.times',
                     'spikes.depths',
                     'spikes.clusters',
                     'clusters.channels',                   
                     'trials.intervals']
    
    D = one.load(eid, dataset_types = dataset_types, dclass_output=True)
    local_path = one.path_from_eid(eid)  
    alf_path = local_path / 'alf'     
    probe_path = alf_path / probe
    
    spikes = alf.io.load_object(probe_path, 'spikes')
      
 
    # bin spikes
    R, times, Clusters = bincount2D(
        spikes['times'], spikes['clusters'], T_BIN)
   
   
    # Get for each cluster the location x,y,z
    clusters = alf.io.load_object(probe_path, 'clusters')        
    cluster_chans = clusters['channels'][Clusters]      
    els = load_channel_locations(eid, one=one)   
    acronyms = els[probe]['acronym'][cluster_chans]   


    D = R.T    
    
    print(D.shape)
    
    c = get_PCA(D,acronyms,reg)

    
    if reaction_time_type == 'full':
        d = get_full_stim_intervals(eid)
    if reaction_time_type == 'still':    
        d = get_still_intervals(eid)
    if reaction_time_type == 'constant':   
        d = constant_reaction_time(eid, 0.150)

    # Get lists of trials
    pcs = []
    times_ = []
    contrasts = []
    sides = []
    pl = []
    ch = []
    for i in d:
        start_idx = find_nearest(times,d[i][0])
        end_idx = start_idx + int(d[i][1]/T_BIN)   
        if end_idx < start_idx:
            print(i,'end_idx < start_idx')   
        pcs.append(c[:,start_idx:end_idx])
        times_.append(times[start_idx:end_idx])
        contrasts.append(d[i][2])
        sides.append(d[i][3])
        pl.append(d[i][5])
        ch.append(d[i][4])
    
    #return pcs, pl
    #return pcs, times_, acronyms, contrasts, sides
    plot_state_space3d_multi(pcs,pl)   
    print(Counter(sides))
    plt.title(f'{eid} \n {probe}, {reg}, probability left in color')
 


def bin_PCA_double(eid):

    '''
    combine both probes into binned activity
    then reduce dimenions via PCA to 3,
    then keep only bins in trials,
    keep block id info for each bin
    '''
    
    plt.ioff()
    
    ret_dat = False
    
    one = ONE()    
    dataset_types = ['spikes.times','spikes.clusters']
    
    D = one.load(eid, dataset_types = dataset_types, dclass_output=True)
    local_path = one.path_from_eid(eid)  
    alf_path = local_path / 'alf'  
    
    sks = []
    for probe in ['probe00','probe01']:    
        probe_path = alf_path / probe    
        spikes = alf.io.load_object(probe_path, 'spikes')
        sks.append(spikes)
     
    # add max cluster of p0 to p1, then concat, sort 
    max_cl0 = max(sks[0]['clusters'])
    sks[1]['clusters'] = sks[1]['clusters'] + max_cl0
     
    times_both = np.concatenate([sks[0]['times'],sks[1]['times']])
    clusters_both = np.concatenate([sks[0]['clusters'],sks[1]['clusters']])
    
    t_sorted = np.sort(times_both)
    c_ordered = clusters_both[np.argsort(times_both)] 

    R, times, _ = bincount2D(t_sorted, c_ordered, T_BIN)  

    D = R.T    
    
    
    obs, n_clus = D.shape
    print(f'duration: {np.round(obs*T_BIN,0)} sec = {np.round(obs*T_BIN/60,2)} min; n_clus = {n_clus}; T_BIN = {T_BIN}')    
    
    c = np.zeros((3,obs))

    pca = PCA()
    pca.fit(D)
    for i in range(3):        
        c[i] = zscore(np.matmul(D,pca.components_[i]),nan_policy='omit')          

    if ret_dat:
        return D, c, times


    # only keep time bins in trials and get trial info
    d = get_full_stim_intervals(eid)

    # Get lists of trials
    pcs = []
    times_ = []
    contrasts = []
    sides = []
    probleft = []
    choice = []
    for i in d:
        start_idx = find_nearest(times,d[i][0])
        end_idx = start_idx + int(d[i][1]/T_BIN)   
        if end_idx < start_idx:
            print(i,'end_idx < start_idx')   
        pcs.append(c[:,start_idx:end_idx])
        times_.append(times[start_idx:end_idx])
        contrasts.append(d[i][2])
        sides.append(d[i][3])
        probleft.append(d[i][5])
        choice.append(d[i][4])
    
    return pcs,probleft, sides, choice 
#    return pcs, times_, contrasts, sides, probleft, choice    
    
    plot_state_space3d_multi(pcs,probleft, sides)   
    plt.title(f'{eid} \n both probes combined, probabilityLeft in color \n including 1 sec before goCue, T_BIN = {T_BIN} sec')
    plt.tight_layout()
    plt.savefig(f'{eid}.png')    
    plt.close()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

  

def plot_state_space3d_multi(pcs,trial_colors): #, sides):  
  
    dim2 = False
    filter_side = False    
  
    cm = plt.get_cmap("cool")    #Accent , cool
    
    if dim2:
        fig,ax = plt.subplots()
    else:            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')   

    Bc = []
    xs = []
    ys = []
    zs = []
    
    k = 0
    
    kk = 0
    for i in range(len(pcs)):       
    
        if filter_side:
            if sides[i]==1:
                continue               
               
        tr = pcs[i]    
        cols = np.ones(len(tr[0]))*trial_colors[i]
        #cols = np.ones(len(tr[0]))*i  # by trial number
        #cols = np.arange(len(tr[0])) # by time in trial


        # kk is trial number in block
#        cps = np.where(np.diff(trial_colors)!=0)[0]        
#        if (i+1) in cps:
#            kk = 0 
#        
#        cols = np.ones(len(tr[0]))*kk
        
        kk+=1
        
        Bc.append(cols)                
        xs.append(tr[0])
        ys.append(tr[1])        
        zs.append(tr[2])       
        k+=1
        
    print(k, 'trials used')    
    
    Bflat = [item for sublist in Bc for item in sublist]
    xflat = [item for sublist in xs for item in sublist]
    yflat = [item for sublist in ys for item in sublist]
    zflat = [item for sublist in zs for item in sublist]
    
    if dim2:
        p = ax.scatter(yflat,zflat,c=Bflat,cmap=cm, s=5)
        ax.set_xlabel('pc 2')
        ax.set_ylabel('pc 3')           
        
    else:                
        p = ax.scatter(xflat,yflat,zflat,c=Bflat,cmap=cm, depthshade=False, s=1)
        set_axes_equal(ax)    
        ax.set_xlabel('pc 1')
        ax.set_ylabel('pc 2')          
        ax.set_zlabel('pc 3')
         
    fig.colorbar(p) 

    #return np.array([xflat,yflat,zflat,Bflat]) 
 
 
 
 
def plot_state_space3d_single(B):  
  
    #B = get_PCA(D,acronyms,reg)
  
    cm = plt.get_cmap("Accent")    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
  
    #cols = range(len(B[0]))
    cols = B[3]
  
    p = ax.scatter(B[0],B[1],B[2],c=cols,cmap=cm, depthshade=False, s=1)
    
    ax.set_xlabel('pc 1')
    ax.set_ylabel('pc 2')
    ax.set_zlabel('pc 3')    
    set_axes_equal(ax)
    fig.colorbar(p)    
 
def plot_state_space2d_single(B):  
  
    
    #B = get_PCA(D,acronyms,reg)
  
    cm = plt.get_cmap("Reds")    
    fig, ax = plt.subplots()
  
    p = ax.scatter(B[0],B[2],c=range(len(B[0])),cmap=cm, s=1)
    
    ax.set_xlabel('pc 1')
    ax.set_ylabel('pc 3')
 
 
def plot_state_space2d_multi(neural, acronyms):  
  
    #B = get_PCA(D,acronyms,reg)
  
    cm = plt.get_cmap("Blues")    
    fig, ax = plt.subplots()
    
    for tr in range(len(neural)):
        B = get_PCA(neural[tr],acronyms,'whole_probe')            
        ax.scatter(B[0],B[1],c=range(len(B[0])),cmap=cm, s=1)
    
    ax.set_xlabel('pc 1')
    ax.set_ylabel('pc 2') 
 
 
#get_PCA(D,acronyms,reg  
#ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)  
  
# The one with many Visp regions  
#In [8]: sess.index(['dda5fc59-f09a-4256-9fb5-66c67667a466', 'probe00'])
#Out[8]: 43
  
  
def plot_raster_PCs_trial(eid, D, c, times):

    #D, c, times = bin_PCA_double(eid)
    
    
    one = ONE()    
    trials = one.load_object(eid, 'trials')
 
    duration = 10# in trials
   
    start_trial = 10
    end_trial = start_trial + duration
    
    start_time = trials['intervals'][start_trial][0]
    end_time = trials['intervals'][end_trial][1]
    
    start_idx = find_nearest(times,start_time)
    end_idx = find_nearest(times,end_time)
    
    if end_idx < start_idx:
        print(i,'end_idx < start_idx')
        return   
        
    pcs = c[:,start_idx:end_idx]
    dat = D[start_idx:end_idx]    
    times_ = times[start_idx:end_idx]
    
    fig = plt.figure()
    ax0 = plt.subplot(2,1,1)

    obs, n_clus = dat.shape      
    plt.imshow(dat.T, aspect='auto',
               cmap='binary', vmax=T_BIN / 0.01 / 4,
               extent=np.r_[times[[start_idx, end_idx]],
               np.arange(n_clus)[[0,-1]]], origin='lower', axes = ax0)  

    plt.ylabel('cluster id') 
    plt.title(f'{eid} \n T_BIN = {T_BIN}, both probes combined')
    
    k = 0
    for x in trials['goCue_times'][start_trial:end_trial]:
        ax0.axvline(x=x, linewidth=0.5, linestyle='--', c='r', label="goCue_times" if k == 0 else "")
        k+=1
                
    k = 0                
    for x in trials['feedback_times'][start_trial:end_trial]:
        ax0.axvline(x=x, linewidth=0.5, linestyle='--', c='y',                
 label="feedback_times" if k == 0 else "")
        k+=1
        
    plt.legend()         

    ax1 = plt.subplot(2,1,2, sharex = ax0)
    for i in range(len(pcs)):
        plt.plot(times_, pcs[i], linestyle='',marker='o', label = f'pc {i+1}')
      
    plt.xlabel('time [sec]')
    plt.ylabel(' z-scored pcs [a.u.]')    
    plt.legend()    

    
def block_plot(trials):
    fig, ax = plt.subplots()   
    ax.set_yticks([0.2,0.5,0.8])
    plt.plot(trials['probabilityLeft'],linestyle='', marker='|', markersize=1)
    plt.xlabel('trial number')
    plt.ylabel('probabilityLeft')




    


  
  
  
  
  
  
  
  
    
    
    
    
    
    
    

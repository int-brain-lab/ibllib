# -*- coding: utf-8 -*-
r"""
Created on Wed Apr  1 21:46:59 2020

@author: Noam Roth
Example:
    
  # define eid and probe    
>>>eid = '5cf2b2b7-1a88-40cd-adfc-f4a031ff7412'
>>>probe_name = 'probe_right'
  # run gen_metrics_labels
>>>from brainbox.examples.metrics_new import gen_metrics_labels
>>>gen_metrics_labels(eid,probe_name)

"""


import time
import os
from oneibl.one import ONE
from pathlib import Path
import numpy as np
import alf.io as aio
import matplotlib.pyplot as plt
import brainbox as bb
from brainbox.examples import max_acceptable_isi_viol_2, defined_metrics
from brainbox.examples.max_acceptable_isi_viol_2 import max_acceptable_cont_2
from phylib.stats import correlograms
import pandas as pd
from brainbox.metrics import metrics
from brainbox.examples.defined_metrics import FP_RP, noise_cutoff, peak_to_peak_amp




def gen_metrics_labels(eid,probe_name):

    one=ONE()
    ses_path=one.path_from_eid(eid)    
    alf_probe_dir = os.path.join(ses_path, 'alf', probe_name)
    ks_dir = alf_probe_dir
    spks_b = aio.load_object(alf_probe_dir, 'spikes')  
    units_b = bb.processing.get_units_bunch(spks_b)
    units = list(units_b.amps.keys())
    lengths_samples = [len(v) for k, v in units_b.samples.items()]
    units_nonzeros=[i for i,d in enumerate(lengths_samples) if d>0]
    n_units = len(units_nonzeros) #only compute metrics for units with no samples
    
    
    #for cases where raw data is available locally: 
    ephys_file_dir = os.path.join(ses_path, 'raw_ephys_data', probe_name)
    ephys_file = os.path.join(ses_path, 'raw_ephys_data', probe_name,'_iblrig_ephysData.raw_g0_t0.imec.ap.cbin')
    #create params.py file
    params_file = os.path.join(ks_dir,'params.py')
    if os.path.exists(ephys_file) and not os.path.exists(params_file):
        f = open(params_file,"w+")
        f.write('dat_path = ' + 'r"' + ephys_file + '"\n' +  '''n_channels_dat = 385
        dtype = 'int16'
        offset = 0
        sample_rate = 30000
        hp_filtered = False
        uidx=0''' )
        f.close()
                
    # Initialize metrics
    cum_amp_drift = np.full((n_units,), np.nan)
    cum_depth_drift = np.full((n_units,), np.nan)
    cv_amp = np.full((n_units,), np.nan)
    cv_fr = np.full((n_units,), np.nan)
    frac_isi_viol = np.full((n_units,), np.nan)
    frac_missing_spks = np.full((n_units,), np.nan)
    fp_est = np.full((n_units,), np.nan)
    pres_ratio = np.full((n_units,), np.nan)
    pres_ratio_std = np.full((n_units,), np.nan)
    ptp_sigma = np.full((n_units,), np.nan)

    units_missing_metrics = set()
    label=np.empty([len(units)])
    RefPViol = np.empty([len(units)])
    NoiseCutoff = np.empty([len(units)])
    MeanAmpTrue = np.empty([len(units)])
    
    for idx,unit in enumerate(units_nonzeros):
        if unit == units_nonzeros[0]:
            t0 = time.perf_counter()  # used for computation time estimate
        
        print('computing metrics for unit ' + str(unit) + '...' )

        #load relevant data for unit
        ts = units_b['times'][str(unit)]
        amps = units_b['amps'][str(unit)]
        samples = units_b['samples'][str(unit)]
        depths = units_b['depths'][str(unit)]

        
        RefPViol[idx] = FP_RP(ts)
        NoiseCutoff[idx] = noise_cutoff(amps,quartile_length=.25)
        
        #create 'label' based on RPviol,NoiseCutoff, and MeanAmp
        if len(samples>50): #only compute mean amplitude for units with more than 50 samples
            try:
                MeanAmpTrue[int(unit)] = peak_to_peak_amp(ephys_file, samples, nsamps=20)
    
                if (FP_RP(ts) and noise_cutoff(amps,quartile_length=.25)<20 and MeanAmpTrue[int(unit)]>50) : 
                    label[idx] = 1
                else:
                    label[idx] = 0
            except:
                if (FP_RP(ts) and noise_cutoff(amps,quartile_length=.25)<20) : 
                    label[idx] = 1
                else:
                    label[idx] = 0
                
        else: #no ephys file, do not include true mean amps
            if (FP_RP(ts) and noise_cutoff(amps,quartile_length=.25)<20) : 
                label[idx] = 1
            else:
                label[idx] = 0



        #now compute additional metrics that label does not depend on: 
            
        # Cumulative drift of spike amplitudes, normalized by total number of spikes.
        try:
            cum_amp_drift[idx] = bb.metrics.cum_drift(amps)
        except Exception as err:
            print("Failed to compute 'cum_drift(amps)' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)

        # Cumulative drift of spike depths, normalized by total number of spikes.
        try:
            cum_depth_drift[idx] = bb.metrics.cum_drift(depths)
        except Exception as err:
            print("Failed to compute 'cum_drift(depths)' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)

        # Coefficient of variation of spike amplitudes.
        try:
            cv_amp[idx] = np.std(amps) / np.mean(amps)
        except Exception as err:
            print("Failed to compute 'cv_amp' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)

        # Coefficient of variation of computed instantaneous firing rate.
        try:
            fr = bb.singlecell.firing_rate(ts, hist_win=0.01, fr_win=0.25)
            cv_fr[idx] = np.std(fr) / np.mean(fr)
        except Exception as err:
            print("Failed to compute 'cv_fr' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)

        # Fraction of isi violations.
        try:
            frac_isi_viol[idx], _, _ = bb.metrics.isi_viol(ts, rp=0.002)
        except Exception as err:
            print("Failed to compute 'frac_isi_viol' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)

        # Estimated fraction of missing spikes.
        try:
            frac_missing_spks[idx], _, _ = bb.metrics.feat_cutoff(
                amps, spks_per_bin=10, sigma=4, min_num_bins=50)
        except Exception as err:
            print("Failed to compute 'frac_missing_spks' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)
    
        # Estimated fraction of false positives.
        try:
            fp_est[idx] = bb.metrics.fp_est(ts, rp=0.002)
        except Exception as err:
            print("Failed to compute 'fp_est' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)

        # Presence ratio
        try:
            pres_ratio[idx], _ = bb.metrics.pres_ratio(ts, hist_win=10)
        except Exception as err:
            print("Failed to compute 'pres_ratio' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)
    
        # Presence ratio over the standard deviation of spike counts in each bin
        try:
            pr, pr_bins = bb.metrics.pres_ratio(ts, hist_win=10)
            pres_ratio_std[idx] = pr / np.std(pr_bins)
        except Exception as err:
            print("Failed to compute 'pres_ratio_std' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)


    #append metrics to the current clusters.metrics
    metrics_read = pd.read_csv(Path(alf_probe_dir,'clusters.metrics.csv'))

    try:
        label_df = pd.DataFrame(label)
        pd.DataFrame.insert(metrics_read,1,'label',label_df)  
    except ValueError:
        pd.DataFrame.drop(metrics_read,columns = 'label')
        pd.DataFrame.insert(metrics_read,1,'label',label_df)  
    except:
        print("Could not save 'label' to .csv.")

    try:
        df_cum_amp_drift = pd.DataFrame(cum_amp_drift.round(2))
        metrics_read['cum_amp_drift'] = df_cum_amp_drift
    except Exception as err:
        print("Could not save 'cum_amp_drift' to .csv. Details: \n {}".format(err))
    
    try:
        df_cum_depth_drift = pd.DataFrame(cum_depth_drift.round(2))
        metrics_read['cum_depth_drift'] = df_cum_depth_drift
    except Exception as err:
        print("Could not save 'cum_depth_drift' to .tsv. Details: \n {}".format(err))
    
    try:
        df_cv_amp = pd.DataFrame(cv_amp.round(2))
        metrics_read['cv_amp'] = df_cv_amp
    except Exception as err:
        print("Could not save 'cv_amp' to .tsv. Details: \n {}".format(err))
    
    try:
        df_cv_fr = pd.DataFrame(cv_fr.round(2))
        metrics_read['cv_fr'] = df_cv_fr 
    except Exception as err:
        print("Could not save 'cv_fr' to .tsv. Details: \n {}".format(err))
    
    try:
        df_frac_isi_viol = pd.DataFrame(frac_isi_viol.round(2))
        metrics_read['frac_isi_viol'] = df_frac_isi_viol
    except Exception as err:
        print("Could not save 'frac_isi_viol' to .tsv. Details: \n {}".format(err))
    
    try:
        df_frac_missing_spks = pd.DataFrame(frac_missing_spks.round(2))
        metrics_read['frac_missing_spks'] = df_frac_missing_spks
    except Exception as err:
        print("Could not save 'frac_missing_spks' to .tsv. Details: \n {}".format(err))
    
    try:
        df_fp_est = pd.DataFrame(fp_est.round(2))
        metrics_read['fp_est'] = df_fp_est
    except Exception as err:
        print("Could not save 'fp_est' to .tsv. Details: \n {}".format(err))
    
    try:
        df_pres_ratio = pd.DataFrame(pres_ratio.round(2))
        metrics_read['pres_ratio'] = df_pres_ratio
    except Exception as err:
        print("Could not save 'pres_ratio' to .tsv. Details: \n {}".format(err))
    
    try:
        df_pres_ratio_std = pd.DataFrame(pres_ratio_std.round(2))
        metrics_read['pres_ratio_std'] = df_pres_ratio_std
    except Exception as err:
        print("Could not save 'pres_ratio_std' to .tsv. Details: \n {}".format(err))
    
            
    try:
        df_refp_viol = pd.DataFrame(RefPViol)
        pd.DataFrame.insert(metrics_read,2,'refp_viol',df_refp_viol)  
    except ValueError:
        pd.DataFrame.drop(metrics_read,columns = 'refp_viol')
        pd.DataFrame.insert(metrics_read,2,'refp_viol', df_refp_viol)  
    except Exception as err:
        print("Could not save 'RefPViol' to .tsv. Details: \n {}".format(err))
        
    try:
        df_noise_cutoff = pd.DataFrame(NoiseCutoff)
        pd.DataFrame.insert(metrics_read,3,'noise_cutoff',df_noise_cutoff)  
    except ValueError:
        pd.DataFrame.drop(metrics_read,columns = 'noise_cutoff')
        pd.DataFrame.insert(metrics_read,3,'noise_cutoff',df_noise_cutoff)  
    except Exception as err:
        print("Could not save 'NoiseCutoff' to .tsv. Details: \n {}".format(err))
        
    try:
        df_mean_amp_true = pd.DataFrame(MeanAmpTrue)
        pd.DataFrame.insert(metrics_read,4,'mean_amp_true',df_mean_amp_true)  
    except ValueError:
        pd.DataFrame.drop(metrics_read,columns = 'mean_amp_true')
        pd.DataFrame.insert(metrics_read,4,'mean_amp_true',df_mean_amp_true)  
    except Exception as err:
        print("Could not save 'Mean Amp True' to .tsv. Details: \n {}".format(err))
        
        
        
    #now add df to csv 
    metrics_read.to_csv(Path(alf_probe_dir,'clusters.metrics.csv'))
    
    try:    
        numpass=int(sum(label))
        print("\n Number of units that pass: ", numpass)
    
        numpassRP=int(sum(RefPViol))
        numpassAC=int(sum(NoiseCutoff[~np.isnan(NoiseCutoff)]<2.5))
        ntot = len(label)
        
        print("Number of units that pass RP threshold: ", numpassRP)
        print("Number of units that pass Amp Cutoff threshold: ", numpassAC)
        print("Number of total units: ", ntot)
    except Exception as err:
        print ("Could not compute number of units that pass. Details \n {}".format(err))
        
    return metrics_read 
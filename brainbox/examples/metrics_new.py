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
# from max_acceptable_isi_viol_2 import max_acceptable_cont_2
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
    n_units = np.max(spks_b.clusters) + 1
    n_units_nonzeros = len(units_nonzeros)
    
    
    #if raw data available locally: 
    try:
        # ephys_file_dir = r'C:\Users\Steinmetz Lab User\Downloads\FlatIron\zadorlab\Subjects\CSK-scan-008\2019-12-11\001\raw_ephys_data\probe01'
        
        ephys_file_dir = os.path.join(ses_path, 'raw_ephys_data', probe_name)
        # rms_amps, rms_times = (aio.load_object(ephys_file_dir, '_iblqc_ephysTimeRmsAP')).values()
        ephys_file = os.path.join(ses_path, 'raw_ephys_data', probe_name,'_iblrig_ephysData.raw_g0_t0.imec.ap.cbin')
    except Exception:
        print('raw ephys data was not found; some metrics will not be computed')

    # alf_probe_dir = r'C:\Users\Steinmetz Lab User\Downloads\FlatIron\zadorlab\Subjects\CSK-scan-008\2019-12-11\001\alf\probe01'  
   
    uidx=0
                
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
    for unit in units:
        if unit == units[0]:
            t0 = time.perf_counter()  # used for computation time estimate
        
        ts = units_b['times'][unit]
        amps = units_b['amps'][unit]
        samples = units_b['samples'][unit]
        
        RefPViol[int(unit)] = FP_RP(ts)
        NoiseCutoff[int(unit)] = noise_cutoff(amps,quartile_length=.25)
        print(unit)
        if len(samples>50):
            print('running this')
            try:
                MeanAmpTrue[int(unit)] = peak_to_peak_amp(ephys_file, samples, nsamps=2)
    
                if (FP_RP(ts) and noise_cutoff(amps,quartile_length=.25)<20 and MeanAmpTrue[int(unit)]>50) : 
                    label[int(unit)] = 1
                else:
                    label[int(unit)] = 0
            except:
                if (FP_RP(ts) and noise_cutoff(amps,quartile_length=.25)<20) : 
                    label[int(unit)] = 1
                else:
                    label[int(unit)] = 0
                
        else: #no ephys file, do not include true mean amps
            if (FP_RP(ts) and noise_cutoff(amps,quartile_length=.25)<20) : 
                label[int(unit)] = 1
            else:
                label[int(unit)] = 0



        depths = units_b['depths'][unit]

        # Cumulative drift of spike amplitudes, normalized by total number of spikes.
        try:
            cum_amp_drift[int(unit)] = bb.metrics.cum_drift(amps)
        except Exception as err:
            print("Failed to compute 'cum_drift(amps)' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)

        # Cumulative drift of spike depths, normalized by total number of spikes.
        try:
            cum_depth_drift[int(unit)] = bb.metrics.cum_drift(depths)
        except Exception as err:
            print("Failed to compute 'cum_drift(depths)' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)

        # Coefficient of variation of spike amplitudes.
        try:
            cv_amp[int(unit)] = np.std(amps) / np.mean(amps)
        except Exception as err:
            print("Failed to compute 'cv_amp' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)

        # Coefficient of variation of computed instantaneous firing rate.
        try:
            fr = bb.singlecell.firing_rate(ts, hist_win=0.01, fr_win=0.25)
            cv_fr[int(unit)] = np.std(fr) / np.mean(fr)
        except Exception as err:
            print("Failed to compute 'cv_fr' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)

        # Fraction of isi violations.
        try:
            frac_isi_viol[int(unit)], _, _ = bb.metrics.isi_viol(ts, rp=0.002)
        except Exception as err:
            print("Failed to compute 'frac_isi_viol' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)

        # Estimated fraction of missing spikes.
        try:
            frac_missing_spks[int(unit)], _, _ = bb.metrics.feat_cutoff(
                amps, spks_per_bin=10, sigma=4, min_num_bins=50)
        except Exception as err:
            print("Failed to compute 'frac_missing_spks' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)
    
        # Estimated fraction of false positives.
        try:
            fp_est[int(unit)] = bb.metrics.fp_est(ts, rp=0.002)
        except Exception as err:
            print("Failed to compute 'fp_est' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)

        # Presence ratio
        try:
            pres_ratio[int(unit)], _ = bb.metrics.pres_ratio(ts, hist_win=10)
        except Exception as err:
            print("Failed to compute 'pres_ratio' for unit {}. Details: \n {}".format(unit, err))
            units_missing_metrics.add(unit)
    
        # Presence ratio over the standard deviation of spike counts in each bin
        try:
            pr, pr_bins = bb.metrics.pres_ratio(ts, hist_win=10)
            pres_ratio_std[int(unit)] = pr / np.std(pr_bins)
        except Exception as err:
            print("Failed to compute 'pres_ratio_std' for unit {}. Details: \n {}"
                  .format(unit, err))
            units_missing_metrics.add(unit)

        # The mean peak-to-peak value over the background noise of the channel of max amplitude.
        # if ephys_file_path:
        #     try:
        #         ch = clstrs_b['channels'][int(unit)]  # channel of max amplitude
        #         ptp_sigma[int(unit)] = bb.metrics.ptp_over_noise(
        #             ephys_file_path, ts, ch, t=2.0, sr=30000, n_ch_probe=385,
        #             dtype='int16', car=False)
        #     except Exception as err:
        #         print("Failed to compute 'ptp_sigma' for unit {}. Details: \n {}"
        #               .format(unit, err))
        #         units_missing_metrics.add(unit)

        # if unit == units[0]:  # give time estimate
        #     dt = time.perf_counter() - t0
        #     print('\nComputing metrics. Estimated time is {:.2f} mins\n'
        #           .format(len(units) * dt / 60))

    # Extract to a .csv file #
    # --------------------- #
#test code
# metrics_read = pd.read_csv(Path(alf_probe_dir,'clusters.metrics.csv'))
# xx = np.arange(0,704,1)
# metrics_read['extracolumn'] = xx

    
    metrics_read = pd.read_csv(Path(alf_probe_dir,'clusters.metrics.csv'))

    try:
        label_df = pd.DataFrame(label)
        pd.DataFrame.insert(metrics_read,2,'label',label_df)  
        # label_df.to_csv(Path(alf_probe_dir, 'clusters.metrics_validation.csv'),
                        # header=['label'],index=False)
    except Exception as err:
        print("Could not save 'label' to .csv. Details: \n {}".format(err))
  

    #read this csv file, append all metrics to it
    # df_csv = pd.read_csv(Path(alf_probe_dir, 'clusters.metrics_validation.csv'))

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
    
    # if ephys_file_path:
    #     try:
    #         df_ptp_sigma = pd.DataFrame(ptp_sigma.round(2))
    #         df_ptp_sigma.to_csv(Path(ks_dir, 'ptp_sigma.tsv'),
    #                                 sep='\t', header=['cum_amp_drift'])
    #     except Exception as err:
    #         print("Could not save 'cum_amp_drift' to .tsv. Details: \n {}".format(err))
    
            
            
        
    try:
        df_refp_viol = pd.DataFrame(RefPViol)
        # metrics_read['refp_viol'] = df_refp_viol
        pd.DataFrame.insert(metrics_read,3,'refp_viol',df_refp_viol)  

    except Exception as err:
        print("Could not save 'RefPViol' to .tsv. Details: \n {}".format(err))
        
    try:
        df_noise_cutoff = pd.DataFrame(NoiseCutoff)
        # metrics_read['noise_cutoff'] = df_noise_cutoff
        pd.DataFrame.insert(metrics_read,4,'noise_cutoff',df_noise_cutoff)  

    except Exception as err:
        print("Could not save 'NoiseCutoff' to .tsv. Details: \n {}".format(err))
        
    try:
        df_mean_amp_true = pd.DataFrame(MeanAmpTrue)
        # metrics_read['noise_cutoff'] = df_noise_cutoff
        pd.DataFrame.insert(metrics_read,5,'mean_amp_true',df_mean_amp_true)  
    
    except Exception as err:
        print("Could not save 'Mean Amp True' to .tsv. Details: \n {}".format(err))
        
        
        
     #now add df to csv 
    # df_csv.to_csv(Path(alf_probe_dir, 'clusters.metrics_validation.csv'))
    metrics_read.to_csv(Path(alf_probe_dir,'clusters.metrics_validationß.csv'))
        
    numpass=int(sum(label))
    print("Number of units that pass: ", numpass)

    numpassRP=int(sum(RefPViol))
    numpassAC=int(sum(NoiseCutoff[~np.isnan(NoiseCutoff)]<2.5))
    ntot = len(label)
    
    print("Number of units that pass RP threshold: ", numpassRP)
    print("Number of units that pass Amp Cutoff threshold: ", numpassAC)
    print("Number of total units: ",ntot)
    return numpass, numpassRP, numpassAC, ntot
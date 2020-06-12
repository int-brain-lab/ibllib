"""
Get passive CW session and data.

STEPS:
- Load fixture data
- Cut out part about ephysCW
- Find spacer + check number found
- Get number of TTL switch (f2ttl, audio, valve) within each spacer
- Associate TTL found for each stim type + check number found
- Package and output data (alf format?)
"""
# Author: Olivier W, Gaelle C
import alf.io
from oneibl.one import ONE
from pathlib import Path
import numpy as np
import json
import ibllib.io.extractors.passive as passive
from ibllib.io.extractors import ephys_fpga
import ibllib.io.raw_data_loaders as rawio

# hardcoded var
FRAME_FS = 60  # Sampling freq of the ipad screen, in Hertz
FS_FPGA = 30000  # Sampling freq of the neural recording system screen, in Hertz

# load data
one = ONE()
dataset_types = ['_spikeglx_sync.times',
                 '_spikeglx_sync.channels',
                 '_spikeglx_sync.polarities',
                 '_iblrig_RFMapStim.raw',
                 '_iblrig_stimPositionScreen.raw',
                 '_iblrig_syncSquareUpdate.raw',
                 'ephysData.raw.meta',
                 '_iblrig_taskSettings.raw'
                 ]

eid = one.search(subject='CSH_ZAD_022', date_range='2020-05-24', number=1)[0]
local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)

session_path = alf.io.get_session_path(local_paths[0])

# load session fixtures
settings = rawio.load_settings(session_path)
ses_nb = settings['SESSION_ORDER'][settings['SESSION_IDX']]
path_fixtures = Path(ephys_fpga.__file__).parent.joinpath('ephys_sessions')
pcs = np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_pcs.npy'))
delays = np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_stimDelays.npy'))
ids = np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_stimIDs.npy'))

# load general metadata
json_file = path_fixtures.joinpath('passive_stim_meta.json')
with open(json_file, 'r') as f:
    meta = json.load(f)

# assign key to stimuli
sp_key = passive.key_vis_stim(text_append='VISUAL_STIM_',
                              dict_vis=meta['VISUAL_STIMULI'],
                              value_search='SPACER')
rf_key = passive.key_vis_stim(text_append='VISUAL_STIM_',
                              dict_vis=meta['VISUAL_STIMULI'],
                              value_search='receptive_field_mapping')
ts_key = passive.key_vis_stim(text_append='VISUAL_STIM_',
                              dict_vis=meta['VISUAL_STIMULI'],
                              value_search='task_stimuli')
sa_key = passive.key_vis_stim(text_append='VISUAL_STIM_',
                              dict_vis=meta['VISUAL_STIMULI'],
                              value_search='spontaneous_activity')

# load stimulus sequence
stim_order = np.array(meta['STIM_ORDER'])

# load ephys sync pulses
sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
fpga_sync = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'])

# load Frame2ttl / audio / valve signal
fttl = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'])
audio = ephys_fpga._get_sync_fronts(sync, sync_map['audio'])
valve = ephys_fpga._get_sync_fronts(sync, sync_map['bpod'])

# truncate fttl signal so as to contain only what comes after ephysCW
t_end_ephys = passive.ephysCW_end(session_path=session_path)
fttl_trunk = passive.truncate_ttl_signal(ttl=fttl, time_cutoff=t_end_ephys)

# load and get spacer information, do corr to find spacer timestamps
ttl_signal = fttl_trunk['times']
spacer_template = np.array(meta[sp_key]['ttl_frame_nums'], dtype=np.float32) / FRAME_FS
jitter = 3 / FRAME_FS  # allow for 3 screen refresh as jitter
t_quiet = meta[sp_key]['delay_around']
spacer_times, _ = passive.get_spacer_times(
    spacer_template=spacer_template, jitter=jitter,
    ttl_signal=ttl_signal, t_quiet=t_quiet)

# Check correct number of spacer is found
indx_0 = np.where(stim_order == 0)  # Hardcoded 0 for spacer
n_exp_spacer = np.size(indx_0)
if n_exp_spacer != np.size(spacer_times) / 2:
    raise ValueError(f'The number of expected spacer ({n_exp_spacer}) '
                     f'is different than the one found on the raw '
                     f'trace ({np.size(spacer_times)/2})')

# once spacer is found, truncate ttl signals so as to contain only what comes after ephysCW
t_start_passive = spacer_times[0, 0]
fttl_trunk = passive.truncate_ttl_signal(ttl=fttl, time_cutoff=t_start_passive)
audio_trunk = passive.truncate_ttl_signal(ttl=audio, time_cutoff=t_start_passive)
valve_trunk = passive.truncate_ttl_signal(ttl=valve, time_cutoff=t_start_passive)

# split ids into relevant HW categories
gabor_id = [s for s in ids if 'G' in s]
gabor_index = np.where(ids == 'G')[0]
valve_id = [s for s in ids if 'V' in s]
matched = ['T', 'N']
sound_id = [z for z in ids if z in matched]

# Test correct number is found in metadata (hardcoded from protocol)
# Note: This could be done upon creation of the npy file
len_g_pr = 20 + 20 * 4 * 2
if len_g_pr != len(gabor_id):
    raise ValueError("N Gabor stimulus in metadata incorrect")
else:
    meta[ts_key] = dict()
    meta[ts_key]['ttl_num'] = len(gabor_id)  # TODO put this into JSON ?
len_v_pr = 40
if len_v_pr != len(valve_id):
    raise ValueError("N Valve stimulus in metadata incorrect")
len_s_pr = 40 * 2
if len_s_pr != len(sound_id):
    raise ValueError("N Sound stimulus in metadata incorrect")

# Test correct number is found in ttl data (audio / valve, f2ttl done separately)
if len(valve_id) != np.size(valve_trunk['polarities']) / 2:
    raise ValueError("N Valve stimulus in ttl data is incorrect")
if len(sound_id) != np.size(audio_trunk['polarities']) / 2:
    raise ValueError("N Sound stimulus in ttl data incorrect")

# load RF matrix and reshape
RF_file = Path.joinpath(session_path, 'raw_passive_data', '_iblrig_RFMapStim.raw.bin')
RF_frames, RF_ttl_trace = passive.reshape_RF(
    RF_file=RF_file, meta_stim=meta[rf_key])
rf_id_up, rf_id_dw, RF_n_ttl_expected = \
    passive.get_id_raisefall_from_analogttl(RF_ttl_trace)
meta[rf_key]['ttl_num'] = RF_n_ttl_expected

# Check that correct number of f2ttl switch is found for each visual stim type
# Hardcode as only 3 visual stim
# Add some jitter (0.2s) to not catch Bonsai update

# 1. spont act
passive.check_n_ttl_between(n_exp=meta[sa_key]['ttl_num'],
                            key_stim=sa_key,
                            t_start_search=spacer_times[0, 1] + 0.2,
                            t_end_search=spacer_times[1, 0] - 0.2,
                            ttl=fttl_trunk)

# 2. RF
RF_times = \
    passive.check_n_ttl_between(n_exp=meta[rf_key]['ttl_num'],
                                key_stim=rf_key,
                                t_start_search=spacer_times[1, 1] + 0.2,
                                t_end_search=spacer_times[2, 0] - 0.2,
                                ttl=fttl_trunk)

# take only 1 out of 2 values (only care about value when stim change,
# not when ttl goes back to level after first ttl pulse)
RF_times_1 = RF_times[0::2]
# Interpolate times for RF before outputting dataset
times_interp_RF =\
    passive.interpolate_rf_mapping_stimulus(idxs_up=rf_id_up,
                                            idxs_dn=rf_id_dw,
                                            times=RF_times_1,
                                            Xq=np.arange(RF_frames.shape[0]),
                                            t_bin=1 / FRAME_FS)

# 3. gabor
gabor_times = \
    passive.check_n_ttl_between(n_exp=meta[ts_key]['ttl_num'] * 2,  # *2 for rise/fall
                                key_stim=ts_key,
                                t_start_search=spacer_times[2, 1] + 0.2,
                                t_end_search=fttl_trunk['times'][-1],
                                ttl=fttl_trunk)

# # plot for debug
# from ibllib.plots import squares
# import matplotlib.pyplot as plt

# # -- Gabor --
# times, _ = passive.find_between(t_start_search=spacer_times[2, 1] + 0.2,
#                                 t_end_search=fttl_trunk['times'][-1],
#                                 ttl=fttl_trunk)
# squares(fttl_trunk['times'], fttl_trunk['polarities'])
# plt.plot(times, 0.5 * np.ones(len(times)), '.')

# # -- RF --
# times, _ = passive.find_between(t_start_search=spacer_times[1, 1] + 0.2,
#                                 t_end_search=spacer_times[2, 0] - 0.2,
#                                 ttl=fttl_trunk)
# squares(fttl_trunk['times'], fttl_trunk['polarities'])
# plt.plot(times, 0.5 * np.ones(len(times)), '.')

# # Plot time diff gabor
# cs_delays = np.cumsum(delays)
# diff_delays = np.diff(cs_delays[gabor_index])[1:]
# gb_diff_ts = np.diff(gabor_times)[1::2]
# plt.plot([0,3], [0,3], linewidth=2.0)
# plt.plot(diff_delays, gb_diff_ts, '.')
# plt.xlabel('saved delays diff [s]')
# plt.ylabel('measured times diff [s]')
# pearson_r = np.corrcoef(diff_delays, gb_diff_ts)[1, 0]

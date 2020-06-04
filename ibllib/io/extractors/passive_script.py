"""
Get passive CW session and data.

STEPS:
- Load fixture data
- Find spacer (still do convolution?) + check number found
- Cut out part about ephysCW
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
# plot for debug
# from ibllib.plots import squares
# import matplotlib.pyplot as plt

# import ibllib.io.extractors.passive as passive

import ibllib.io.raw_data_loaders as rawio
from ibllib.io.extractors import ephys_fpga

# hardcoded var
FRAME_FS = 60  # Sampling freq of the ipad screen, in Hertz

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

# load ephys sync pulses
sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
fpga_sync = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'])

# load Frame2ttl / audio / valve signal
fttl = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'])
audio = ephys_fpga._get_sync_fronts(sync, sync_map['audio'])
valve = ephys_fpga._get_sync_fronts(sync, sync_map['bpod'])
# todo check that bpod does not output any other signal than valve in this task protocol

# load RF matrix and reshape
RF_file = Path.joinpath(session_path, 'raw_passive_data', '_iblrig_RFMapStim.raw.bin')
frame_array = np.fromfile(RF_file, dtype='uint8')
# Reshape matrix, todo make test for reshape
y_pix, x_pix, _ = meta['VISUAL_STIM_1']['stim_file_shape']
frames = np.transpose(
    np.reshape(frame_array, [y_pix, x_pix, -1], order='F'), [2, 1, 0])

# todo
# TTL_data = squeeze(pr_RFmetadata(1,1,:));
#
# % -- Convert values to 0,1,-1 for simplicity
# TTL_data01 = zeros(size(TTL_data));
# TTL_data01(find(TTL_data==0)) = -1;
# TTL_data01(find(TTL_data==255)) = 1;
#
# % -- Find number of passage from [128 0] and [128 255]  (converted to 0,1,-1)
# d_TTL_data01 = diff(TTL_data01);
#
# id_raise = find(TTL_data01==0 & [d_TTL_data01; 0]==1);
# id_fall = find(TTL_data01==0 & [d_TTL_data01; 0]==-1);
#
# % -- number of rising TTL pulse expected in frame2ttl trace
# TTL_data_Rise_Expected = ...
#     length(id_raise) + ...
#     length(id_fall);

# Find spacer in f2ttl, check number found is valid

# Re-create raw f2ttl signal
FS_FPGA = 30000  # Hz
inc_f = np.round(fttl['times'] * FS_FPGA)
inc_f = inc_f.astype(int)
vect_f = np.zeros((1, max(inc_f) + 1))
vect_f[0][inc_f] = fttl['polarities']
signal_f = np.cumsum(vect_f[0])

# cut f2ttl signal so as to contain only what comes after ephysCW
bpod_raw = rawio.load_data(session_path)
t_end_ephys = bpod_raw[-1]['behavior_data']['States timestamps']['exit_state'][0][-1] + 60
inc_t = np.round(t_end_ephys * FS_FPGA)
inc_t = inc_t.astype(int)
signal_f_passive = signal_f[inc_t:]

ttl_signal = signal_f_passive
# load and get spacer information
spacer_template = meta['VISUAL_STIM_0']['ttl_frame_nums']
jitter = 1 / FRAME_FS * 3
t_quiet = meta['VISUAL_STIM_0']['delay_around']
# spacer_times, conv_dttl = passive.get_spacer_times(
#     spacer_template=spacer_template, jitter=jitter,
#     ttl_signal=ttl_signal, t_quiet=t_quiet)

# todo corr

# === TODO PUT INTO FUNCTION HERE FOR DEBUGGING
diff_spacer_template = np.diff(spacer_template)
# add jitter;
# remove extreme values
spacer_model = jitter + diff_spacer_template[2:-2]
# diff ttl signal to compare to spacer_model
dttl = np.diff(ttl_signal)
# remove diffs larger than max diff in model to clean up signal
dttl[dttl > np.max(spacer_model)] = 0
# convolve cleaned diff ttl signal w/ spacer model
conv_dttl = np.correlate(dttl, spacer_model, mode='full')
# find spacer location
thresh = 3.0
idxs_spacer_middle = np.where(
    (conv_dttl[1:-2] < thresh) &
    (conv_dttl[2:-1] > thresh) &
    (conv_dttl[3:] < thresh))[0]
# adjust indices for
# - `np.where` call above
# - length of spacer_model
idxs_spacer_middle += 2 - int((np.floor(len(spacer_model) / 2)))
# pull out spacer times (middle)
ts_spacer_middle = ttl_signal[idxs_spacer_middle]
# put beginning/end of spacer times into an array
spacer_length = np.max(spacer_template)
spacer_times = np.zeros(shape=(ts_spacer_middle.shape[0], 2))
for i, t in enumerate(ts_spacer_middle):
    spacer_times[i, 0] = t - (spacer_length / 2) - t_quiet
    spacer_times[i, 1] = t + (spacer_length / 2) + t_quiet
# =====

# load stimulus sequence
# todo

# split ids into relevant HW categories
gabor_id = [s for s in ids if 'G' in s]
valve_id = [s for s in ids if 'V' in s]

matched = ['T', 'N']
sound_id = [z for z in ids if z in matched]

# Test correct number is found in metadata (hardcoded from protocol)
# Todo is this necessary here? This should be done upon creation of the npy file
len_g_pr = 20 + 20 * 4 * 2
if len_g_pr != len(gabor_id):
    raise ValueError("N Gabor stimulus in metadata incorrect")
len_v_pr = 40
if len_v_pr != len(valve_id):
    raise ValueError("N Valve stimulus in metadata incorrect")
len_s_pr = 40 * 2
if len_s_pr != len(sound_id):
    raise ValueError("N Sound stimulus in metadata incorrect")

# import json
# json_file = '/Users/gaelle/Desktop/passive_stim_meta.json'
# with open(json_file, 'w+') as f:
#     string = json.dumps(meta_stim, indent=1)
#     f.write(string)

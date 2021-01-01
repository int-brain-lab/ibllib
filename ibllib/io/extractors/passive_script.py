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
NVALVE = 40  # number of expected valve clicks
NGABOR = 20 + 20 * 4 * 2  # number of expected Gabor patches
NTONES = 40
NNOISES = 40
DEBUG_PLOTS = False

# load data
one = ONE()
dataset_types = ['_spikeglx_sync.times',
                 '_spikeglx_sync.channels',
                 '_spikeglx_sync.polarities',
                 '_iblrig_RFMapStim.raw',
                 '_iblrig_stimPositionScreen.raw',
                 '_iblrig_syncSquareUpdate.raw',
                 'ephysData.raw.meta',
                 '_iblrig_taskSettings.raw',
                 '_iblrig_taskData.raw'
                 ]

eid = one.search(subject='CSH_ZAD_022', date_range='2020-05-24', number=1)[0]
local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)

session_path = alf.io.get_session_path(local_paths[0])

# load session fixtures
settings = rawio.load_settings(session_path)
ses_nb = settings['SESSION_ORDER'][settings['SESSION_IDX']]
path_fixtures = Path(ephys_fpga.__file__).parent.joinpath('ephys_sessions')
fixture = {'pcs': np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_pcs.npy')),
           'delays': np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_stimDelays.npy')),
           'ids': np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_stimIDs.npy'))}

# load general metadata
with open(path_fixtures.joinpath('passive_stim_meta.json'), 'r') as f:
    meta = json.load(f)
t_end_ephys = passive.ephysCW_end(session_path=session_path)
# load stimulus sequence
sync, sync_map = ephys_fpga.get_main_probe_sync(session_path, bin_exists=False)
fpga_sync = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'])
fttl = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'], tmin=t_end_ephys)


def get_spacers():
    """
    load and get spacer information, do corr to find spacer timestamps
    returns t_passive_starts, t_starts, t_ends
    """
    spacer_template = np.array(meta['VISUAL_STIM_0']['ttl_frame_nums'],
                               dtype=np.float32) / FRAME_FS
    jitter = 3 / FRAME_FS  # allow for 3 screen refresh as jitter
    t_quiet = meta['VISUAL_STIM_0']['delay_around']
    spacer_times, _ = passive.get_spacer_times(spacer_template=spacer_template, jitter=jitter,
                                               ttl_signal=fttl['times'], t_quiet=t_quiet)

    # Check correct number of spacers found
    n_exp_spacer = np.sum(np.array(meta['STIM_ORDER']) == 0)  # Hardcoded 0 for spacer
    if n_exp_spacer != np.size(spacer_times) / 2:
        raise ValueError(f'The number of expected spacer ({n_exp_spacer}) '
                         f'is different than the one found on the raw '
                         f'trace ({np.size(spacer_times)/2})')

    spacer_times = np.r_[spacer_times.flatten(), sync['times'][-1]]
    return spacer_times[0], spacer_times[1::2], spacer_times[2::2]


# loop over stimuli , get start/end times and meta dictionary key
t_start_passive, t_starts, t_ends = get_spacers()
stim_numbers = list(filter(lambda x: x != 0, meta['STIM_ORDER']))
STIMS = {}  # this is the metadata we will save as intervals
for i, istim in enumerate(stim_numbers):
    stim = meta['VISUAL_STIMULI'][str(istim)]  # get stim label
    meta_key = passive.key_vis_stim(
        text_append='VISUAL_STIM_', dict_vis=meta['VISUAL_STIMULI'], value_search=stim)
    STIMS[stim] = {'start': t_starts[i],
                   'end': t_ends[i],
                   'mkey': meta_key}

"""
1/3 Spontaneous activity: assert there is no frame detected
"""
s = STIMS['spontaneous_activity']
fttl = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'], tmin=s['start'], tmax=s['end'])
passive.check_n_ttl_between(n_exp=meta[s['mkey']]['ttl_num'],
                            key_stim=s,
                            t_start_search=s['start'] + 0.2,
                            t_end_search=s['end'] - 0.2,
                            ttl=fttl)

"""
2/3 Receptive field mapping: get matrix and times
"""
s = STIMS['receptive_field_mapping']
fttl = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'], tmin=s['start'], tmax=s['end'])
RF_file = Path.joinpath(session_path, 'raw_passive_data', '_iblrig_RFMapStim.raw.bin')
RF_frames, RF_ttl_trace = passive.reshape_RF(RF_file=RF_file, meta_stim=meta[s['mkey']])
rf_id_up, rf_id_dw, RF_n_ttl_expected = passive.get_id_raisefall_from_analogttl(RF_ttl_trace)
meta[s['mkey']]['ttl_num'] = RF_n_ttl_expected
RF_times = passive.check_n_ttl_between(n_exp=meta[s['mkey']]['ttl_num'],
                                       key_stim=s['mkey'],
                                       t_start_search=s['start'] + 0.2,
                                       t_end_search=s['end'] - 0.2,
                                       ttl=fttl)
RF_times_1 = RF_times[0::2]
# Interpolate times for RF before outputting dataset
times_interp_RF = passive.interpolate_rf_mapping_stimulus(
    idxs_up=rf_id_up,
    idxs_dn=rf_id_dw,
    times=RF_times_1,
    Xq=np.arange(RF_frames.shape[0]),
    t_bin=1 / FRAME_FS)
# return RF_frames, times_interp_RF

"""
3/3 Replay of task stimuli
take only 1 out of 2 values (only care about value when stim change,
not when ttl goes back to level after first ttl pulse)
"""
## check metadata ?!?, why not
assert NGABOR == np.sum(fixture['ids'] == 'G'), "N Gabor stimulus in metadata incorrect"
assert NVALVE == np.sum(fixture['ids'] == 'V'), "N Valve stimulus in metadata incorrect"
assert NTONES == np.sum(fixture['ids'] == 'T'), "N Sound stimulus in metadata incorrect"
assert NNOISES == np.sum(fixture['ids'] == 'T'), "N Sound stimulus in metadata incorrect"
# chop-off sync fronts to the relevant time interval
s = STIMS['task_stimuli']
fttl = ephys_fpga._get_sync_fronts(
    sync, sync_map['frame2ttl'], tmin=s['start'] + 0.1, tmax=s['end'])
audio = ephys_fpga._get_sync_fronts(sync, sync_map['audio'], tmin=s['start'], tmax=s['end'])
bpod = ephys_fpga._get_sync_fronts(sync, sync_map['bpod'], tmin=s['start'], tmax=s['end'])

tready, terror = ephys_fpga._assign_events_audio(audio['times'], audio['polarities'])
_, t_valve_open, _ = ephys_fpga._assign_events_bpod(bpod['times'], bpod['polarities'],
                                                    ignore_first_valve=False)
t_gabor = fttl['times']
# QC 1/2 the order of pulses: this won't work as the events are asynchronous and delays unknown
ids = np.array(['T'] * tready.size + ['N'] * terror.size +
               ['V'] * t_valve_open.size + ['G'] * t_gabor[::2].size)
tids = np.r_[tready, terror, t_valve_open, t_gabor[::2]]
isort = np.argsort(tids)
# np.all(ids[isort] == fixture['ids'])  # this will not
for i in range(len(ids)):
    print(str(i).zfill(3), ids[isort[i]], fixture['ids'][i],
          ids[isort[i]] == fixture['ids'][i], tids[isort[i]])


# QC 2/2 check that FPGA events and delays match the ones from the task fixture
igabor = np.where(fixture['ids'] == 'G')[0]
gabor_int = np.diff(t_gabor)[1::2]  # interval between up and down
tdelays = fixture['delays'].copy()

tdelays[fixture['ids'] == 'T'] += 0.102 + 0.04  # tone
tdelays[fixture['ids'] == 'N'] += 0.510 + 0.04  # error tone (noise)
tdelays[fixture['ids'] == 'V'] += 0.04  # valve
tdelays[fixture['ids'] == 'G'] += 0.3  # gabor patch
# tdelays += 0.3


gabor_fixtures = np.cumsum(tdelays)[igabor]
valve_fixtures = np.cumsum(tdelays)[np.where(fixture['ids'] == 'V')[0]]
gb_diff_ts = np.diff(t_gabor[0::2])
pearson_r = np.corrcoef(np.diff(gabor_fixtures), gb_diff_ts)[1, 0]
assert pearson_r > 0.95

DEBUG_PLOTS = True
if DEBUG_PLOTS:
    # plots for debug
    t0 = np.median(t_valve_open - valve_fixtures)
    from ibllib.plots import squares, vertical_lines, color_cycle
    import matplotlib.pyplot as plt
    pl, ax = plt.subplots(2, 1)
    for i, lab in enumerate(['frame2ttl', 'audio', 'bpod']):
        sy = ephys_fpga._get_sync_fronts(sync, sync_map[lab], tmin=t_start_passive)
        squares(sy['times'], sy['polarities'], yrange=[0.1 + i, 0.9 + i], color='k', ax=ax[0])

    vertical_lines(np.r_[t_start_passive, t_starts, t_ends], ymin=-1, ymax=4, color=color_cycle(0),
                   ax=ax[0], label='spacers')
    vertical_lines(gabor_fixtures + t0, ymin=-1, ymax=4, color=color_cycle(1),
                   ax=ax[0], label='fixtures gabor')
    vertical_lines(t_valve_open, ymin=-1, ymax=4, color=color_cycle(2), ax=ax[0], label='valve')
    vertical_lines(valve_fixtures + t0, ymin=-1, ymax=4, color=color_cycle(2), ax=ax[0],
                   linestyle='--', label='fixtures valve')

    ax[0].legend()

    ax[1].plot([0, 3], [0, 3], linewidth=2.0)
    # plt.plot(diff_delays, gb_diff_ts, '.')
    plt.xlabel('saved delays diff [s]')
    plt.ylabel('measured times diff [s]')
    # scatter plot
    plt.scatter(np.diff(gabor_fixtures), gb_diff_ts,
                c=(fixture['ids'][igabor - 1] == 'G')[:-1], s=10)
    plt.colorbar()

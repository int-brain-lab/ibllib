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

import ibllib.io.raw_data_loaders as rawio
from ibllib.io.extractors import ephys_fpga

# import ibllib.io.extractors.passive as passive
# import importlib
# importlib.reload(passive)

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

# session_path = Path('/datadisk/FlatIron/zadorlab/Subjects/CSH_ZAD_022/2020-05-24/001')

# load session fixtures
settings = rawio.load_settings(session_path)
ses_nb = settings['SESSION_ORDER'][settings['SESSION_IDX']]
path_fixtures = Path(ephys_fpga.__file__).parent.joinpath('ephys_sessions')
pcs = np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_pcs.npy'))
delays = np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_stimDelays.npy'))
ids = np.load(path_fixtures.joinpath(f'session_{ses_nb}_passive_stimIDs.npy'))

# load ephys sync pulses
sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
fpga_sync = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'])

# get Frame2ttl / audio / valve signal
fttl = ephys_fpga._get_sync_fronts(sync, sync_map['frame2ttl'])
audio = ephys_fpga._get_sync_fronts(sync, sync_map['audio'])
valve = ephys_fpga._get_sync_fronts(sync, sync_map['bpod'])
# todo check that bpod does not output any other signal than valve in this task protocol

# load RF matrix
RF_file = Path.joinpath(session_path, 'raw_passive_data', '_iblrig_RFMapStim.raw.bin')
frame_array = np.fromfile(RF_file, dtype='uint8')
# todo reshape matrix, make test for reshape, need shape info
# frames = np.transpose(
#     np.reshape(frame_array, [y_pix, x_pix, -1], order='F'), [2, 1, 0])
# -- Convert values to 0,1,-1 for simplicity
# -- Find number of passage from [128 0] and [128 255]  (converted to 0,1,-1)
# -- number of rising TTL pulse expected in frame2ttl trace

# load spacer information
# todo

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

# Find spacer in f2ttl, check number found is valid
# todo convolution ?

"""
Get passive CW session and data.
"""
# Author: Olivier W
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

# load RF matrix
RF_file = Path.joinpath(session_path, 'raw_passive_data', '_iblrig_RFMapStim.raw.bin')
frame_array = np.fromfile(RF_file, dtype='uint8')
# need to reshape matrix - todo function reusing Matt's code

from pathlib import Path
import numpy as np

from oneibl.one import ONE
import alf.io as aio
from ibllib.io import spikeglx
from ibllib.io.spikeglx import glob_ephys_files
from brainbox.core import Bunch


# load required dataset_types
one = ONE()
eid  = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
probe = 'probe_right'
dtypes = ['clusters.amps',
          'clusters.channels',
          'clusters.depths',
          'clusters.metrics',
          'clusters.peakToTrough',
          'clusters.probes',
          'clusters.uuids',
          'clusters.waveforms',
          'clusters.waveformsChannels',
          'spikes.amps',
          'spikes.clusters',
          'spikes.depths',
          'spikes.samples',
          'spikes.templates',
          'spikes.times',
          'ephysData.raw.ap',
          'ephysData.raw.ch',
          'ephysData.raw.lf',
          'ephysData.raw.meta',
          'ephysData.raw.nidq',
          'ephysData.raw.sync',
          'ephysData.raw.timestamps',
          'ephysData.raw.wiring']
one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)
session_path = one.path_from_eid(eid)
alf_dir = Path.joinpath(session_path, 'alf')
alf_probe_dir = Path.joinpath(alf_dir, probe)
ephys_files = glob_ephys_files(session_path)
ephys_file_path = None
if ephys_files:
    try:
        ephys_file_path = ephys_files[0]['ap']
    except Exception as e:
        print('Something went wrong getting ap ephys file path: ' + e.args[0])
else:
    raise FileNotFoundError('Ephys file not found!')

# get spike samples and channels of max amplitude
spks_b = aio.load_object(alf_probe_dir, 'spikes')
clstrs_b = aio.load_object(alf_probe_dir, 'clusters')
# chnls_b = aio.load_object(alf_probe_dir, 'channels') #from Noam: this broke -- no file on flatiron??
n_units = np.max(spks_b.clusters) + 1

# instantiate spikeglx Reader object and amps bunch
s_reader = spikeglx.Reader(ephys_file_path)
# get total number of samples
n_samples = s_reader.meta['fileTimeSecs'] * s_reader.meta['imSampRate']

# amps bunch will be a bunch where each key is a unit whose value is a numpy array of [n_spikes x
# n_ch], where each value in this array will be the raw voltage value.
# e.g. get the 10 amplitude values across 10 channels of max amplitude for the first spike of
# unit 0: `amps_b['0'][0,:]
import time 
amps_b = Bunch()
n_ch = 10  # number of channels to return for each sample
for unit in range(3):#n_units):
    start = time.time()
    spk_samples = np.where(spks_b.clusters == unit)[0]
    max_ch = max_ch = clstrs_b['channels'][unit]
    if max_ch < n_ch:  # take only channels greater than `max_ch`.
        ch = np.arange(max_ch, max_ch + n_ch)
    elif (max_ch + n_ch) > 385:  # take only channels less than `max_ch`.
        ch = np.arange(max_ch - n_ch, max_ch)
    else:  # take `n_c_ch` around `max_ch`.
        ch = np.arange(max_ch - (n_ch / 2), max_ch + (n_ch / 2))

    amps = np.zeros((len(spk_samples), n_ch))
    # for each spike, index into ephys file
    for i, spk in enumerate(spk_samples):
        amps[i, :], _ = s_reader.read_samples(spk, spk+1, ch)

    amps_b[str(unit)] = amps
    print(time.time()-start)
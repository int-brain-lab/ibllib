import numpy as np

from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE

one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")
eids = one.search(subject='ZM_2407', task_protocol='ephys')

channels = load_channel_locations(eids[0], one=one)
spikes, clusters = load_spike_sorting(eids[0], one=one)

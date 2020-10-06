"""
Download data and plot drift over the session
==============================================

Downloads LFP power spectrum for a given session and probe and plots a heatmap of power spectrum
on the channels along probe against frequency
"""

# import modules
import alf.io
from oneibl.one import ONE
from brainbox.metrics import electrode_drift
import matplotlib.pyplot as plt

# instantiate one
one = ONE()

# Specify subject, date and probe we are interested in
subject = 'CSHL049'
date = '2020-01-08'
sess_no = 1
probe_label = 'probe00'
eid = one.search(subject=subject, date=date, number=sess_no)[0]

# define datasets to download
dtypes = ['spikes.times',
          'spikes.depths',
          'spikes.amps']

# Download the data and get paths to downloaded data
_ = one.load(eid, dataset_types=dtypes, download_only=True)
alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)

# Load in spikes object and use brainbox function to compute drift over session
spikes = alf.io.load_object(alf_path, 'spikes')
drift = electrode_drift.estimate_drift(spikes['times'], spikes['amps'], spikes['depths'],
                                       display=True)
plt.show()



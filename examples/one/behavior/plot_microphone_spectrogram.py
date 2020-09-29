"""
For a given session eid, plot spectrogram of sound recorded via the microphone.
"""
# Author: Gaelle Chapuis

import numpy as np
import matplotlib.pyplot as plt
import alf.io
from oneibl.one import ONE

one = ONE()

dataset_types = [
    '_iblmic_audioSpectrogram.frequencies',
    '_iblmic_audioSpectrogram.power',
    '_iblmic_audioSpectrogram.times_mic']

eid = '098bdac5-0e25-4f51-ae63-995be7fe81c7'  # TEST EXAMPLE

one.load(eid, dataset_types=dataset_types, download_only=True)
session_path = one.path_from_eid(eid)

# -- Get spectrogram
TF = alf.io.load_object(session_path.joinpath('raw_behavior_data'),
                        'audioSpectrogram', namespace='iblmic')

# -- Plot spectrogram
tlims = TF['times_mic'][[0, -1]].flatten()
flims = TF['frequencies'][0, [0, -1]].flatten()
fig = plt.figure(figsize=[16, 7])
ax = plt.axes()
im = ax.imshow(20 * np.log10(TF['power'].T), aspect='auto', cmap=plt.get_cmap('magma'),
               extent=np.concatenate((tlims, flims)),
               origin='lower')
ax.set_xlabel(r'Time (s)')
ax.set_ylabel(r'Frequency (Hz)')
plt.colorbar(im)
im.set_clim(-100, -60)

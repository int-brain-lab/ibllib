"""
Plot audio spectrogtam
======================
For a given session eid (ephys session), plot spectrogram of sound recorded via the microphone.
Example of using soundfile to read in .flac file extensions
"""
# Author: Gaelle Chapuis

from ibllib.io.extractors.training_audio import welchogram
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
# import alf.io
from oneibl.one import ONE

one = ONE()

dataset_types = ['_iblrig_micData.raw']

eid = '9a7e3a4b-8b68-4817-81f1-adb0f48088eb'  # TEST EXAMPLE

one.load(eid, dataset_types=dataset_types, download_only=True)
session_path = one.path_from_eid(eid)

# -- Get raw data
filename = session_path.joinpath('raw_behavior_data', '_iblrig_micData.raw.flac')

with open(filename, 'rb') as f:
    wav, fs = sf.read(f)

# -- Compute spectrogram over first 2 minutes
t_idx = 120 * fs
tscale, fscale, W, detect = welchogram(fs, wav[:t_idx])

# -- Put data into single variable
TF = {}

TF['power'] = W.astype(np.single)
TF['frequencies'] = fscale[None, :].astype(np.single)
TF['onset_times'] = detect
TF['times_mic'] = tscale[:, None].astype(np.single)

# # -- Plot spectrogram
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
plt.show()

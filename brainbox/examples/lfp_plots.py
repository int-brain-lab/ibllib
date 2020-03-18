import numpy as np
import brainbox as bb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import alf.io as ioalf
from oneibl.one import ONE
from ibllib.io import spikeglx

# Download data
one = ONE()
eid = one.search(subject='ZM_2240', date_range=['2020-01-23', '2020-01-23'])
lf_path = one.load(eid[0], dataset_types=['ephysData.raw.lf', 'ephysData.raw.meta',
                                          'ephysData.raw.ch'],
                   download_only=True)[0]

# Read in data
raw = spikeglx.Reader(lf_path)
signal = raw.read(nsel=slice(None, 100000, None), csel=slice(None, None, None))[0]
signal = np.rot90(signal)

freqs, psd = bb.lfp.power_spectrum(signal, raw.fs)

# %% Create plots

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[:, 0])
pos = ax1.imshow(np.log10(psd[:, :150]), aspect='auto', vmax=-5)
ax1.set(ylabel='Channels', xlabel='Frequency (Hz)')
cbar = fig.colorbar(pos, ax=ax1)
cbar.set_label('log10 power ($V^2$)', rotation=270, labelpad=10)

ax2 = fig.add_subplot(gs[0, 1])
random_channel = np.random.choice(raw.nc)
ax2.plot(freqs, psd[random_channel, :])
ax2.set(xlim=[1, 150], yscale='log', ylabel='Power ($V^2$)',
        xlabel='Frequency (Hz)', title='Channel %d' % random_channel)

ax3 = fig.add_subplot(gs[1, 1])
random_channel = np.random.choice(raw.nc)
ax3.plot(freqs, psd[random_channel, :])
ax3.set(xlim=[1, 150], yscale='log', ylabel='Power ($V^2$)',
        xlabel='Frequency (Hz)', title='Channel %d' % random_channel)

plt.tight_layout(pad=5)

from pathlib import Path
import alf.io
import matplotlib.pyplot as plt
from brainbox.processing import bincount2D
import ibllib.plots as iblplt
import numpy as np
from sklearn.linear_model import LogisticRegression
import smoothing

main_path = Path('/home/sebastian/Desktop/')
SES = {
    'A': main_path.joinpath(Path('ZM_1735/2019-08-01/001')),
    'B': main_path.joinpath(Path('ibl_witten_04/2019-08-04/002')),
    'C': main_path.joinpath(Path('ZM_1736/2019-08-09/004')),
    'D': main_path.joinpath(Path('ibl_witten_04/2018-08-11/001')),
    'E': main_path.joinpath(Path('KS005/2019-08-29/001')),
    'F': main_path.joinpath(Path('KS005/2019-08-30/001')),
}

# select a session from the bunch
sid = 'F'
ses_path = Path(SES[sid])

# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')

# compute raster map as a function of cluster number

T_BIN = 0.1
R, times, clusts = bincount2D(spikes['times'], spikes['clusters'], T_BIN)
y = (trials.choice + 1) / 2

#events = (trials.response_times / T_BIN).astype(int)
#events = (trials.stimOn_times / T_BIN).astype(int)
#events = (trials.feedback_times / T_BIN).astype(int)
events = (trials.feedback_times[trials.feedbackType == 1] / T_BIN).astype(int)

window_L, window_R = (100, 100)

indexes = np.arange(window_L + window_R)
indexes = np.tile(indexes, (events.size, 1))
indexes += events[:, np.newaxis]


for clust in range(clusts[-1]):
    if np.sum(R[clust]) < 12000:
        continue

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
    ax1.set_title("Raster plot and PSTH of cluster {} around reward delivery".format(clust))
    ax1.imshow(R[clust, indexes], aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4)
    ax1.axvline(window_L, 0, events.size, c='r')

    ax2.plot(np.sum(R[clust, indexes], axis=0))
    ax2.axvline(window_L, 0, events.size, c='r')

    plt.xticks(np.arange(0, window_L + window_R + 1, (window_L + window_R) / 5), np.arange(-window_L, window_R + 1, (window_L + window_R) / 5).astype(int))
    ax2.set_xlabel("Time in 10 ms")
    ax2.set_ylabel("Summed activity")
    ax1.set_ylabel("Spikes in individual trials")

    plt.show()

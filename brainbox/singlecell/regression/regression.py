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
sid = 'A'
ses_path = Path(SES[sid])

# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')

# compute raster map as a function of cluster number


T_BIN = .001
R, times, clusts = bincount2D(spikes['times'], spikes['clusters'], T_BIN)
y = (trials.choice + 1) / 2
max = 0
max_clust = 0


quit()
clust = 325
temp = smoothing.smoothing_times(spikes.times[spikes.clusters == clust], frame_length=0.01, window_type='square', window_length=50)

"""for i in range(30):



    # plot raster map
    plt.plot(temp)
    # plot trial start and reward time
    reward = trials['feedback_times'][trials['feedbackType'] == 1]
    iblplt.vertical_lines(trials.response_times[trials.choice == 1] * 100, ymin=0, ymax=50,
                     color='k', linewidth=0.5, label='choice right')
    iblplt.vertical_lines(trials.response_times[trials.choice == -1] * 100, ymin=0, ymax=50,
                     color='r', linewidth=0.5, label='choice left')
    plt.xlabel('Time (s)')
    plt.ylabel('Cluster #')
    plt.xlim(left=i * 10000, right=10000 + i * 10000)
    plt.legend()
    plt.show()"""



for clust in range(clusts[-1]):
#for clust in [383]:
    print(clust)
    if clust in [19, 31, 62, 70, 71, 335, 355, 380]:
        continue

    temp = smoothing.smoothing_times(spikes.times[spikes.clusters == clust], frame_length=0.01, window_type='square', window_length=50)
    X = temp[(trials.response_times * 100 - 25).astype(int)]
    clf = LogisticRegression(solver='lbfgs').fit(X.reshape(-1, 1), y)
    print(clf.coef_)

    if np.abs(clf.coef_) > max:
        max = np.abs(clf.coef_)
        max_clust = clust

    continue
    # plot raster map
    plt.plot(temp)
    # plot trial start and reward time
    reward = trials['feedback_times'][trials['feedbackType'] == 1]
    iblplt.vertical_lines(trials.response_times[trials.choice == 1] * 100, ymin=0, ymax=1,
                     color='k', linewidth=0.5, label='choice right')
    iblplt.vertical_lines(trials.response_times[trials.choice == -1] * 100, ymin=0, ymax=1,
                     color='r', linewidth=0.5, label='choice left')
    plt.xlabel('Time (s)')
    plt.ylabel('Cluster #')
    plt.xlim(left=0, right=till)
    plt.legend()
    plt.show()

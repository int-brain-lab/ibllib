import matplotlib.pyplot as plt

import alf.io
import brainbox.plot as bbp

from oneibl.one import ONE

one = ONE()
eid = one.search(lab='wittenlab', date='2019-08-04')
datasets = one.load(eid, download_only=True)
ses_path = datasets[0].local_path.parent  # local path where the data has been downloaded

spikes = alf.io.load_object(ses_path, 'spikes')
trials = alf.io.load_object(ses_path, 'trials')

# For a simple peth plot without a raster, all we need to input is spike times, clusters, event
# times, and the identity of the cluster we want to plot, e.g. in this case cluster 121

ax = bbp.peri_event_time_histogram(spikes.times, spikes.clusters, trials.goCue_times, 121)

# Or we can include a raster plot below the PETH:

fig = plt.figure()
ax = plt.gca()
bbp.peri_event_time_histogram(spikes.times,  # Spike times first
                              spikes.clusters,  # Then cluster ids
                              trials.goCue_times,  # Event markers we want to plot against
                              121,  # Identity of the cluster we plot
                              t_before=0.4, t_after=0.4,  # Time before and after the event
                              error_bars='sem',  # Whether we want Stdev, SEM, or no error
                              include_raster=True,  # adds a raster to the bottom
                              n_rasters=55,  # How many raster traces to include
                              ax=ax)  # Make sure we plot to the axis we created

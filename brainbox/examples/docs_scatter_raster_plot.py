import numpy as np
from brainbox.plot.ephys_plots import scatter_raster_plot
from oneibl.one import ONE
import matplotlib.pyplot as plt
import matplotlib

one = ONE()

eid = '671c7ea7-6726-4fbe-adeb-f89c2c8e489b'
probe = 'probe00'

spikes = one.load_object(eid, obj='spikes', collection=f'alf/{probe}')
metrics = one.load_dataset(eid, dataset='clusters.metrics', collection=f'alf/{probe}')

# Find the clusters that have been labelled as good and their corresponding spike indices
good_clusters = np.where(metrics.label == 1)
spike_idx = np.where(np.isin(spikes['clusters'], good_clusters))[0]

# Also filter for nans in amplitude and depth
kp_idx = np.where(~np.isnan(spikes['depths'][spike_idx]) & ~np.isnan(spikes['amps'][spike_idx]))[0]

# Get ScatterPlot object
data = scatter_raster_plot(spikes['amps'][kp_idx], spikes['depths'][kp_idx],
                           spikes['times'][kp_idx])

# Add v lines 10s after start and 10s before end or recording
x1 = np.min(spikes['times'][kp_idx] + 100)
x2 = np.max(spikes['times'][kp_idx] - 100)
data.add_lines(pos=x1, orientation='v', style='solid', width=10, color='r')
data.add_lines(pos=x2, orientation='v', style='dashed', width=8, color='k')

plot_dict = data.convert2dict()

fig, ax = plt.subplots()
scat = ax.scatter(x=plot_dict['data']['x'], y=plot_dict['data']['y'], c=plot_dict['color']/255,
                  s=plot_dict['marker_size'], cmap=plot_dict['cmap'],
                  vmin=plot_dict['clim'][0], vmax=plot_dict['clim'][1])
ax.hlines([20, 3840], *ax.get_xlim(), linestyles='dashed', linewidth=3, colors='k')

norm = matplotlib.colors.Normalize(vmin=plot_dict['clim'][0], vmax=plot_dict['clim'][1], clip=True)
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plot_dict['cmap']), ax=ax)
cbar.set_label(plot_dict['labels']['clabel'])
plt.show()


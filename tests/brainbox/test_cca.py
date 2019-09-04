from brainbox.population import cca
import numpy as np
import matplotlib.pylab as plt
# test plotting
corrs = np.array([.6, .2, .1, .001])
errs = np.array([.1, .05, .04, .0005])
fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
cca.plot_correlations(corrs, errs, ax=ax1, color='blue')
cca.plot_correlations(corrs * .1, errs, ax=ax1, color='orange')
plt.show()

# Shuffle data
# ...
# fig, ax1 = plt.subplots(1,1,figsize(10,10))
# plot_correlations(corrs, ... , ax=ax1, color='blue')
# plot_correlations(shuffled_coors, ..., ax=ax1, color='red')
# plt.show()




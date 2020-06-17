
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(4)
total_activities = np.random.rand(10, 5)
sns.heatmap(np.corrcoef(total_activities), square=True, cbar=False)

plt.plot([-1, 11], [2, 2], 'k', clip_on=False, linewidth=2)

plt.show()

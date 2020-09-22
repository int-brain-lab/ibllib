'''
Get all trajectories from given provenance,
that have better QC than given status,
plot as scatter
'''

import matplotlib.pyplot as plt
import pandas as pd
from oneibl.one import ONE
one = ONE()

traj = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                     django='probe_insertion__session__project__name__'
                            'icontains,ibl_neuropixel_brainwide_01,'
                            'probe_insertion__session__qc__lt,50')

# Ephys aligned histology track, Histology track, Micro-manipulator, Planned

# QC_CHOICES = [
#     (50, 'CRITICAL',),
#     (40, 'FAIL',),
#     (30, 'WARNING',),
#     (0, 'NOT_SET',),
#     (10, 'PASS',),
# ]

x = [p['x'] for p in traj]
y = [p['y'] for p in traj]


# Read CSV containing all x / y positions to be done
data = pd.read_csv("/Users/gaelle/Documents/Git/Scrapbook/Needles/Firstpassmap_x_y.csv")
x_fm = data['ap_um']
y_fm = data['ml_um']

# Plot
fig, ax = plt.subplots()
ax.scatter(x_fm, y_fm, color='black', alpha=0.1)
ax.scatter(x, y, color='green', alpha=0.4)

ax.set_xlim(4000, -8000)  # decreasing x

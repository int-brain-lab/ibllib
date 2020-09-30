'''
Get all trajectories from given provenance,
that have better QC than given status,
plot as scatter
'''

from ibl_pipeline import acquisition
from ibl_pipeline.analyses import behavior as behavior_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from oneibl.one import ONE

one = ONE()

traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
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

eids_traj = np.unique([p['session']['id'] for p in traj])
print(f'N sess QC<Critical (Alyx database) : {len(eids_traj)}')

# DATAJOINT query to combine with behavioral criterion


def datadj_to_session_uids_unique(datadj):
    data_uids = datadj.proj('session_uuid')
    dataframe = data_uids.fetch(format='frame').reset_index()

    uids_list = dataframe['session_uuid'].values.tolist()
    uids_unique = np.unique([str(_) for _ in uids_list])
    return uids_unique

data_all = acquisition.Session & [{'session_uuid': i_e} for i_e in eids_traj]

data_good = (acquisition.Session & [{'session_uuid': i_e} for i_e in eids_traj] &
             (behavior_analysis.SessionTrainingStatus & 'good_enough_for_brainwide_map=1'))

data_notpass = (acquisition.Session & [{'session_uuid': i_e} for i_e in eids_traj] &
                (behavior_analysis.SessionTrainingStatus & 'good_enough_for_brainwide_map=0'))

data_notcomputedbehav = (acquisition.Session & [{'session_uuid': i_e} for i_e in eids_traj]) -\
                        behavior_analysis.SessionTrainingStatus

eids_all = datadj_to_session_uids_unique(data_all)  # Sanity check, should be same number as per Alyx
eids_good = datadj_to_session_uids_unique(data_good)
eids_notpass = datadj_to_session_uids_unique(data_notpass)
eids_notcomputedbehav = datadj_to_session_uids_unique(data_notcomputedbehav)

print(f'N sess on DJ with similar eids than on Alyx (sanity check)  : {len(eids_all)} \n \n'
      f'N sess good behav on DJ : {len(eids_good)} \n'
      f'N sess fail behav on DJ : {len(eids_notpass)} \n'
      f'N sess behav not computed on DJ : {len(eids_notcomputedbehav)} \n'
      f'TOTAL sessions according to DJ queries above : {len(eids_good)+len(eids_notpass)+len(eids_notcomputedbehav)}\n'
      )

# Get ml / ap of only those that are good
traj_dict = {str(p['session']['id']): (p['x'], p['y']) for p in traj}
ml = np.array([traj_dict[eid][0] for eid in eids_good])
ap = np.array([traj_dict[eid][1] for eid in eids_good])

# Read CSV containing all x / y positions to be done
data = pd.read_csv(
    "/Users/gaelle/Documents/Git/Scrapbook/Needles/Firstpassmap_x_y.csv")
ap_fm = data['ap_um']
ml_fm = data['ml_um']

# Plot
fig, ax = plt.subplots()
ax.scatter(ap_fm, ml_fm, color='black', alpha=0.1)
ax.scatter(ap, ml, color='green', alpha=0.4)

ax.set_xlim(4000, -8000)  # decreasing x

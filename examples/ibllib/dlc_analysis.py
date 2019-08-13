from pathlib import Path
import alf.io
from oneibl.one import ONE
from dlc_pupil import pupil_features
import matplotlib.pyplot as plt

# Find and load data from ONE
one = ONE()
one.list(None, keyword='data')
eids = one.search(dataset_types='_ibl_leftCamera.dlc')
eid = eids[0]
dtypes = ['_ibl_leftCamera.dlc', '_iblrig_leftCamera.timestamps']
d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
ses_path = Path(d.local_path[0]).parent
segments = alf.io.load_object(ses_path, '_ibl_leftCamera')

# Fit circle on pupil points
vec_x = [segments['pupil_top_r_x'][0:1000], segments['pupil_left_r_x'][0:1000], segments['pupil_right_r_x'][0:1000], segments['pupil_bottom_r_x'][0:1000]]
vec_y = [segments['pupil_top_r_y'][0:1000], segments['pupil_left_r_y'][0:1000], segments['pupil_right_r_y'][0:1000], segments['pupil_bottom_r_y'][0:1000]]
x, y, diameter = pupil_features(vec_x, vec_y)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
ax1.plot(x)
ax2.plot(y)
ax3.plot(diameter)
plt.show()

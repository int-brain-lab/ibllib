import matplotlib.pyplot as plt
import pandas as pd

from oneibl.one import ONE
from ibllib.time import isostr2date

# import sys
# sys.path.extend('/home/owinter/PycharmProjects/WGs/BehaviourAnaysis/python')
from load_mouse_data import get_behavior
from behavior_plots import plot_psychometric

one = ONE()
# https://alyx.internationalbrainlab.org/admin/actions/session/e752b02d-b54d-4373-b51e-0b31be5f8ee5/change/
# first get the subject information
subject_details = one.alyx.rest('subjects', 'read', 'IBL_14')

# plot the weight curve
# https://alyx.internationalbrainlab.org/admin-actions/water-history/37c8f897-cbcc-4743-bad6-764ccbbfb190
wei = pd.DataFrame(subject_details['weighings'])
wei['date_time'].apply(isostr2date)
wei.sort_values('date_time', inplace=True)
plt.plot(wei.date_time, wei.weight)

# now let's get some session information
ses_ids = one.search(subjects='IBL_14', date_range='2018-11-27')
print(one.list(ses_ids[0]))
df = get_behavior('IBL_14', date_range='2018-11-27')
plt.figure()
plot_psychometric(df, ax=plt.axes(), color="orange")

import matplotlib.pyplot as plt
import pandas as pd

from oneibl.one import ONE
from ibllib.time import isostr2date

one = ONE()
wei = one._alyxClient.get('/weighings?nickname=437')

for w in wei:
    w['date_time'] = isostr2date(w['date_time'])

wei = pd.DataFrame(wei)
wei.sort_values('date_time', inplace=True)
plt.plot(wei.date_time, wei.weight)

from pathlib import Path
import pandas as pd
import numpy as np
import alf.io
from oneibl.one import ONE
import json

# Find and load data from ONE
one = ONE()
one.list(None, keyword='data')
eids = one.search(dataset_types='_ibl_leftCamera.dlc')
eid = eids[0]
dtypes = ['_ibl_leftCamera.dlc', '_iblrig_leftCamera.timestamps']
d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
ses_path = Path(d.local_path[0]).parent
segments = alf.io.load_object(ses_path, '_ibl_leftCamera')

# Load in data
with open(d.local_path[0], 'r') as myfile:
    data = myfile.read()
names = json.loads(data)
traces = np.load(d.local_path[1]) # Load in DLC traces
ts = pd.read_csv(d.local_path[2], delimiter=' ') # Load in timestamps


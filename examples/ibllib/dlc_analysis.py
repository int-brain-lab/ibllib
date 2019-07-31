from pathlib import Path
import pandas as pd

import alf.io
from oneibl.one import ONE

one = ONE()

one.list(None, keyword='data')

eids = one.search(dataset_types='_ibl_leftCamera.dlc')
eid = eids[0]

dtypes = ['_ibl_leftCamera.dlc', '_iblrig_leftCamera.timestamps']
d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
ses_path = Path(d.local_path[0]).parent
segments = alf.io.load_object(ses_path, '_ibl_leftCamera')


ts = pd.read_csv(d.local_path[2], delimiter=' ')


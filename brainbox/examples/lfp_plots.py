import numpy as np
import brainbox as bb
import alf.io as ioalf
from oneibl.one import ONE


# Download data
one = ONE()
eid = one.search(subject='ZM_2240', date_range=['2020-01-22', '2020-01-22'])
lf_path = one.load(eid[0], dataset_types=['ephysData.raw.lf'], download_only=True)[0]
raw = ioalf.load_object(lf_path)

#

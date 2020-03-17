import numpy as np
import brainbox as bb
import alf.io as ioalf
from oneibl.one import ONE
from ibllib.io import spikeglx

# Download data
one = ONE()
eid = one.search(subject='ZM_2240', date_range=['2020-01-23', '2020-01-23'])
lf_path = one.load(eid[0], dataset_types=['ephysData.raw.lf', 'ephysData.raw.meta',
                                          'ephysData.raw.ch'],
                   download_only=True)[0]
raw = spikeglx.Reader(lf_path)
data = raw.read(nsel=slice(0, 100, None))

#

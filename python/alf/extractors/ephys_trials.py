from pathlib import Path

from ibllib.io import spikeglx
import ibllib.dsp as dsp
session_path = '/mnt/s0/Data/Subjects/ZM_1150/2019-05-07/001'

SYNC_BATCH_SIZE_SAMPLES = 2 ** 18  # number of samples to read at once in bin file for sync
OVERLAP = 30000  # number of overlapping samples for sync signal (1 sec at 30 kHz)

#
session_path = Path(session_path)
raw_ephys_path = session_path / 'raw_ephys_data'


raw_ephys_apfiles = raw_ephys_path.rglob('*.ap.bin')



#
for raw_ephys_apfile in raw_ephys_apfiles:
    sr = spikeglx.Reader(raw_ephys_apfile)
    wg = dsp.WindowGenerator(sr.ns, SYNC_BATCH_SIZE_SAMPLES, OVERLAP)

    for sl in wg.slice:
        print(sl)
        break
    ss = sr.read_sync(sl)

##
from ibllib.plots import traces
traces(ss, fs=sr.fs)


# dsp.WindowGenerator
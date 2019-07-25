import ibllib.dsp as dsp
import ibllib.io.spikeglx
import ibllib.io.extractors.ephys_fpga

BATCH_SIZE_SAMPLES = 50000

# full path to the raw ephys
raw_ephys_apfile = ('/datadisk/Data/Subjects/ZM_1150/2019-05-07/001/raw_ephys_data/probe_right/'
                    'ephysData_g0_t0.imec.ap.bin')
output_path = '/home/olivier/scratch'

# load reader object, and extract sync traces
sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
sync = ibllib.io.extractors.ephys_fpga._sync_to_alf(sr, output_path, save=False)

# if the data is needed as well, loop over the file
# raw data contains raw ephys traces, while raw_sync contains the 16 sync traces
wg = dsp.WindowGenerator(sr.ns, BATCH_SIZE_SAMPLES, overlap=1)
for first, last in wg.firstlast:
    rawdata, rawsync = sr.read_samples(first, last)
    wg.print_progress()

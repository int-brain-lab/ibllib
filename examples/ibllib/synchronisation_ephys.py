import ibllib.io.spikeglx
import ibllib.io.extractors.ephys_fpga

# full path to the raw ephys
raw_ephys_apfile = ('/datadisk/Data/Subjects/ZM_1150/2019-05-07/001/raw_ephys_data/probe_right/'
                    'ephysData_g0_t0.imec.ap.bin')
output_path = '/home/olivier/scratch'

# load reader object, and extract sync traces
sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
sync = ibllib.io.extractors.ephys_fpga._sync_to_alf(sr, output_path, save=False)

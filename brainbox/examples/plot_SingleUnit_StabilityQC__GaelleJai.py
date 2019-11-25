
# CODE from Jai :
# https://github.com/int-brain-lab/ibllib/blob/f495abac5812fb7e5bb161418acfae71ee13d6a1/brainbox/plot/plot.py
#
#  Work on sessions:
#
# EPHYS:	mainenlab   ZM_2407	    2019-11-05  	Guido
#           Dual	3A	_iblrig_ephysChoiceWorld6.0.6


import brainbox as bb
import alf.io as aio
# import ibllib.ephys.spikes as e_spks
from oneibl.one import ONE
one = ONE()

# -- GET EID FOR SPECIFIC SESSION
# Check that each dataset type needed is present
dataset_types = ['ephysData.raw.ap', 'spikes.times', 'clusters.depths']
eid, sinfo = one.search(
    datasets=dataset_types,
    subjects='ZM_2407',
    date_range='2019-11-05',
    details=True)
assert(len(eid) > 0), 'No EID found with those search terms'
dtypes_session = one.list(eid)[0]
if not set(dataset_types).issubset(set(dtypes_session)):
    missing_dtypes = [dt for dt in dataset_types if dt not in dtypes_session]
    raise ValueError('Missing datasets: ' + ','.join(missing_dtypes))

# In case the above did not run,
# you can find all sessions containing raw data by looking into sinfos:
# eids, sinfos = one.search(
#    datasets=['ephysData.raw.ap', 'spikes.times'], task_protocol='certification', details=True)


# -- LOAD ALF DATA
alf_info = one.load(eid, download_only=True)  # Download the whole alf folder
alf_info = alf_info[0]
path_alf = alf_info.local_path

# -- LOAD RAW DATA
file_paths_raw = one.load(eid, dataset_types='ephysData.raw.ap', download_only=True)
file_paths_raw = file_paths_raw[0]
path_ephys = file_paths_raw.local_path  # TODO CHECK THIS WORKS

# -- Fig 1
spks = aio.load_object(path_alf, 'spikes')
fig, var_vals, p_vals = bb.plot.feat_vars(spks)


# -- Fig 2
clstrs = aio.load_object(path_alf, 'clusters')
fig1, wf1, wf2 = bb.plot.single_unit_wf_comp(path_ephys, spks, clstrs, unit=1)

# -- Fig 3
# Get a units bunch, and plot waveforms for unit2 from the first to second minute
# across 15 channels.
units = bb.processing.get_units_bunch(spks, ['times'])
ts1 = units['times']['2'][:50]
ts2 = units['times']['2'][-50:]
fig2, wf1_2, wf2_2 = bb.plot.single_unit_wf_comp(path_ephys, spks, clstrs, unit=1, ts1=ts1,
                                                 ts2=ts2)

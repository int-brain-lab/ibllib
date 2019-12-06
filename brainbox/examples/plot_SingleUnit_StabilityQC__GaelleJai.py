
# CODE from Jai :
# https://github.com/int-brain-lab/ibllib/blob/f495abac5812fb7e5bb161418acfae71ee13d6a1/brainbox/plot/plot.py
#
#  Work on sessions:
#
# EPHYS:	mainenlab   ZM_2407	    2019-11-05  	Guido
#           Dual	3A	_iblrig_tasks_ephys_certification


import brainbox as bb
import alf.io as aio
# import ibllib.ephys.spikes as e_spks
from oneibl.one import ONE
one = ONE()

need_load = bool(0)

if need_load:
    # This is only useful if you do not have the data accessible on the machine

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
    path_ephys = file_paths_raw.local_path
    # TODO get the file
else:
    path_ephys = '/Users/gaelle/Downloads/FlatIron/mainenlab/Subjects/ZM_2407/2019-11-05/003/raw_ephys_data/probe_00/_spikeglx_ephysData_probe00.raw_g0_t0.imec.ap.cbin'
    path_alf = '/Users/gaelle/Downloads/FlatIron/mainenlab/Subjects/ZM_2407/2019-11-05/003/alf/probe_00'

# -- Get spks and clusters
spks = aio.load_object(path_alf, 'spikes')
clstrs = aio.load_object(path_alf, 'clusters')

# Fig 1 -- ALL UNITS
# Create a bar plot of the variances of the spike amplitudes for each unit.
fig, var_vals, p_vals = bb.plot.feat_vars(spks)

# Fig 2 -- SINGLE UNIT
# Plot cutoff line indicating the fraction of spikes missing from a unit based on the recorded
#  unit's spike amplitudes, assuming the distribution of the unit's spike amplitudes is symmetric.
fig, fraction_missing = bb.plot.feat_cutoff(spks, 1)

# Fig 3 -- SINGLE UNIT
# Compare first and last 100 spike waveforms for unit1, across 20 channels around the channel
# of max amplitude, and compare the first and last 50 spike waveforms for unit2, across 15
# channels around the mean.
fig1, wf1, wf2 = bb.plot.single_unit_wf_comp(path_ephys, spks, clstrs, unit=1)

# -- Fig 4 -- SINGLE UNIT
# Get a units bunch, and plot waveforms for unit2 from the first to second minute
# across 15 channels.
units = bb.processing.get_units_bunch(spks, ['times'])
ts1 = units['times']['2'][:50]
ts2 = units['times']['2'][-50:]
fig2, wf1_2, wf2_2 = bb.plot.single_unit_wf_comp(path_ephys, spks, clstrs, unit=1, ts1=ts1,
                                                 ts2=ts2)

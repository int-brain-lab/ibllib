from oneibl.one import ONE
import brainbox.io.one as bbone

one = ONE()




spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
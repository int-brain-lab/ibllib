import numpy as np
import alf.io
from brainbox.processing import bincount2D
from oneibl.one import ONE
from pathlib import Path


def check_for_quiet_spike_periods(eid):
    '''
    This functions reads in spikes for a given session,
    bins them into time bins and computes for how many of them,
    there is too little activity across all channels such that
    this must be an artefact (saturation)
    '''

    T_BIN = 0.2  # time bin in sec
    ACT_THR = 0.05  # maximal activity for saturated segment
    print('Bin size: %s [ms]' % T_BIN)
    print('Activity threshold: %s [fraction]' % ACT_THR)

    probes = ['probe00', 'probe01']
    probeDict = {'probe00': 'probe_left', 'probe01': 'probe_right'}

    one = ONE()
    dataset_types = ['spikes.times', 'spikes.clusters']
    D = one.load(eid, dataset_types=dataset_types, dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent
    print(alf_path)

    for probe in probes:
        probe_path = alf_path / probe
        if not probe_path.exists():
            probe_path = alf_path / probeDict[probe]
            if not probe_path.exists():
                print("% s doesn't exist..." % probe)
                continue

        spikes = alf.io.load_object(probe_path, 'spikes')

        # bin spikes
        R, times, Clusters = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN)

        saturated_bins = np.where(np.mean(R, axis=0) < 0.15)[0]

        print(probe)
        print('Number of saturated bins: %s of %s' %
              (len(saturated_bins), len(times)))

        if len(saturated_bins) > 1:
            print('WARNING: Saturation present!')

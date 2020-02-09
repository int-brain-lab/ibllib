import alf.io
from oneibl.one import ONE

one = ONE()


def load_spike_sorting(eid, dataset_types=None):
    """
    From an eid, hits the Alyx database and downloads a standard set of dataset types to perform
    analysis.
    :param eid:
    :param dataset_types: additional spikes/clusters objects to add to the standard list
    :return:
    """
    # This is a first draft, no safeguard, no error handling and a draft dataset list.
    session_path = one.path_from_eid(eid)
    dtypes = [
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'spikes.clusters',
        'spikes.times',
        'probes.description',
    ]
    if dataset_types:
        dtypes = list(set(dataset_types + dtypes))

    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')
    spikes = []
    clusters = []
    for i, _ in enumerate(probes['description']):
        probe_path = session_path.joinpath('alf', probes['description'][i]['label'])
        clusters.append(alf.io.load_object(probe_path, object='clusters'))
        spikes.append(alf.io.load_object(probe_path, object='spikes'))

    return spikes, clusters

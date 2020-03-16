import alf.io
from oneibl.one import ONE


def load_ephys_session(eid, one=None, dataset_types=None):
    spikes, clusters = load_spike_sorting(eid, one=None, dataset_types=None)
    trials = one.load_object(eid, obj='trials')

    return spikes, clusters, trials


def load_spike_sorting(eid, one=None, dataset_types=None):
    """
    From an eid, hits the Alyx database and downloads a standard set of dataset types to perform
    analysis.
    :param eid:
    :param dataset_types: additional spikes/clusters objects to add to the standard list
    :return:
    """
    if not one:
        one = ONE()
    # This is a first draft, no safeguard, no error handling and a draft dataset list.
    session_path = one.path_from_eid(eid)
    if not session_path:
        print("no session path")
        return (None, None), 'no session path'

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
    try:
        probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')
    except FileNotFoundError:
        print("no probes")
        return (None, None), 'no probes'
    spikes = {}
    clusters = {}
    for i, _ in enumerate(probes['description']):
        probe_path = session_path.joinpath('alf', probes['description'][i]['label'])
        try:
            cluster = alf.io.load_object(probe_path, object='clusters')
        except FileNotFoundError:
            print("one probe missing")
            return (None, None), "one probe missing"
        spike = alf.io.load_object(probe_path, object='spikes')
        label = probes['description'][i]['label']
        clusters[label] = cluster
        spikes[label] = spike

    return (spikes, clusters), ''

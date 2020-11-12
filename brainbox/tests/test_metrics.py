import numpy as np
from brainbox.metrics import quick_unit_metrics

REC_LEN_SECS = 1000
fr = 200


def multiple_spike_trains(firing_rates=None, rec_len_secs=1000, cluster_ids=None):
    """

    :param firing_rates: list or np.array of firing rates (spikes per second)
    :param rec_len_secs: recording length in seconds
    :return: spike_times, spike_amps, spike_clusters
    """
    if firing_rates is None:
        firing_rates = np.random.randint(150, 600, 10)
    if cluster_ids is None:
        cluster_ids = np.arange(firing_rates.size)
    st = np.empty(0)
    sa = np.empty(0)
    sc = np.empty(0)
    for i, firing_rate in enumerate(firing_rates):
        t, a = single_spike_train(firing_rate=firing_rate, rec_len_secs=rec_len_secs)
        st = np.r_[st, t]
        sa = np.r_[sa, a]
        sc = np.r_[sc, np.zeros(t.size, dtype=np.int32) + cluster_ids[i]]

    ordre = st.argsort()
    st = st[ordre]
    sa = sa[ordre]
    sc = sc[ordre]
    return st, sa, sc


def single_spike_train(firing_rate=200, rec_len_secs=1000):
    """
    Basic spike train generator following a poisson process for spike-times and
    :param firing_rate:
    :param rec_len_secs:
    :return: spike_times (secs) , spike_amplitudes (V)
    """

    # spike times: exponential decay prob
    st = np.cumsum(- np.log(np.random.rand(int(rec_len_secs * firing_rate * 1.5))) / firing_rate)
    st = st[:np.searchsorted(st, rec_len_secs)]

    # spike amplitudes: log normal (estimated from an IBL session)
    nspi = np.size(st)
    sa = np.exp(np.random.normal(5.5, 0.5, nspi)) / 1e6  # output is in V

    return st, sa


def test_clusters_metrics():
    frs = [3, 200, 259, 567]  # firing rates
    t, a, c = multiple_spike_trains(firing_rates=frs, rec_len_secs=1000, cluster_ids=[0, 1, 3, 4])
    d = np.sin(2 * np.pi * c / 1000 * t) * 100  # sinusoidal shift where cluster id drives period
    dfm = quick_unit_metrics(c, t, a, d)

    assert np.allclose(dfm['amp_median'] / np.exp(5.5) * 1e6, 1, rtol=1.1)
    assert np.allclose(dfm['amp_std_dB'] / 20 * np.log10(np.exp(0.5)), 1, rtol=1.1)
    assert np.allclose(dfm['drift'], np.array([0, 1, 3, 4]) * 100 * 4 * 3.6, rtol=1.1)

    np.allclose(dfm['firing_rate'], frs)
    # probe_path = "/datadisk/FlatIron/m1ainenlab/Subjects/ZFM-01577/2020-11-04/001/alf/probe00"
    # spikes = alf.io.load_object(probe_path, 'spikes')
    # quick_unit_metrics(spikes['clusters'], spikes['times'], spikes['amps'], spikes['depths'])

import numpy as np
import alf.io
from brainbox.metrics import quick_unit_metrics

REC_LEN_SECS = 1000
fr = 200


def multiple_spike_trains(firing_rates=None, rec_len_secs=1000):
    """

    :param firing_rates:
    :param rec_len_secs:
    :return: spike_times, spike_amps, spike_clusters
    """
    if firing_rates is None:
        firing_rates = np.random.randint(150, 600, 10)
    st = np.empty(0)
    sa = np.empty(0)
    sc = np.empty(0)
    for i, firing_rate in enumerate(firing_rates):
        t, a = single_spike_train(firing_rate=firing_rate, rec_len_secs=rec_len_secs)
        st = np.r_[st, t]
        sa = np.r_[sa, a]
        sc = np.r_[sc, np.zeros(t.size, dtype=np.int32) + i]

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
    :return:
    """

    # spike times: exponential decay prob
    st = np.cumsum(- np.log(np.random.rand(int(rec_len_secs * firing_rate * 1.5))) / firing_rate)
    st = st[:np.searchsorted(st, rec_len_secs)]

    # spike amplitudes: log normal (estimated from an IBL session)
    nspi = np.size(st)
    sa = np.exp(np.random.normal(5.5, 0.5, nspi)) / 1e6

    return st, sa


def test_clusters_metrics():
    t, a, c = multiple_spike_trains(firing_rates=[3, 200, 259, 567], rec_len_secs=1000)
    dfm = quick_unit_metrics(c, t, a, t * 0)

    # probe_path = "/datadisk/FlatIron/mainenlab/Subjects/ZFM-01577/2020-11-04/001/alf/probe00"
    # spikes = alf.io.load_object(probe_path, 'spikes')
    # quick_unit_metrics(spikes['clusters'], spikes['times'], spikes['amps'], spikes['depths'])


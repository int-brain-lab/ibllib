import numpy as np
from brainbox.metrics import electrode_drift

REC_LEN_SECS = 1000
fr = 200


def multiple_spike_trains(firing_rates=None, rec_len_secs=1000, cluster_ids=None,
                          amplitude_noise=20 * 1e-6):
    """
    :param firing_rates: list or np.array of firing rates (spikes per second)
    :param rec_len_secs: recording length in seconds
    :return: spike_times, spike_amps, spike_clusters
    """
    if firing_rates is None:
        firing_rates = np.random.randint(150, 600, 10)
    if cluster_ids is None:
        cluster_ids = np.arange(firing_rates.size)
    ca = np.exp(np.random.normal(5.5, 0.5, firing_rates.size)) / 1e6  # output is in V
    st = np.empty(0)
    sc = np.empty(0)
    for i, firing_rate in enumerate(firing_rates):
        t = generate_spike_train(firing_rate=firing_rate, rec_len_secs=rec_len_secs)
        st = np.r_[st, t]
        sc = np.r_[sc, np.zeros(t.size, dtype=np.int32) + cluster_ids[i]]

    ordre = st.argsort()
    st = st[ordre]
    sc = np.int32(sc[ordre])
    sa = np.maximum(ca[sc] + np.random.randn(st.size) * amplitude_noise, 25 * 1e-6)
    return st, sa, sc


def generate_spike_train(firing_rate=200, rec_len_secs=1000):
    """
    Basic spike train generator following a poisson process for spike-times and
    :param firing_rate:
    :param rec_len_secs:
    :return: spike_times (secs) , spike_amplitudes (V)
    """

    # spike times: exponential decay prob
    st = np.cumsum(- np.log(np.random.rand(int(rec_len_secs * firing_rate * 1.5))) / firing_rate)
    st = st[:np.searchsorted(st, rec_len_secs)]

    return st


def test_clusters_metrics():
    pass  # wip on another branch


def test_drift_estimate():
    """
    From spike depths, xcorrelate drift maps to find a drift estimate
    """
    np.random.seed(42)
    ncells = 200
    cells_depth = np.random.random(ncells) * 3800 + 50
    frs = np.random.randn(ncells) * 50 + 200
    t, a, c = multiple_spike_trains(firing_rates=frs, rec_len_secs=200)
    # test negative times, no drift
    drift, ts = electrode_drift.estimate_drift(t - 2, a, cells_depth[c])
    assert(np.all(np.abs(drift) < 0.01))
    # test drift recovery - sinusoid 40 um peak amplitude
    dcor = np.sin(2 * np.pi * t / np.max(t) * 2) * 50
    drift, ts = electrode_drift.estimate_drift(t, a, cells_depth[c] + dcor, display=False)
    drift_ = np.sin(2 * np.pi * ts / np.max(t) * 2) * 50
    # import matplotlib.pyplot as plt
    # plt.plot(ts, drift_)
    # plt.plot(ts, drift)
    assert np.all(np.abs(drift - drift_)[2:] < 4)

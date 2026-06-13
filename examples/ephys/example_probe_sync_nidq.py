"""
Probe synchronisation for NP1-3B and later (nidq workflow)
===========================================================
Synchronise a Neuropixels probe AP recording to the nidq master clock.

For NP1-3B and all subsequent probes the SpikeGLX nidq board echoes a 1 Hz
square wave on nidq digital channel 3 (``imec_sync``).  The same signal is
recorded on each probe AP stream on channel 6.  Matching those two pulse trains
yields a smooth drift-corrected mapping from probe time to session (nidq) time.

Inputs
------
nidq_bin : Path
    Path to the ``.nidq.bin`` file.
probe_bin : Path
    Path to the probe ``.ap.bin`` file.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

import spikeglx
from ibllib.io.extractors.ephys_fpga import _sync_to_alf, get_sync_fronts
from ibllib.ephys.sync_probes import sync_probe_front_times

# ── Inputs ─────────────────────────────────────────────────────────────────
nidq_bin  = Path('/data/subject/2020-01-01/001/raw_ephys_data/my_run.nidq.bin')
probe_bin = Path('/data/subject/2020-01-01/001/raw_ephys_data/probe00/my_run.imec0.ap.bin')

# imec_sync channel indices for 3B (see CHMAPS['3B'] in ibllib.io.extractors.ephys_fpga)
CH_NIDQ_IMEC_SYNC  = 3
CH_PROBE_IMEC_SYNC = 6

# ── Extract sync fronts from both binary files ─────────────────────────────
# Reads .bin in chunks, detects TTL transitions on all digital channels.
nidq_sync  = _sync_to_alf(nidq_bin)
probe_sync = _sync_to_alf(probe_bin)

nidq_fronts  = get_sync_fronts(nidq_sync,  CH_NIDQ_IMEC_SYNC)
probe_fronts = get_sync_fronts(probe_sync, CH_PROBE_IMEC_SYNC)

# ── Compute probe → nidq clock mapping ────────────────────────────────────
sr = spikeglx.Reader(probe_bin).fs  # sampling rate from .meta file

n = min(nidq_fronts.times.size, probe_fronts.times.size)
sync_points, qc = sync_probe_front_times(
    probe_fronts.times[:n],  # slave: probe clock
    nidq_fronts.times[:n],   # reference: nidq clock
    sr=sr,
    type='smooth',           # linear regression + low-pass residual smoothing
)
# sync_points[:, 0]  probe time (s)
# sync_points[:, 1]  nidq  time (s)
print(f'Sync QC passed: {qc}')

# ── Apply sync to spike times ──────────────────────────────────────────────
spike_times_probe = np.load(probe_bin.parent / 'spikes.times.npy')
fcn = interp1d(sync_points[:, 0], sync_points[:, 1], fill_value='extrapolate')
spike_times_nidq = fcn(spike_times_probe)

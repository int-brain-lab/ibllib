'''
Functions that compute spike features from spike waveforms.
'''

import numpy as np
import brainbox as bb


def depth(ephys_file, spks_b, clstrs_b, chnls_b, tmplts_b, unit, n_ch=12, n_ch_probe=385, sr=30000,
          dtype='int16', car=False):
    '''
    Gets `n_ch` channels around a unit's channel of max amplitude, extracts all unit spike
    waveforms from binary datafile for these channels, and for each spike, computes the dot
    products of waveform by unit template for those channels, and computes center-of-mass of these
    dot products to get spike depth estimates.

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    spks_b : bunch
        A spikes bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all spikes.
    clstrs_b : bunch
        A clusters bunch containing fields with cluster information (e.g. amp, ch of max amp, depth
        of ch of max amp, etc.) for all clusters.
    chnls_b : bunch
        A channels bunch containing fields with channel information (e.g. coordinates, indices,
        etc.) for all probe channels.
    tmplts_b : bunch
        A unit templates bunch containing fields with unit template information (e.g. template
        waveforms, etc.) for all unit templates.
    unit : numeric
        The unit for which to return the spikes depths.
    n_ch : int (optional)
        The number of channels to sample around the channel of max amplitude to compute the depths.
    sr : int (optional)
        The sampling rate (hz) that the ephys data was acquired at.
    n_ch_probe : int (optional)
        The number of channels of the recording.
    dtype: str (optional)
        The datatype represented by the bytes in `ephys_file`.
    car: bool (optional)
        A flag to perform common-average-referencing before extracting waveforms.

    Returns
    -------
    d : ndarray
        The estimated spike depths for all spikes in `unit`.

    See Also
    --------
    io.extract_waveforms

    Examples
    --------
    1) Get the spike depths for unit 1.
        >>> import numpy as np
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        (*Note, if there is no 'alf' directory, make 'alf' directory from 'ks2' output directory):
        >>> e_spks.ks2_to_alf(path_to_ks_out, path_to_alf_out)
        # Get the necessary alf objects from an alf directory.
        >>> spks_b = aio.load_object(path_to_alf_out, 'spikes')
        >>> clstrs_b = aio.load_object(path_to_alf_out, 'clusters')
        >>> chnls_b = aio.load_object(path_to_alf_out, 'channels')
        >>> tmplts_b = aio.load_object(path_to_alf_out, 'templates')
        # Compute spike depths.
        >>> unit1_depths = bb.spike_features.depth(path_to_ephys_file, spks_b, clstrs_b, chnls_b,
                                                   tmplts_b, unit=1)
    '''

    # Set constants: #
    n_c_ch = n_ch // 2  # number of close channels to take on either side of max channel

    # Get unit waveforms: #
    # Get unit timestamps.
    unit_spk_indxs = np.where(spks_b['clusters'] == unit)[0]
    ts = spks_b['times'][unit_spk_indxs]
    # Get `n_close_ch` channels around channel of max amplitude.
    max_ch = clstrs_b['channels'][unit]
    if max_ch < n_c_ch:  # take only channels greater than `max_ch`.
        ch = np.arange(max_ch, max_ch + n_ch)
    elif (max_ch + n_c_ch) > n_ch_probe:  # take only channels less than `max_ch`.
        ch = np.arange(max_ch - n_ch, max_ch)
    else:  # take `n_c_ch` around `max_ch`.
        ch = np.arange(max_ch - n_c_ch, max_ch + n_c_ch)
    # Get unit template across `ch` and extract waveforms from `ephys_file`.
    tmplt_wfs = tmplts_b['waveforms']
    unit_tmplt = tmplt_wfs[unit, :, ch].T
    wf_t = tmplt_wfs.shape[1] / (sr / 1000)  # duration (ms) of each waveform
    wf = bb.io.extract_waveforms(ephys_file=ephys_file, ts=ts, ch=ch, t=wf_t, sr=sr,
                                 n_ch_probe=n_ch_probe, dtype='int16', car=car)

    # Compute center-of-mass: #
    ch_depths = chnls_b['localCoordinates'][[ch], [1]]
    d = np.zeros_like(ts)  # depths array
    # Compute normalized dot product of (waveforms,unit_template) across `ch`,
    # and get center-of-mass, `c_o_m`, of these dot products (one dot for each ch)
    for spk in range(len(ts)):
        dot_wf_template = np.sum(wf[spk, :, :] * unit_tmplt, axis=0)
        dot_wf_template += np.abs(np.min(dot_wf_template))
        dot_wf_template /= np.max(dot_wf_template)
        c_o_m = (1 / np.sum(dot_wf_template)) * np.sum(dot_wf_template * ch_depths)
        d[spk] = c_o_m
    return d

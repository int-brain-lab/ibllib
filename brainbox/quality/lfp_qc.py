"""
Quality control on LFP
Based on code by Olivier Winter:
https://github.com/int-brain-lab/ibllib/blob/master/examples/ibllib/ephys_qc_raw.py

adapted by Anne Urai
Instructions https://docs.google.com/document/d/1lBNcssodWdBILBN0PrPWi0te4f6I8H_Bb-7ESxQdv5U/edit#
"""

from pathlib import Path

import sys
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from ibllib.ephys import ephysqc
import one.alf.io as alfio
# from IPython import embed as shell


def _plot_spectra(outpath, typ, savefig=True):
    '''
    TODO document this function
    '''

    spec = alfio.load_object(outpath, 'ephysQcFreq' + typ.upper(), namespace='spikeglx')

    # hack to ensure a single key name
    if 'power.probe_00' in spec.keys():
        spec['power'] = spec.pop('power.probe_00')
        spec['freq'] = spec.pop('freq.probe_00')
    elif 'power.probe_01' in spec.keys():
        spec['power'] = spec.pop('power.probe_01')
        spec['freq'] = spec.pop('freq.probe_01')

    # plot
    sns.set_style("whitegrid")
    plt.figure(figsize=[9, 4.5])
    ax = plt.axes()
    ax.plot(spec['freq'], 20 * np.log10(spec['power'] + 1e-14),
            linewidth=0.5, color=[0.5, 0.5, 0.5])
    ax.plot(spec['freq'], 20 * np.log10(np.median(spec['power'] + 1e-14, axis=1)), label='median')
    ax.set_xlabel(r'Frequency (Hz)')
    ax.set_ylabel(r'dB rel to $V^2.$Hz$^{-1}$')
    if typ == 'ap':
        ax.set_ylim([-275, -125])
    elif typ == 'lf':
        ax.set_ylim([-260, -60])
    ax.legend()
    ax.set_title(outpath)
    if savefig:
        plt.savefig(outpath / (typ + '_spec.png'), dpi=150)
        print('saved figure to %s' % (outpath / (typ + '_spec.png')))


def _plot_rmsmap(outpath, typ, savefig=True):
    '''
    TODO document this function
    '''

    rmsmap = alfio.load_object(outpath, 'ephysQcTime' + typ.upper(), namespace='spikeglx')

    # hack to ensure a single key name
    if 'times.probe_00' in rmsmap.keys():
        rmsmap['times'] = rmsmap.pop('times.probe_00')
        rmsmap['rms'] = rmsmap.pop('rms.probe_00')
    elif 'times.probe_01' in rmsmap.keys():
        rmsmap['times'] = rmsmap.pop('times.probe_01')
        rmsmap['rms'] = rmsmap.pop('rms.probe_01')

    plt.figure(figsize=[12, 4.5])
    axim = plt.axes([0.2, 0.1, 0.7, 0.8])
    axrms = plt.axes([0.05, 0.1, 0.15, 0.8])
    axcb = plt.axes([0.92, 0.1, 0.02, 0.8])

    axrms.plot(np.median(rmsmap['rms'], axis=0)[:-1] * 1e6, np.arange(1, rmsmap['rms'].shape[1]))
    axrms.set_ylim(0, rmsmap['rms'].shape[1])

    im = axim.imshow(20 * np.log10(rmsmap['rms'].T + 1e-15), aspect='auto', origin='lower',
                     extent=[rmsmap['times'][0], rmsmap['times'][-1], 0, rmsmap['rms'].shape[1]])
    axim.set_xlabel(r'Time (s)')
    axrms.set_ylabel(r'Channel Number')
    plt.colorbar(im, cax=axcb)
    if typ == 'ap':
        im.set_clim(-110, -90)
        axrms.set_xlim(100, 0)
    elif typ == 'lf':
        im.set_clim(-100, -60)
        axrms.set_xlim(500, 0)
    axim.set_xlim(0, 4000)
    axim.set_title(outpath)
    if savefig:
        plt.savefig(outpath / (typ + '_rms.png'), dpi=150)


# ============================== ###
# FIND THE RIGHT FILES, RUN AS SCRIPT
# ============================== ###

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please give the folder path as an input argument!")
    else:
        outpath = Path(sys.argv[1])  # grab from command line input
        fbin = glob.glob(os.path.join(outpath, '*.lf.bin'))
        assert(len(fbin) > 0)
        print('fbin: %s' % fbin)
        # make sure you send a path for the time being and not a string
        ephysqc.extract_rmsmap(Path(fbin[0]))
        _plot_spectra(outpath, 'lf')
        _plot_rmsmap(outpath, 'lf')

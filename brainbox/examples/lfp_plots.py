from pathlib import Path
import glob, os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ibllib.ephys import ephysqc
from ibllib.examples.ibllib import _plot_spectra, _plot_rmsmap
import alf.io
import tkinter.filedialog

# find the correct session to work on
fbinpath = tkinter.filedialog.askdirectory()
os.chdir(fbinpath)
for file in glob.glob("*.lf.bin"):
    print(file)

# make sure you send a path for the time being and not a string
ephysqc.extract_rmsmap(os.path.join(fbinpath, file))

_plot_spectra(fbinpath, 'lf')
_plot_rmsmap(fbinpath, 'lf')
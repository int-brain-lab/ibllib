# IBL Python Libraries

[![Build Status on master](https://travis-ci.org/cortex-lab/alyx.svg?branch=master)](https://travis-ci.org/cortex-lab/alyx)
[![Build Status on dev](https://travis-ci.org/cortex-lab/alyx.svg?branch=dev)](https://travis-ci.org/cortex-lab/alyx)

## Description
Libraries used to implement the International Brain Laboratory data pipelines and analyze data. Currently in active development.

This repository has 4 libraries:
-   **ibllib**: general purpose I/O, signal processing and utilities for IBL data pipelines.
-   **oneibl**: interfaces to the Alyx database of experiments to access IBL data.
-   **alf**: implements [ALF](https://docs.internationalbrainlab.org/en/latest/04_reference.html#alf) file naming convention.
-   **brainbox**: analyses for neural and behavioral data.

[Release Notes here](release_notes.md)

## Requirements
**OS**: Deployed on Linux and Windows. Minimally tested for Mac.

**Python Module**: Python 3.6 or higher, we develop on 3.7.

## Documentation
https://ibllib.readthedocs.io/en/latest/

## Installation
To create a unified environment for using ibllib and other [IBL repositories](https://github.com/int-brain-lab/), download and install [Anaconda](https://www.anaconda.com/distribution/#download-section). The instructions below will tell you how to set up and activate the IBL unified conda environment (`iblenv`) and install the 'ibllib', 'iblapps', 'analysis', and 'IBL-pipeline' repositories.

If git and conda are on your system path (to check, try running `git --version` and `conda --version` in your system terminal), then all you have to do is [download the setup files](https://doc-8o-54-drive-data-export.googleusercontent.com/download/1mi4j1i2ck31ntkk0eig31h68c96q59g/muo5a4aejpdkmej507mtkae4impcoj3n/1586389500000/c46ade10-c6e9-4d0a-889d-674878b8f0a4/100803191023609759489/ADt3v-Nwt1maqeg7tuYgAZTj1YOuBa7Xf1DBd4BZMsdTT7UhK4GxJs4hbSXQt7Dajaq8Bvh8SjxkIy3NDqY7QGfjf7ZMaC3wCyctrHeik4Tj5EhlYm9nsELjOZawLhvHkcc2n0pStjabZPF6bxwllItSJS50VK_wKSzBoKlJQsi3dJrQ2UOjoXAm1sktQQnus3FUKePTOgxcq1Xtz6gHzYqp16y19ZFR5eR4fetOFECdzkU0lvm1NvnBSyQrIKVLXWyn-120aaPKCgB29NCmHnyeogG1rBmgPyUTWXP4PdlVpz2hOtl_gD58bYbvidIPUhJGHHMnYK6L23dYtk5iDTgSsHFOgZ6CvA==?authuser=1&nonce=6hlohk28cedn6&user=100803191023609759489&hash=9njs53iqi4bnkvn01jrm0hun3rkhksci), move these files to the directory in which you want to install the IBL repositories, and run the `iblrepos_setup` script within this directory in your system terminal via the command `./iblrepos_setup`. That's it, you're done!

If either git or conda are not on your system path and you want to add them, click [add conda to system path](https://www.google.com/search?q=add+conda+to+system+path) or [add git to system path](https://www.google.com/search?q=add+git+to+system+path) for instructions on how to do so, and then proceed to download the setup files and run the `iblrepos_setup` script as mentioned above.

If either git or conda are not on your system path and you *don't* want to add them, then in your git terminal, navigate to the directory in which you want to install the ibl repositories, and run the following git commands:
```
git clone https://github.com/int-brain-lab/ibllib.git --branch develop
git clone https://github.com/int-brain-lab/iblapps.git --branch develop
git clone https://github.com/int-brain-lab/analysis.git
git clone https://github.com/int-brain-lab/IBL-pipeline.git
```

and in your conda terminal, navigate to this same directory, and run the following conda commands:

```
conda env create -f ./ibllib/iblenv.yaml python=3.8
conda activate iblenv
conda develop ./ibllib
conda develop ./iblapps
conda develop ./analysis
conda develop ./IBL-pipeline
```

*Notes*: 
- Whenever you run IBL code in Python you should activate the `iblenv` environment, and ensure the IBL repository paths are *not* added directly to your python path. 
- If you want to closely follow feature development across different repositories, you can simply checkout and pull the relevant branches within those repositories.
- If launching GUIs that rely on pyqt from ipython, e.g. the IBL data exploration gui or phy, you should first run the ipython magic command `%gui qt`

https://ibllib.readthedocs.io/en/latest/02_installation_python.html#

## Demonstration
https://ibllib.readthedocs.io/en/latest/_static/one_demo.html

## Contribution and development practices
See developper's installation guide here: https://ibllib.readthedocs.io/en/latest/02_installation_dev_python.html

We use gitflow and Semantic Versioning.

Before commiting to your branch:
-   run tests
-   flake8

This is also enforced by continuous integration.

## Matlab Library
The Matlab library has moved to its own repository here: https://github.com/int-brain-lab/ibllib-matlab/

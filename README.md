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

If git and conda are on your system path (to check, try running `git --version` and `conda --version` in your system terminal), then all you have to do is [download the setup files](https://drive.google.com/open?id=1O1q9C-AfmULzEYtLJxxU23p78qfE-MIe), move these files to the directory in which you want to install the IBL repositories, navigate to this directory in your system terminal, and run the command `./iblrepos_setup.sh` if you are using Mac/Linux, or `./iblrepos_setup.ps1` if you are using Windows. That's it, you're done!

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
- While these IBL repositories are under active development, remember to `git pull` regularly.
- If you want to closely follow feature development across different repositories, you can simply checkout and pull the relevant branches within those repositories.
- If launching GUIs that rely on pyqt from ipython, e.g. the IBL data exploration gui or phy, you should first run the ipython magic command `%gui qt`.

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

# IBL Python Libraries

[![Build Status on master](https://travis-ci.org/cortex-lab/alyx.svg?branch=master)](https://travis-ci.org/cortex-lab/alyx)
[![Build Status on dev](https://travis-ci.org/cortex-lab/alyx.svg?branch=dev)](https://travis-ci.org/cortex-lab/alyx)

## Description
Library used to implement the International Brain Laboratory data pipeline. Currently in active development.
The library as currently 4 main modules:
-   **brainbox**: neuroscience analysis oriented library
-   **ibllib**: general purpose library containing I/O, signal processing and IBL data pipelines utilities.
-   **oneibl**: interface to the Alyx database of experiments to access IBL data.
-   **alf**: implementation of ALF file naming convention
[Release Notes here](release_notes.md)

## Requirements
**OS**: Deployed on Linux and Windows. Minimally tested for Mac.

**Python Module**: Python 3.6 or higher, we develop on 3.7.

## Documentation
https://ibllib.readthedocs.io/en/latest/

## Installation
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

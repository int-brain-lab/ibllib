# IBL Python Libraries 
[![Coverage badge](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fibllib.hooks.internationalbrainlab.org%2Fcoverage%2Fibllib%2Fmaster)](https://ibllib.hooks.internationalbrainlab.org/coverage/master) 
[![Tests status badge](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fibllib.hooks.internationalbrainlab.org%2Ftests%2Fibllib%2Fmaster)](https://ibllib.hooks.internationalbrainlab.org/logs/records/master)
[![Tests status badge](https://img.shields.io/endpoint?label=develop&url=https%3A%2F%2Fibllib.hooks.internationalbrainlab.org%2Ftests%2Fibllib%2Fdevelop)](https://ibllib.hooks.internationalbrainlab.org/logs/records/develop)

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

**Python Module**: Python 3.7 or higher, we develop on 3.8.

## Installation, documentation and examples
https://docs.internationalbrainlab.org


## Contribution and development practices
See https://int-brain-lab.github.io/iblenv/07_contribution.html

We use gitflow and Semantic Versioning.

Before commiting to your branch:
-   run tests
-   flake8
This is also enforced by continuous integration.


## Matlab Library
The Matlab library has moved to its own repository here: https://github.com/int-brain-lab/ibllib-matlab/

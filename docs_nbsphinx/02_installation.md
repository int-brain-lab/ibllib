# Installation

There are currently a number of useful libraries being developed within the IBL 
[(https://github.com/int-brain-lab)](https://github.com/int-brain-lab). Those integral to the IBL data architecture 
include,

1. [**ibllib**](https://github.com/int-brain-lab/ibllib)

    The library used to implement the IBL data architecture pipeline. It currently contains four main modules
    * ibllib - general purpose library containing I/O, signal processing and IBL data pipelines utilities.
    * oneibl - interface to the Alyx database of experiments to access IBL data.
    * alf - implementation of ALF file naming convention
    * brainbox - neuroscience analysis oriented library

2.  [**IBL-pipeline**](https://github.com/int-brain-lab/IBL-pipeline)

    The library used to implement the IBL Datajoint pipeline

## Unified Environment
To facilitate the use of `ibllib` and `IBL-pipeline`, we have compiled all the dependencies into a unified python 
environment `iblenv`. In addition to these two libraries, this environment is also compatible with other visualisation 
tools and analysis pipelines being developed as part of the IBL. 

To install this python environment and get started using the IBL data pipeline, please follow 
[these](https://github.com/int-brain-lab/iblenv) installation instructions.
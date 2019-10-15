# ONE light

ONE light is an experimental interface for uploading and downloading data following the [Open Neurophysiology Environment/ALF convention](https://docs.internationalbrainlab.org/en/latest/04_reference.html). The core idea is to represent arbitrarily complex datasets with individual files representing various attributes of some objects. For example, the attributes of the `spikes` object are `times` (time in seconds of each spike), `clusters` (neuron number of each spike), etc. These attributes are represented as numerical arrays saved, by default, in the [NPY format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).


## Installation

ONE light is currently part of the `ibllib` Python library.

You need Python 3 and NumPy. You can install the development version of the code as follows:

```
git clone -b onelight git@github.com:int-brain-lab/ibllib.git
pip install click requests
pip install -e .
```

## For data users


[To download and visualize data with ONE light, see the Jupyter notebook demo.](../examples/oneibl/test_onelight.ipynb)


## For data sharers

It is most convenient to use the command-line interface to upload data with ONE light. Several repository types are currently supported, notably FTP and figshare. We only give the instructions for figshare here.


### Configuration

1. `python onelight.py add_repo` to add a new repository and follow the instructions.
2. When prompted, type `figshare`.
3. Create an article on figshare with no files, copy the URL, and paste it in the command-line tool when prompted.
4. Follow the instructions to enter your figshare token.
5. Your ONE light client is now properly configured.


### Uploading data

1. Prepare your data directory, let's say `/path/to/dataroot/`. The structure must be *exactly* as follows: `/path/to/dataroot/nicklab/Subjects/mymouse/2019-01-01/001/alf/myobj.myattr.npy` etc. The part after the data root directory (`dataroot` in this example) must follow this rigid structure at the moment.
2. `python onelight.py scan /path/to/dataroot/`: check that you see all of your ALF files in your data root directory.
3. `python onelight.py upload /path/to/dataroot/`: upload all files found above to figshare. This may take a while!

### Searching and downloading data

ONE light provides a command-line tool to search, upload, and download data using an HTTP or figshare repository.

1. `python onelight.py search`: display all sessions
2. `python onelight.py search spikes`: filter on sessions that have a `spikes` object.
3. `python onelight.py search spikes.times spikes.clusters`: filter on sessions that have both `spikes.times` and `spikes.clusters` datasets.
4. `python onelight.py list nicklab/Subjects/Cori/2016-12-14/001`: see all dataset types available for that session.
5. `python onelight.py download nicklab/Subjects/Cori/2016-12-14/001 spikes`: download all spikes objects of the specified session. Omit the `spikes` to download the entire session. By default, data is saved locally in `~/.one/data/...`


### Implementation details

When uploading a dataset, ONE light creates and uploads a special `/path/to/dataroot/.one_root` file, a TSV file with 2 columns:

- the first column has each file's relative path (starting with the lab's name)
- the second column has the figshare download URL

The ONE ligh client will use this file for searching and downloading datasets.

# ONE light

ONE light is an interface for uploading and downloading data following the [Open Neurophysiology Environment/ALF convention](https://docs.internationalbrainlab.org/en/latest/04_reference.html). The core idea is to represent arbitrarily complex datasets with individual files representing various attributes of some objects. For example, the attributes of the `spikes` object are `times` (time in seconds of each spike), `clusters` (neuron number of each spike), etc. These attributes are represented as numerical arrays saved, by default, in the [NPY format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).


## Installation

ONE light is currently part of the `ibllib` Python library.

You need Python 3 and NumPy. You can install ibllib as follows:

```
pip install ibllib
```

## For data users


[To download and visualize data with ONE light, see the Jupyter notebook demo.](../examples/oneibl/test_onelight.ipynb)

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
from oneibl.onelight import ONE
one = ONE()
```

We set the current repository to a figshare article, that was specially created with ONE light:

```python
one.set_figshare_url("https://figshare.com/articles/steinmetz/9974357")
```

We search all sessions that have files with a given dataset type. We could pass multiple dataset types. Here, we get all sessions that have spikes:

```python
sessions = one.search(['spikes'])
```

Within a repository, every session is uniquely identified by its full name, which has the following structure: `labname/Subjects/subjectname/date/session`.

```python
sessions
```

    ['nicklab/Subjects/Cori/2016-12-14/001',
     'nicklab/Subjects/Cori/2016-12-17/001',
     'nicklab/Subjects/Cori/2016-12-18/001',
     'nicklab/Subjects/Forssmann/2017-11-01/001',
     'nicklab/Subjects/Forssmann/2017-11-02/001',
     'nicklab/Subjects/Forssmann/2017-11-04/001',
     ...
     'nicklab/Subjects/Tatum/2017-12-08/001',
     'nicklab/Subjects/Tatum/2017-12-09/001',
     'nicklab/Subjects/Theiler/2017-10-11/001']

We take the first session.

```python
session = sessions[0]
```

What are the dataset types contained in this session?

```python
one.list(session)
```

    ['Cori_2016-12-14_M2_g0_t0.imec',
     'Cori_2016-12-14_V1_g0_t0.imec',
     'channels.brainLocation',
     'channels.probe',
     ...
     'sparseNoise.positions',
     'sparseNoise.times',
     'spikes.amps',
     'spikes.clusters',
     'spikes.depths',
     'spikes.times',
     'spontaneous.intervals',
     'trials.feedbackType',
     'trials.feedback_times',
     ...
     'trials.visualStim_contrastRight',
     'trials.visualStim_times',
     'wheel.position',
     'wheel.timestamps',
     'wheelMoves.intervals',
     'wheelMoves.type']

We can load either single files, or full objects.
First, let's load the spike times:

```python
one.load_dataset(session, 'spikes.times')
```
    array([[3.36666667e-03],
           [4.73333333e-03],
           ...,
           [2.70264313e+03],
           [2.70264316e+03]])

Now, we load all `spikes.*` files:

```python
spikes = one.load_object(session, 'spikes')
```

The `spikes` object is an instance of a dictionary, that also allows for the more convenient syntax interface `spikes.times` in addition to `spikes['times']`. Here, we display a raster plot of the first 100,000 spikes:

```python
plt.plot(spikes.times[:100000], spikes.clusters[:100000], ',')
```

## For data sharers

It is most convenient to use the command-line interface when uploading data with ONE light. Several repository types are currently supported, notably FTP and figshare. We only give the instructions for figshare here.


### Configuration

1. `onelight add_repo` to add a new repository and follow the instructions.
2. When prompted, type `myrepo`.
3. When prompted, type `figshare`.
4. When prompted, create a public article on figshare with no files, copy and paste the public URL.
5. Follow the instructions to generate a figshare token on the figshare website, copy and paste it.
6. Type `onelight repo` to check that your repository has been properly configured.


### Uploading data

1. Prepare your data directory, let's say `/path/to/dataroot/`. The structure must be *exactly* as follows: `/path/to/dataroot/nicklab/Subjects/mymouse/2019-01-01/001/alf/myobj.myattr.npy` etc. The part after the data root directory (`dataroot` in this example) must follow this rigid structure at the moment.
2. `onelight scan /path/to/dataroot/`: check that you see all of your ALF files in your data root directory.
3. `onelight upload /path/to/dataroot/`: upload all files found above to figshare. This may take a while!
4. `onelight clean_publish`: make the dataset public.


### Searching and downloading data

ONE light provides a command-line tool to search, upload, and download data using an HTTP or figshare repository.

1. `onelight search`: display all sessions
2. `onelight search spikes`: filter on sessions that have a `spikes` object.
3. `onelight search spikes.times spikes.clusters`: filter on sessions that have both `spikes.times` and `spikes.clusters` datasets.
4. `onelight list nicklab/Subjects/Cori/2016-12-14/001`: see all dataset types available for that session.
5. `onelight download nicklab/Subjects/Cori/2016-12-14/001 spikes`: download all spikes objects of the specified session. Omit the `spikes` to download the entire session. By default, data is saved locally in `~/.one/data/...`


### Implementation details

When uploading a dataset, ONE light creates and uploads a special `/path/to/dataroot/.one_root` file, a TSV file with 2 columns:

- the first column has each file's relative path (starting with the lab's name)
- the second column has the figshare download URL

The ONE light client will use this file for searching and downloading datasets.

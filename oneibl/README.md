# ONE light

## Installation

```
git clone -b onelight git@github.com:int-brain-lab/ibllib.git
cd ibllib/oneibl/
python onelight.py  # this should display the help of the CLI tool
```


## Configuration

1. `python onelight.py add_repo` to add a new repository and follow the instructions.
2. When prompted, type `figshare`
3. figshare article URL: type `https://figshare.com/articles/Dataset_from_Steinmetz_et_al_2019/9598406`
4. [data uploaders only]: follow the instructions to enter your figshare token
5. Your ONE light client is now properly configured.


## [data uploaders only] Uploading data to a figshare article you own

1. Prepare your data directory, let's say `/path/to/dataroot/`. The structure must be *exactly* as follows: `/path/to/dataroot/nicklab/Subjects/mymouse/2019-01-01/001/alf/myobj.myattr.npy` etc. The part after the data root directory (`dataroot` in this example) must follow this rigid structure at the moment.
2. `python onelight.py scan /path/to/dataroot/`: check that you see all of your ALF files in your data root directory.
3. `python onelight.py upload /path/to/dataroot/`: upload all files found above to figshare. This may take a while!
4. `python onelight.py search`: you should see all of your sessions uploaded on figshare. You can do `python onelight.py search clusters probes` to search for sessions that contain both `clusters.*` and `probes.*` dataset types.



## Searching and  downloading data

ONE light provides a command-line tool to search, upload, and download data using an HTTP or figshare repository.

1. `python onelight.py search`: display all sessions
2. `python onelight.py search spikes`: filter on sessions that have a `spikes` object.
3. `python onelight.py search spikes.times spikes.clusters`: filter on sessions that have both `spikes.times` and `spikes.clusters` datasets.
4. `python onelight.py list nicklab/Subjects/Cori/2016-12-14/001`: see all dataset types available for that session.
5. `python onelight.py download nicklab/Subjects/Cori/2016-12-14/001 spikes`: download all spikes objects of the specified session. Omit the `spikes` to download the entire session. By default, data is saved locally in `~/.one/data/...`


## Loading data in Python

Here is how to search, download, and load data in Python.

Open an IPython terminal or a Jupyter notebook and write the following:

```python
import onelight as one
sessions = one.search(['trials'])  # search for all sessions that have a trials object
session = sessions[0]  # take the first session
print(one.list_(session))  # show all available dataset types in that session
trials = one.load_object(session, 'trials')  # load the trials object
print(trials.intervals)  # trials is a Bunch, values are NumPy arrays or pandas DataFrames
print(trials.goCue_times)
```


## Implementation details

When uploading a dataset, ONE light creates and uploads a special `/path/to/dataroot/.one_root` file, a TSV file with 2 columns:

- the first column has each file's relative path (starting with the lab's name)
- the second column has the figshare download URL

Other ONE light clients will use this file for searching and downloading datasets.

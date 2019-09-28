# ONE light

## Installation

```
git clone git@github.com:int-brain-lab/ibllib.git
cd ibllib/oneibl/
python onelight.py
```


## figshare configuration

1. Log in to figshare and go to https://figshare.com/account/applications
2. Click on "Create personal token" at the bottom.
3. Enter "ONE" as token name (for example).
4. Copy the generated token.
5. Edit `~/.one/config.json` and modify the second repository as follows:

```
    {
      "article_id": 9598406,  # this is the number at the end of the figshare article URL, e.g. https://figshare.com/articles/Dataset_from_Steinmetz_et_al_2019/9598406
      "name": "nick19",  # use any short name for this repository
      "token": "",  # copy the token here
      "type": "figshare"
    }
```

6. Set the current repository with `python onelight.py repo nick19`


## Uploading and downloading data

ONE light provides a command-line tool to search, upload, and download data using an HTTP or figshare repository.

1. `python onelight.py search`: see all uploaded ONE sessions on the figshare repo. There shouldn't be anything yet.
2. Prepare your data directory as follows. The structure must be *exactly* as follows: `/path/to/dataroot/nicklab/Subjects/mymouse/2019-01-01/001/alf/myobj.myattr.npy` etc. The part after the data directory root (`dataroot` in this example) must follow this rigid structure at the moment.
3. `python onelight.py scan /path/to/dataroot/`: check that ONE light correctly recognizes your files and sessions on your local computer. You should see all of your ALF files.
4. `python onelight.py upload /path/to/dataroot/`: upload all files found in the previous command to figshare. This may take a while! ONE light also creates and uploads a special `/path/to/dataroot/.one_root` file, a TSV file with 2 columns, the first with each file's relative path (starting with the lab's name), the second with the figshare download URL. Other ONE light clients will use this file for searching and downloading datasets.
5. `python onelight.py search`: this time, you should see all uploaded sessions.
    - `python onelight.py search spikes`: filter on sessions that have a `spikes` object.
    - `python onelight.py search spikes.times spikes.clusters`: filter on sessions that have both `spikes.times` and `spikes.clusters` datasets.
6. `python onelight.py download nicklab/Subjects/Cori/2016-12-14/001 spikes`: download all spikes objects of the specified session. Omit the `spikes` to download the entire session. By default, data is saved locally in `~/.one/data/...`


## Loading data in Python

Here is how to search, download, and load data in Python.

Open an IPython terminal or a Jupyter notebook and write the following:

```python
import onelight as one
sessions = one.search(['trials'])  # search for all sessions that have a trials object
session = sessions[0]  # take the first session
trials = one.load_object(session, 'trials')  # load the trials object
print(trials.intervals)  # trials is a Bunch, values are NumPy arrays or pandas DataFrames
print(trials.goCue_times)
```

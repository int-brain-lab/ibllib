# ONE Tutorial

Before you begin, make sure you have installed ibllib properly on your system as per the instructions.
For this tutorial we will be connecting to a  test database with a test user. The default paremeters from the git repository have the proper configuration.


## Initialize


The first step is to import the ONE class. In the IBL case, the class has to be instantiated: behind the scenes, the constructor connects to the Alyx database and gets credentials. The connections settings are defined in the *params.py* and the *params_secret.py* files.

```python
from oneibl.one import ONE
one = ONE() # need to instantiate the class to have the API.
```
## Info method
Similar to the Alyx database, this library uses sessions UUID as experiments ID.
If the EEID is known, one can get information about a session this way.
```
from ibllib.misc import pprint
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
dlist = one.list(eid)
pprint(dlist)
``` 

For more detailed info, this will return a dataclass with dataset_type, url and dataset_id fields among others:
```python
d = one.session_data_info(eid)
print(d)
```

## Load method
### General Use

Similar to the Alyx database, this library uses sessions UUID as experiments ID.
If the EEID is known, One can access directly the numpy arrays this way:

```python
dataset_types = ['clusters.templateWaveforms', 'clusters.probes', 'clusters.depths']
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
wf, pr, d = one.load(eid, dataset_types=dataset_types)
```

Depending on the use case, it may be handier to wrap the arrays in a dataclass
(a structure for Matlab users) so that a bit of context is included with the array. This would be useful when concatenated information for datasets belonging to several sessions.

```python
my_data = one.load(eid, dataset_types=dataset_types, dclass_output=True)
from ibllib.misc import pprint
pprint(my_data.local_path)
pprint(my_data.dataset_type)
```
```python
[
    "/home/owinter/Downloads/FlatIronCache/clusters.templateWaveforms.2291afac-1d42-4021-a07c-c5539865f42c.npy",
    "/home/owinter/Downloads/FlatIronCache/clusters.probes.66567f54-a5f4-45d1-a9e6-b103ece86339.npy",
    "/home/owinter/Downloads/FlatIronCache/clusters.depths.a26662b5-ff9c-4f15-a8cf-5e9c9e85690f.npy"
]
[
    "clusters.templateWaveforms",
    "clusters.probes",
    "clusters.depths"
]
```
The dataclass contains the following keys, each of which contains a list of 3 items corresponding the the 3 queried datasets

-   data (*numpy.array*): the numpy array
-   dataset_id (*str*): the UUID of the dataset in Alyx
-   local_path (*str*): the local full path of the file
-   dataset_type (*str*): as per Alyx table
-   url (*str*): the link on the FlatIron server
-   eid (*str*): the session UUID in Alyx

It is also possible to query all datasets attached to a given session, in which case
the output has to be a dictionary:
```python
eid, ses_info = one.search(subject='flowers')
my_data = one.load(eid[0])
pprint(my_data.dataset_type)
```

### Specific cases
If a dataset type queried doesn't exist or is not on the FlatIron server, an empty list
is returned. This allows to keep the proper order of output arguments
```python
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
dataset_types = ['clusters.probes', 'thisDataset.IveJustMadeUp', 'clusters.depths']
t, empty, cl = one.load(eid, dataset_types=dataset_types)
```
Returns an empty list for *cr* so that *t* and *cl* still get assigned the proper values.

## ls method
The methods allow to access 3 tables of the current database:
-   dataset-type
-   users
-   subjects

For example to print a list of the dataset-types in the command window:
```python
dtypes, jsondtypes = one.ls_dataset_types()
users, jsonusers = one.ls_users()
subjects, jsonusers = one.ls_subjects()
```
The second argument is a detailed Json list containing the transcript from the database REST query: this will provide table fields from the database.

Also possible to query multiple fields at once:
```python
list_types , dtypes = one.ls(table=['dataset-types','users'])
pprint(list_types)
pprint(dtypes)
```
This will give the following output:
```
[
    [
        "Channel mapping",
        "channels.brainLocation",
        "channels.probe",
        "channels.rawRow",
        "channels.site",
        "channels.sitePositions",
        "clusters.amps",
        "clusters.depths",
        "clusters.meanWaveforms",
        "clusters.peakChannel",
        "clusters._phy_annotation",
        "clusters.probes",
        "clusters.templateWaveforms",
        "clusters.waveformDuration",
        "ephys.raw",
        "ephys.timestamps",
        "eye.area",
        "eye.blink",
        "eye.raw",
        "eye.timestamps",
        "eye.xyPos",
        "_ibl_code.files",
        "_ibl_encoderEvents.bonsai_raw",
        "_ibl_encoderPositions.bonsai_raw",
        "_ibl_encoderTrialInfo.bonsai_raw",
        "_ibl_extraRewards.times",
        "_ibl_lickPiezo.raw",
        "_ibl_lickPiezo.timestamps",
        "_ibl_passiveBeeps.times",
        "_ibl_passiveNoise.intervals",
        "_ibl_passiveTrials.contrastLeft",
        "_ibl_passiveTrials.contrastRight",
        "_ibl_passiveTrials.included",
        "_ibl_passiveTrials.stimOn_times",
        "_ibl_passiveValveClicks.times",
        "_ibl_passiveWhiteNoise.times",
        "_ibl_pycwBasic.data",
        "_ibl_pycwBasic.settings",
        "_ibl_sparseNoise.times",
        "_ibl_sparseNoise.xyPos",
        "_ibl_trials.choice",
        "_ibl_trials.contrastLeft",
        "_ibl_trials.contrastRight",
        "_ibl_trials.feedback_times",
        "_ibl_trials.feedbackType",
        "_ibl_trials.goCue_times",
        "_ibl_trials.included",
        "_ibl_trials.intervals",
        "_ibl_trials.repNum",
        "_ibl_trials.response_times",
        "_ibl_trials.stimOn_times",
        "_ibl_wheelMoves.intervals",
        "_ibl_wheelMoves.type",
        "_ibl_wheel.position",
        "lfp.raw",
        "lfp.timestamps",
        "licks.times",
        "probes.description",
        "probes.insertion",
        "probes.rawFilename",
        "probes.sitePositions",
        "raw_behavior_data",
        "raw_ephys_data",
        "raw_imaging_data",
        "raw_video_data",
        "spikes.amps",
        "spikes.clusters",
        "spikes.depths",
        "spikes.times",
        "spontaneous.intervals",
        "unknown"
    ],
    [
        "jmontijn",
        "nbonacchi",
        "olivier",
        "test_user"
    ]
]
[
    [
        {
            "name": "Channel mapping",
            "created_by": "olivier",
            "description": "NOTE \"channel\" refers ONLY to those channels that made it through to spike sorting (probably not all of them). \"rawRow\" means a row in the raw recording file, of which there may be more",
            "filename_pattern": "Channel mapping.*"
        },
    # and so on...
```


## Search method
The search methods allows to query the database to filter the list of UUIDs according to
the following fields:
-   dataset_types
-   users
-   subject
-   date_range

### One-to-one matches: subjects
This is the simplest case that queries EEIDs (sessions) associated with a subject. There can only
be one subject per session.

```python
from oneibl.one import ONE
myone = ONE() # need to instantiate the class to have the API.
eid, ses_info = one.search(subject='flowers')
pprint(eid)
pprint(ses_info)

```

Here is the simple implementation of the filter, where we query for the EEIDs (sessions) co-owned by
all of the following users: olivier and niccolo (case-sensitive).
```python
eid, ses_info = one.search(users=['nbonacchi', 'olivier'])
```

The following would get all of the dataset for which olivier is an owner or a co-owner:
```python
eid , ses_info=  one.search(users=['olivier'])
pprint(eid)
```

It is also possible to filter sessions using a date-range:
```python
eid, ses_info = one.search(users='olivier', date_range=['2018-08-24', '2018-08-24'])
```


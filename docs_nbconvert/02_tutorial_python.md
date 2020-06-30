# ONE Tutorial Python

Before you begin, make sure you have installed ibllib properly on your system as per the previous instructions.
[Here](./_static/one_demo.html) is a shorter introduction to the One module in Ipython notebook format.

The full module documentation is found [Here](./_autosummary/oneibl.one.html)

## Create ONE object

Once the One class is instantiated and setup, we can create an **one object** (here labelled `one`). 

### With default connection settings

Type in python prompt: 

```python
from oneibl.one import ONE
one = ONE() # need to instantiate the class to have the API. You will have to write these lines of code everytime you re-open python.
```

Reminder: connection parameters inserted via `one.setup()` will modify the JSON *.one_params* file [see here](02a_installation_python.html)


### With different connection settings for single time use

For this tutorial we will be connecting to a **test database** with a **test user**. As these credentials will be used for this tutorial only, we do not want to change the base parameters of the JSON *.one_params* file.

```python
from oneibl.one import ONE
one = ONE(username='test_user',
          password='TapetesBloc18',
          base_url='https://test.alyx.internationalbrainlab.org')
```


## Search method : find session EIDs
Each experimental session is identified by a unique string known as the "experiment ID" (EID). To find an EID, use the `one.search` command.

The searchable fields are listed with the following method:
```python
one.search_terms()

# Example search keywords: 
['dataset_types',
 'date_range',
 'lab',
 'location',
 'number',
 'performance_gte',
 'performance_lte',
 'subject',
 'task_protocol',
 'users']

```

### Comprehensive example

The following example shows how to find the experiment(s) performed by the user `olivier`, on the 24 Aug 2018:

```python
from ibllib.misc import pprint
eid = one.search(users='olivier', date_range=['2018-08-24', '2018-08-24'])
pprint(eid)
```
returns
```python
[
    "cf264653-2deb-44cb-aa84-89b82507028a"
]
```
To get all context information about the returned sessions, use the flag details:
```python
eid, ses_info= one.search(users='olivier', date_range=['2018-08-24', '2018-08-24'], details=True)
```



## List method : list datasets for given session EID
Once you know the EID, you can list all the datasets for the experiment using the list command:
```python
one.list(eid)
``` 
returns
```python
['_ibl_lickPiezo.raw', '_ibl_lickPiezo.timestamps', '_ibl_wheel.position', 'channels.brainLocation', 'channels.probe', 'channels.rawRow', 'channels.site', 'channels.sitePositions', 'clusters._phy_annotation', 'clusters.depths', 'clusters.peakChannel', 'clusters.probes', 'clusters.templateWaveforms', 'clusters.waveformDuration', 'eye.area', 'eye.blink', 'eye.timestamps', 'eye.xyPos', 'licks.times', 'probes.description', 'probes.insertion', 'probes.rawFilename', 'probes.sitePositions', 'spikes.amps', 'spikes.clusters', 'spikes.depths', 'spikes.times', 'spontaneous.intervals', 'unknown']
```

For more detailed datasets info, this will return a dataclass with `dataset_type`, `data_url` and `dataset_id` fields amongst others:
```python
d = one.list(eid, details=True)
print(d)
```

To get the full contextual information about the session, get all fields:
```python
one.list(eid, 'all')
```

To navigate the database, it may be useful to get the range of possible keywords values to search for sessions.
For example to print a list of the dataset-types, users and subjects in the command window:
```python
one.list(None, 'dataset-types')
one.list(None, 'users')
one.list(None, 'subjects')
one.list(None, 'labs')
```


## Load method : load data for given session EID
### General Use

To load data for a given EID, use the `one.load` command:

```python
dataset_types = ['clusters.templateWaveforms', 'clusters.probes', 'clusters.depths']
eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
wf, pr, d = one.load(eid, dataset_types=dataset_types)
```

Depending on the use case, it may be handier to wrap the arrays in a dataclass
(a structure for Matlab users) so that a bit of context is included with the array. This would be useful when concatenated information for datasets belonging to several sessions.

```python
my_data = one.load(eid, dataset_types=dataset_types, dclass_output=True)
from ibllib.misc import pprint
print(my_data.local_path)
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
eid = one.search(subjects='flowers')
my_data = one.load(eid[0])
pprint(my_data.dataset_type)
```

### Specific cases
If a dataset type queried doesn't exist or is not on the FlatIron server, an empty list
is returned. This allows to keep the proper order of output arguments
```python
eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
dataset_types = ['clusters.probes', 'thisDataset.IveJustMadeUp', 'clusters.depths']
t, empty, cl = one.load(eid, dataset_types=dataset_types)
```
Returns an empty list for *cr* so that *t* and *cl* still get assigned the corresponding datasets values.

## Getting help on methods via the command line
These commands will return the docstring of each of the method; for example, for `one.search` type in an ipython terminal:
```
help(one.search)
```
or
```
print(one.search.__doc__)
```

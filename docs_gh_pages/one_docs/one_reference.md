# ONE Reference
## Open Neurophysiology Environment

Neurophysiology needs data standardization. A scientist should be able to analyze data collected from multiple labs using a single analysis program, without spending untold hours figuring out new file formats. Substantial efforts have recently been put into developing neurodata file standards, with the most successful being the [Neurodata Without Borders](https://www.nwb.org/) (NWB) format. The NWB format has a comprehensive and careful design that allows one to store all data and metadata pertaining to a neurophysiology experiment. Nevertheless, its comprehensive design also means a steep learning curve for potential users, which has limited its adoption.

### How the open neurophysiology environment works

Here we provide a solution to this problem: a set of four simple functions, that allow users to access and search data from multiple sources. To adopt the standard, data providers can use any format they like - all they need to do is implement these functions to fetch the data from their server and load it into Python or MATLAB. Users can then analyze this data with the same exact code as data from any other provider. The learning curve will be simple, and the loader functions would be enough for around 90% of common use cases.

By a *data provider* we mean an organization that hosts a set of neurophysiology data on an internet server (for example, the [International Brain Lab](https://www.internationalbrainlab.com/)). The Open Neurophysiology Environment (ONE) provides a way for scientists to analyze data from multiple data providers using the same analysis code. There is no need for the scientist to explicitly download data files or understand their format - this is all handled seamlessly by the ONE framework. The ONE protocol can also be used to access a scientist's own experiments stored on their personal computer, but we do not describe this use-case here.

When a user wants to analyze data released by a provider, they first import that provider's loader functions. In python, to analyze IBL data, they would type
```
from oneibl.one import ONE
```
Because it is up to data providers to maintain the loader functions, all a user needs to do to work with data from a specific provider is import their loader module. To analyze Allen data, they could instead type `import one_allen`. After that, all other analysis code will be the same, regardless of which provider's data they are analyzing.

Every experiment a data provider releases is identified by an *experiment ID* (eID) -- a small token that uniquely identifies a particular experiment. It is up to the data provider to specify the format of their eIDs. 

#### Loading data

If a user already knows the eID of an experiment they are interested in, they can load data for the experiment using a command like:
```
st, sc, cbl = ONE().load(eID, dataset_types=['spikes.times', 'spikes.clusters', 'clusters.brain_location'])
```
This command will download three datasets containing the times and cluster assignments of all spikes recorded in that experiment, together with an estimate of the brain location of each cluster. (In practice, the data will be cached on the user's local machine so it can be loaded repeatedly with only one download.)

Many neural data signals are time series, and synchronizing these signals is often challenging. ONE will provide a function to interpolate any required timeseries to an evenly or unevenly-sampled timescale of the user's choice. For example the command:
```
hxy, hth, t = ONE().loadTS(eID, dataset_types=['headTracking.xyPos', 'lfp.raw'], sample_rate=1000)
```
would interpolate head position and lfp to a common 1000 Hz sampling rate. The sample times are returned as `t`.

#### Searching for experiments
Finally, a user needs to be able to search the data released by a provider, to obtain the eIDs of experiments they want to analyze. To do so they would run a command like:
```
eIDs = ONE().search(lab='CortexLabUCL', subject='hercules', dataset_types=['spikes.times', 'spikes.clusters','headTracking.xyPos'])
eIDs, eInfo = ONE().search(details=True, lab='CortexLabUCL', subject='hercules', dataset_types=['spikes.times', 'spikes.clusters','headTracking.xyPos'])
```
This would find the eIDs for all experiments collected in the specified lab for the specified experimental subject, for which all of the required data is present. There will be more metadata options to refine the search (e.g. dates, genotypes, experimenter), and additional metadata on each matching experiment is returned in `eInfo`. However, the existence of dataset types is normally enough to find the data you want. For example, if you want to analyze electrophysiology in a particular behavior task, the experiments you want are exactly those with datasets describing the ephys recordings and that task's parameters.

### Standardization

The key to ONE's standardization is the concept of a "standard dataset type". When a user requests one of these (such as `'spikes.times'`), they are guaranteed that each data provider will return them the same information, organized in the same way - in this case, the times of all extracellularly recorded spikes, measured in seconds relative to experiment start, and returned as a 1-dimensional column vector. It is guaranteed that any dataset types of the form `*.times` or `*.*_times` will be measured in seconds relative to experiment start. Furthermore, all dataset types differing only in their last word (e.g. `spikes.times` and `spikes.clusters`) will have the same number of rows, describing multiple attributes of the same objects. Finally, words matching across dataset types encode references: for example, `spikes.clusters` is guaranteed to contain integers, and to find the brain location of each of these one looks to the corresponding row of `clusters.brain_location`, counting from 0.

Not all data can be standardized, since each project will do unique experiments. Data providers can thereform add their own project-specific dataset types. The list of standard dataset types will be maintained centrally, and will start small but increase over time as the community converges on good ways to standardize more information. It is therefore important to distinguish dataset types agreed as universal standards from types specific to individual projects. To achieve this, names beginning with an underscore are guaranteed never to be standard. It is recommended that nonstandard names identify the group that produces them: for example the dataset types `_ibl_trials.stimulusContrast` and `clusters._ibl_task_modulation` could contain information specific to the IBL project.

### Versioning and subcollections

Data are often released in multiple versions. Most users will want always to have the latest version, but sometimes a user working will want to continue working with a historical version even after it has been updated, to maintain consistency with previous work. To enable versioning, the ONE functions will have accept an optional argument of the form `version='v1'`, which will ensure this specific version is loaded. If the argument is not passed, the latest version will be loaded. 

Sometimes the data will contain multiple measurements of the same type, for example if recordings are made with multiple recording probes simultaneously. In these cases, the dataset types for probe 0 will have names like `probe00/spikes.times`, `probe00/spikes.clusters`, and `probe00/clusters.brain_location`; data for probe 1 will be `probe01/spikes.times`, etc. Encoding of references works within a subcollection: i.e. the entries of `probe00/spikes.clusters` point to the rows of `probe00/clusters.brain_location`, starting from 0, and independently of any datasets starting with `probe01/`.

### For data sharers

Data standards are only adopted when they are easy to use, for both providers and users of data. For users, the three ONE functions will be simple to learn, and will cover most common use cases.

For providers, a key advantage of this framework is its low barrier to entry. To share data with ONE, providers do not need to run and maintain a backend server, just to upload their data to a website. We provide a "ONE light" implementation of the ONE loader functions that searches, downloads and caches files from a web server. This will allow producers who do not have in-house computational staff two simple paths to achieve ONE compatibility. The first is to place the data in a directory, using a standard file-naming convention described below, in standard formats including `npy`, `csv`, `json`, and `tiff`. Next, the user runs a program in this directory, which uploads the files to a website or to figshare. Users can then access this data using ONE light. And example of ONE light data is [here](https://figshare.com/articles/Test1/9917741). The ONE light code is [here](https://github.com/int-brain-lab/ibllib/tree/onelight/oneibl#one-light).


## ALF

ALF stands for "ALyx Files". It not a format but rather a format-neutral file-naming convention. ALF is how IBL organizes files that will be loaded via the ONE protocol.

In ALF, the measurements in an experiment are represented by collections of files in a directory. Each filename has three parts, for example `spikes.times.npy` or `spikes.clusters.npy`. The first two correspond to the ONE dataset type, and we refer to them as the _object_ and the _attribute_. The third part of the filename, the _extension_, specifies what physical format the file is in - we primarily use .npy and .tsv but you could use any format, for example video or json .

Each file contains information about particular attribute of the object. For example `spikes.times.npy` indicates the times of spikes and `spikes.clusters.npy` indicates their cluster assignments. You could have another file `spikes.amplitudes.npy` to convey their amplitudes. The important thing is that every file describing an object has the same number of rows (i.e. the 1st dimension of an npy file, number of frames in a video file, etc).  You can therefore think of the files for an object as together defining a table, with column headings given by the attribute in the file names, and values given by the file contents.

ALF objects can represent anything. But three types of object are special:

### Event series

If there is a file with attribute `times`, (i.e. filename `obj.times.ext`), it indicates that this object is an event series. The `times` file contains a numerical array containing times of the events in seconds, relative to a universal timescale common to all files. Other attributes of the events are stored in different files. If you want to represent times relative to another timescale, do this by appending to `timescale` after an underscore (e.g. `spikes.times_ephysClock.npy`). By convention, any other file with attribute that ends in `_times` is understood to be a time in universal seconds; for example `trials.reward_times.npy`. An attribute ending with `_times_timescale` is by convention a time in that timescale.

### Interval series

If there is a file with attribute `intervals`, (i.e. filename `tones.intervals.npy`), it should have two columns, indicating the start and end times of each interval relative to the universal timescale. Again, other attributes of the events can be stored in different files (e.g. `tones.frequencies.npy`. Event times relative to other timescales can be represented by a file with attribute `intervals_timescale`. Again, any other attributes of the form `trials.cue_intervals.npy` are by convention measured in universal seconds.

### Continuous timeseries

If there is a file with attribute `timestamps`, it indicates the object is a continuous timeseries. The timestamps file represents information required to synchronize the timeseries to the universal timebase, even if they were unevenly sampled. There are 2 possibilities:
-	The `timestamps` file contains a single column containing time of every sample within the timescale.
-	The `timestamps` file contains two rows. Each row of the `timestamps` file represents a synchronization point, with the first column giving the sample number (counting from 0), and the second column giving the corresponding time in universal seconds. The times corresponding to all samples are then found by linear interpolation. Note that the `timestamps` file is an exception to the rule that all files representing a continuous timeseries object must have one row per sample, as it will often have substantially less. Note that an evenly-sampled recording should have just two timestamps, giving the times of the first and last sample.  


### File types
ALF can deal with any sort of file, as long as it has a concept of a number of rows (or primary dimension). The type of file is recognized by its extension. Preferred choices:

.npy: numpy array file. This is recommended over flat binary since datatype and shape is stored in the file. If you want to name the columns, use a metadata file. If you have an array of 3 or more dimensions, the first dimension counts as the number of rows. To access npy files from MATLAB use [this](https://github.com/kwikteam/npy-matlab).

.tsv: tab-delimited text file. This is recommended over comma-separated files since text fields often have commas in. All rows should have the same number of columns. The first row contains tab-separated names for each column.

.bin: flat binary file. It's better to use .npy for storing binary dat,a but some recording systems save in flat binary. Rather than convert them, you can ALFize a flat binary file by adding a metadata file, which specifies the number of columns (as the size of the "columns" array) and the binary datatype as a top-level key "dtype", using numpy naming conventions.

### Relations

Encoding of relations between objects can be achieved by a simplified relational model. If the attribute name of one file matches the object name of a second, then the first file is guaranteed to contain integers referring to the rows of the second. For example, `spikes.clusters.npy` would contain integer references to the rows of `clusters.brain_location.json` and `clusters.probes.npy`; and `clusters.probes.npy` would contain integer references to `probes.insertion.json`. Be careful of plurals (`clusters.probe.npy` would not correspond to `probes.insertion.json`) and remember we count arrays starting from 0.

### Longer file names
A proposed extension to ALF would allow encoding of additional information in filenames with more than 3 parts. In this proposal, file names could have as many parts as you like: object.attribute.x1.x2. ... .xN.extension. The extra name parts play no formal role in the ALF conventions, but can serve several additional purposes. For example, if you want unique file names to make archiving and backups easier, they could contain a unique string, for example a Dataset ID from Alyx, or the md5 hash of the file. Extra parts could be used  to encode the subject name if you are worried about accidentally moving files between directories. The filenames might get long; however the important information in the filename is in the first two parts, which users can tab-autocomplete when typing them at the command line; also, because the extension is last, they can also double-click the file to open it with a default application such as a movie viewer.

Finally, if there are multiple files with the same object, attribute, and extensions but different extra parts, these should be treated as files to be concatenated, for example to allow multiple-part tif files as produced by scanimage to be encoded in ALF. The concatenation would happen in hierarchical lexicographical order: i.e. by lexicographic order of x1, then x2 if x1 is equal, etc.

### Metadata
Sometimes you will want to provide metadata on the columns or rows of a data file. For example, clusters.ccf_location.tsv could be a 4-column tab-delimited text file in which the first 3 columns contain xyz coordinates of a cluster and the 4th contains its inferred brain location as text. In this case, an additional JSON file clusters.ccf_location.metadata.json can provide information about the columns and rows. The metadata file can contain anything, but if it has a top-level key "columns", that should be an array of size the number of columns, and if it has a top-level key "rows" that should be an array of size the number of rows. If the entries in the columns and rows arrays have a key "name" that defines a name for the column or row; a key "unit" defines a unit of measurement. You can add anything else you want.

Note that in ALF you should not have generally two data files with the same object and attribute: if you have tones.frequencies.npy, you can't also  have tones.frequencies.tsv. Metadata files are an exception to this: if you have tones.frequencies.npy, you can also have tones.frequencies.metadata.json.

### Versioning and subcollections
Versioning and subcollections are achieved with subdirectories. If your main ALF directory contains a subfolder `v1`, any files in there are assumed to be version `v1`. The root directory is assumed to contain the most recent files. 

Subcollections are also achieved with subdirectories: a subfolder `probe00` of the main ALF directory contains the files specific to probe 0; you can also have a subcollection within a version such as `v1/probe00`.

# Open Neurophysiology Environment

## Neurodata standardization needs a new approach
Neurophysiology badly needs data standardization. A scientist should be able to analyze data collected from multiple labs using a single analysis program, without spending untold hours figuring out new file formats.

Substantial efforts have recently been put into developing neurodata file standards, with the most successful being the [Neurodata Without Borders](https://www.nwb.org/) (NWB) format. The NWB format has a comprehensive and careful design that allows one to store all data and metadata pertaining to a neurophysiology experiment. Nevertheless, its comprehensive design also means a steep learning curve for potential users, which has limited its adoption.

The same thing has happened before in other scientific fields. For example, the open microscopy environment ([OME](https://www.openmicroscopy.org/)) group developed a standard file format for microscopy data, which is hardly ever used. But they also developed a set of loader functions, which allow scientists to analyze data in multiple native formats using a single program. These loader functions successfully standardized microscopy data. In principle, NWB also allows data providers to reimplement a set of API functions -- however the comprehensive nature of NWB again means this API is complex, limiting its adoption.

## How the open neurophysiology environment would work

Here we propose a solution to this problem: a set of three simple loader functions, that would allow users to access and search data from multiple sources. To adopt the standard, data providers can use any format they like - all they need to do is implement these three functions to fetch the data from their server and load it into Python or MATLAB. Users can then analyze this data with the same exact code as data from any other provider. The learning curve will be simple, and the loader functions would be enough for around 90% of common use cases.

By a *data provider* we mean an organization that hosts a set of neurophysiology data on an internet server (for example, the [International Brain Lab](https://www.internationalbrainlab.com/)). The Open Neurophysiology Environment (ONE) provides a way for scientists to analyze data from multiple data providers using the same analysis code. There is no need for the scientist to explicitly download data files or understand their format - this is all handled seamlessly by the ONE framework. The ONE protocol can also be used to access a scientist's own experiments stored on their personal computer, but we do not describe this use-case here.

When a user wants to analyze data released by a provider, they first import that provider's loader functions. In python, to analyze IBL data, they would type
```
from one_ibl.one import ONE
```
Because it is up to data providers to maintain the loader functions, all a user needs to do to work with data from a specific provider is import their loader module. To analyze Allen data, they could instead type `import one_allen`. After that, all other analysis code will be the same, regardless of which provider's data they are analyzing.

Every experiment a data provider releases is identified by an *experiment ID* (eID) -- a small token that uniquely identifies a particular experiment. It is up to the data provider to specify the format of their eIDs. 

### Loading data

If a user already knows the eID of an experiment they are interested in, they can load data for the experiment using a command like:
```
st, sc, cbl = ONE.load(eID, dataset_types=['spikes.times', 'spikes.clusters', 'clusters.brain_location'])
```
This command will download three datasets containing the times and cluster assignments of all spikes recorded in that experiment, together with an estimate of the brain location of each cluster. (In practice, the data will be cached on the user's local machine so it can be loaded repeatedly with only one download.)

Many neural data signals are time series, and synchronizing these signals is often challenging. ONE would provide a function to interpolate any required timeseries to an evenly or unevenly-sampled timescale of the user's choice. For example the command:
```
hxy, hth, t = ONE.loadTS(eID, dataset_types=['headTracking.xyPos', 'lfp.raw'], sample_rate=1000)
```
would interpolate head position and lfp to a common 1000 Hz sampling rate. The sample times are returned as `t`.

### Searching for experiments
Finally, a user needs to be able to search the data released by a provider, to obtain the eIDs of experiments they want to analyze. To do so they would run a command like:
```
eIDs, eInfo = ONE.search(lab='CortexLabUCL', subject='hercules', dataset_types=['spikes.times', 'spikes.clusters','headTracking.xyPos'])
```
This would find the eIDs for all experiments collected in the specified lab for the specified experimental subject, for which all of the required data is present. There will be more metadata options to refine the search (e.g. dates, genotypes, experimenter), and additional metadata on each matching experiment is returned in `eInfo`. However, the existence of dataset types is normally enough to find the data you want. For example, if you want to analyze electrophysiology in a particular behavior task, the experiments you want are exactly those with datasets describing the ephys recordings and that task's parameters.

## Standardization

The key to ONE's standardization is the concept of a "standard dataset type". When a user requests one of these (such as `'spikes.times'`), they are guaranteed that each data provider will return them the same information, organized in the same way - in this case, the times of all extracellularly recorded spikes, measured in seconds relative to experiment start, and returned as a 1-dimensional column vector. It is guaranteed that any dataset types of the form `*.times` or `*.*_times` will be measured in seconds relative to experiment start. Furthermore, all dataset types differing only in their last word (e.g. `spikes.times` and `spikes.clusters`) will have the same number of rows, describing multiple attributes of the same objects. Finally, words matching across dataset types encode references: for example, `spikes.clusters` is guaranteed to contain integers, and to find the brain location of each of these one looks to the corresponding row of `clusters.brain_location`, counting from 0.

Not all data can be standardized, since each project will do unique experiments. Data providers can thereform add their own project-specific dataset types. The list of standard dataset types will be maintained centrally, and will start small but increase over time as the community converges on good ways to standardize more information. It is therefore important to distinguish dataset types agreed as universal standards from types specific to individual projects. To achieve this, names beginning with an underscore are guaranteed never to be standard. It is recommended that nonstandard names identify the group that produces them: for example the dataset types `_ibl_trials.stimulusContrast` and `clusters._ibl_task_modulation` could contain information specific to the IBL project.

## Ease of use

Data standards are only adopted when they are easy to use, for both providers and users of data. For users, the three ONE functions will be simple to learn, and will cover most common use cases.

For providers, a key advantage of this framework is its low barrier to entry. To share data with ONE, providers will not need to run and maintain a backend server, just to upload their data to a website. We will provide a reference implementation of the ONE loader functions that searches, downloads and caches files from a web server. This will allow producers who do not have in-house computational staff two simple paths to achieve ONE compatibility. The first is to upload data to a website using a standard file-naming convention [[here](https://github.com/cortex-lab/ALF)], in standard formats including `npy`, `csv`, `json`, and `tiff`, which will allow users to read it using the reference implementation. The second is to post data on a web site using their own file-naming conventions and formats, then clone and adapt our reference implementation to their specific formats.

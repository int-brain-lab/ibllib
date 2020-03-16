### Release Notes 1.4.13 - 2020-02-07
-   Hotfix: rig transfer - create probes. One variable used before assignation. 

### Release Notes 1.4.12 - 2020-02-06
-   ONE.load overwrites local file if filesizes different or hash mismatch
-   ephys extraction:
    -   registration sets the session.procedure field to acute recording
    -   bugfix synchronization on re-extraction: always recompute spike.times from spike.samples
    -   ephys registration sets the session.procedure field to acute recording
-   training extraction:
    -   added biasedVisOffChoiceWorld as training extractor
-   wheel data
    -   dropping support for wheel velocity, not extracted anymore

### Release Notes 1.4.11 - 2020-02-14
-   bugfix: Include sessions data files for ephys mock
-   bugfix: Single probe 3B gets synchronized

### Release Notes 1.4.10 - 2020-02-10
-   bugfix: Include sessions data files in pip package 

### Release Notes 1.4.9 - 2020-02-09
-   Big brainbox merge and release
-   bugfix: clusters.metrics spiking rates accurate 
-   probability left for ephys choice world contain generative probabilities, not outcomes

### Release Notes 1.4.8 - 2020-01-27
-   ONE Light Windows fixes
-   Installation documentation separates conda and virtualenv options
-   Conda yaml environement file for ibllib

### Release Notes 1.4.7 
-   ONE Light for behaviour paper data release   
-   ONE() standard syntax matching the one light examples

### Release Notes 1.4.6 
-   Alyx registration: add md5, version and filesize to the pipeline registration
-   Data Patcher: allows to register data from anywhere through FTP/SSH/GLobus
-   ONE Light for behaviour paper data release 

### Release Notes 1.4.5 Hotfix
-   Ephys extraction: left probability bug when sequence was 0 - fixed

### Release Notes 1.4.4
-   Ephys extraction:
    -   left probability extracted properly
    -   add robustness to audio fronts extraction in FPGA
-   Passive stimulus: raw data registered in pipeline
-   Training extraction: microphone extraction for habituation sessions
-   ALF: specific to_dataframe method for Bunch
    
### Release Notes 1.4.3 Hotfix
-   Ephys extraction: handle fringe case where recording is interrupted in the middle
-   Wheel extraction: if rotary encoder version is outdated and stores data in the wrong unit, auto-detect and output in seconds even for new versions

### Release Notes 1.4.2
-  FPGA/bpod events synchronization performed even when their counts do not match
-  Updated requirement versions for mtscomp and phylib

### Release Notes 1.4.1
-   wheel extraction outputs a timestamps attribute, not times
-   make the wheel extraction more robust

### Release Notes 1.4.0
-   Ephys extraction:
    -   un-synchronized spike sortings not uploaded on flat-iron
    -   reaction times extracted
    -   valve-open times bugfix
-   Wheel extraction:
    -   training wheel position and timing are now correct
    -   ephys & training: units: radians mathematical convention
-   ONE:
    -   Alyx client handles pagination seamlessly

### Release Notes 1.3.10 Hotfix
-   cross-platform get of session folder for rig computer copy to server

### Release Notes 1.3.9
-   spikeglx.verify_hash() method to check file integrity after transfers/manipulations
-   create wirings settings files on ephys computer transfer to server

### Release Notes 1.3.8
-   Ephys Extraction:
    -   duplicate probe.trajectories bugfix
    -   extraction works with unoperational fram2ttl at beginning of ephys session
    -   clusters.metrics.csv has consistent size with npy cluster objects
    -   ephys transfer: create ephys extraction flags after the transfer is complete

### Release Notes 1.3.7 hotfixes- 21-NOV-2019
-   Rename spike.times on failed sync to reflect the clock as per ALF convention
-   sync 3A fails if first cam event whithin 200ms of start
-   compress ephys goes through a tempfile to not interfere with transfers/globbing

### Release Notes 1.3.6 - 20-NOV-2019
-   Ephys Extraction (phylib)
    -   convert ks2 amplitudes to volts for spikes.amps, clusters.amps. templates.waveforms, clusters.waveforms to get uV
    -   generates Cluster UUIDs file
    -   individual spike depths computed from PC features  
-   Ephys Synchronization
    -   use frame2TTL split for 3A by default, if not found look for right_camera
    -   output individual probe sync in ALF timestamps format

### Release Notes 1.3.5 - 18-NOV-2019
-   registration ignores ks2alf probes subfolders
-   fix typo in raw qc dataset types

### Release Notes 1.3.4 - 18-NOV-2019
-   Alyx registration adds the relative path to session root as dataset.subcollection
-   Ephys extraction:
    -   split probe folders output alf/probe00 and alf/probe01 instead of merge
    -   outputs templates.waveforms and clusters.waveforms in sparse arrays
    -   outputs probes.description and probes.trajectory
    -   renamed the raw ephys QC output
    -   outputs clusters.metrics
-   Bugfixes:
    -   3B raw ephys QC output ap.file not found on nidq object
    -   3A sync probe threshold set to 2.1 samples

### Release Notes 1.3.1 - 12-NOV-2019
-   transfer scripts from ephys/video/rig computers to local servers
-   bugfix spigeglx.glob_ephys_files when metadata file without ap.bin file

### Release Notes 1.3.0 - 11-NOV-2019
-   Transfer rig data takes into account session type (ephys/training) to create flags
-   Ephys video compression in pipeline
-   Ephys audio compression in pipeline

### Release Notes 1.2.9
-   Ephys extraction: provide full 3B default wirings if files do not exist.

### Release Notes 1.2.8
-   Ephys extraction: merge sync ephys in the pipeline overwrites ks2_alf directory if it already exists

### Release Notes 1.2.7
-   Added `biasedScanningChoiceWorld` task to biased extractor for Zador lab

### Release Notes 1.2.6
#### feature/ephys_compression
- `spikeglx.Reader` supports mtscomp ephys binaries
- server pipeline for compression of ephys files
#### feature/peths
- `brainbox.singlecell.peths` with tests
- `brainbox.processing.bincount2D` supports aggregation on fixed scale
- simple examples script and notebook

### Release Notes 1.2.5
- examples/brainbox/plot_peths.py: by Matt. W.
- examples/brainbox/rasters by Michaël S.
- Allen Atlas framework and probe registration base functions
- server pipeline for audio extraction of training sessions

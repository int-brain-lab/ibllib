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

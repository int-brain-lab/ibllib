# Task QC Viewer
This will download the TTL pulses and data collected on Bpod and/or FPGA and plot the results
alongside an interactive table.
The UUID is the session id. 

## Usage: command line

Launch the Viewer by typing `python task_qc.py session_UUID` , example:
```
python task_qc.py c9fec76e-7a20-4da4-93ad-04510a89473b
# or with ipython
ipython task_qc.py -- c9fec76e-7a20-4da4-93ad-04510a89473b
```

Or just using a local path (on a local server for example):
```
python task_qc.py /mnt/s0/Subjects/KS022/2019-12-10/001 --local
# or with ipython
ipython task_qc.py -- /mnt/s0/Subjects/KS022/2019-12-10/001 --local
```

## Usage: from ipython prompt
``` python
from iblapps.task_qc_viewer.task_qc import show_session_task_qc
session_path = "/datadisk/Data/IntegrationTests/ephys/choice_world_init/KS022/2019-12-10/001"
show_session_task_qc(session_path, local=True)
```

## Plots
1) Sync pulse display:
- TTL sync pulses (as recorded on the Bpod or FPGA for ephys sessions) for some key apparatus (i
.e. frame2TTL, audio signal). TTL pulse trains are displayed in black (time on x-axis, voltage on y-axis), offset by an increment of 1 each time (e.g. audio signal is on line 3, cf legend).
- trial event types, vertical lines (marked in different colours)

2) Wheel display:
- the wheel position in radians
- trial event types, vertical lines (marked in different colours)

3) Interactive table:
Each row is a trial entry.  Each column is a trial event

When double-clicking on any field of that table, the Sync pulse display time (x-) axis is adjusted so as to visualise the corresponding trial selected.

### What to look for
Tests are defined in the SINGLE METRICS section of ibllib/qc/task_metrics.py: https://github.com/int-brain-lab/ibllib/blob/master/ibllib/qc/task_metrics.py#L148-L149

### Exit
Close the GUI window containing the interactive table to exit.

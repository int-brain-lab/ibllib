"""
Use the framework developed by Niccolo B. to review the quality
of the task control.
Save data locally per EID as computations take time.
"""
# Author : Gaelle C.
import ibllib.qc.bpodqc_metrics as bpodqc
from ibllib.qc.bpodqc_extractors import load_bpod_data
from oneibl.one import ONE
from time import perf_counter
import numpy as np
from pathlib import Path
import os

one = ONE()
# Get list of all locations (some are labs, some are rigs)
locations = one.alyx.rest('locations', 'list')
# Filter to get only names containing _iblrig_
iblrig = [s['name'] for s in locations if "_iblrig_" in s['name']]
# Filter to get only names containing _ephys_
ephys_rig = [s for s in iblrig if "_ephys_" in s]

# -- Var init
dtypes = ['ephysData.raw.lf', 'ephysData.raw.meta', 'ephysData.raw.ch']

# Exception eids:
list_eid_reject = [
    # -- '_iblrig_mrsicflogel_ephys_0' -- (22 eids total, 3 failures)
    '614e1937-4b24-4ad3-9055-c8253d089919',  # KeyError: 'hide_stim'
    'd564619d-43d8-40bf-8525-50b09da2ecb7',  # KeyError: 'hide_stim'
    '68baa22a-232f-4b8f-b491-018c6375bab2',  # KeyError: 'hide_stim'


    # -- '_iblrig_angelaki_ephys_0' -- (17 eids total, 4 failures)
    '279fa50f-223c-43ac-ac0c-89753c77949e',  # KeyError: 'hide_stim'
    '4c53f746-7763-478d-b251-5315c26c4b5f',  # KeyError: 'hide_stim'
    'badc7140-b917-44a2-aa48-0e44f357baee',  # KeyError: 'hide_stim'
    '147d9be2-ab3a-4dd6-a9b8-fdf6fc129d84',  # KeyError: 'hide_stim'

    # -- '_iblrig_danlab_ephys_0' -- (37 eids total, 0 failures)

    # -- '_iblrig_churchlandlab_ephys_0' -- (52 sessions, 10 failures)
    'c607c5da-534e-4f30-97b3-b1d3e904e9fd',  # KeyError: 'hide_stim'
    '46b0d871-23d3-4630-8a6b-c79f99b2958c',  # KeyError: 'hide_stim'
    'b985b86f-e0e1-4d63-afaf-448b91cb4d74',  # KeyError: 'hide_stim'
    'f3f406bd-e138-44c2-8a02-7f11bf8ce87a',  # KeyError: 'hide_stim'
    'af5a1a37-9209-4c1e-8d7a-edf39ee4420a',  # KeyError: 'hide_stim'
    '63b83ddf-b7ea-40db-b1e2-93c2a769b6e5',  # KeyError: 'hide_stim'
    '713cf757-688f-4fc1-a2f6-2f997c9915c0',  # KeyError: 'hide_stim'
    'f6f947b8-c123-4e27-8933-f624a8c3e8cc',  # KeyError: 'hide_stim'
    '8c2e6449-57f0-4632-9f18-66e6ca90c522',  # KeyError: 'hide_stim'
    '4330cd7d-a513-4385-86ea-ca1a6cc04e1d',  # KeyError: 'hide_stim'

    # -- '_iblrig_zadorlab_ephys_0' -- (8 sessions, 0 failures)

    # -- '_iblrig_wittenlab_ephys_0' -- (3 sessions, x failures)
    '49368f16-de69-4647-9a7a-761e94517821',   # KeyError: 'hide_stim'
    '5139ce2c-7d52-44bf-8129-692d61dd6403',   # KeyError: 'hide_stim'
    '1211f4af-d3e4-4c4e-9d0b-75a0bc2bf1f0',   # KeyError: 'hide_stim'

    # -- '_iblrig_carandinilab_ephys_0' -- (24 sessions, x failures)
    'c6d5cea7-e1c4-48e1-8898-78e039fabf2b',   # KeyError: 'hide_stim'
    'aad23144-0e52-4eac-80c5-c4ee2decb198',   # KeyError: 'hide_stim'
    'a3df91c8-52a6-4afa-957b-3479a7d0897c',   # KeyError: 'hide_stim'
    '15f742e1-1043-45c9-9504-f1e8a53c1744',   # KeyError: 'hide_stim'
    'dd87e278-999d-478b-8cbd-b5bf92b84763',   # KeyError: 'hide_stim'
    'd6d829f9-a5b9-4ea5-a916-c7d2aadeccba'
]

# Saving path
cachepath = Path(one._par.CACHE_DIR)

# Plots for 1 rig at a time
for i_ephysrig in range(0, len(ephys_rig)):
    print(f'Rig {i_ephysrig + 1} / {len(ephys_rig)}')

    rig_location = ephys_rig[i_ephysrig]

    # Save folder
    outdir = cachepath.joinpath('BPODQC', rig_location)
    # Create target Directory if don't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get session eIDs, for 1 rig
    eIDs, ses_det = one.search(
        location=rig_location,
        # dataset_types=dtypes,
        task_protocol='_iblrig_tasks_ephysChoiceWorld',
        details=True)

    # Download all session and save data frame

    for i_eid in range(0, len(eIDs)):

        eid = eIDs[i_eid]
        if eid not in list_eid_reject:
            # Show session number and start compute time counter for session
            print(f'Session {i_eid+1} / {len(eIDs)}')
            start = perf_counter()

            # Start compute
            data = load_bpod_data(eid)
            bpod_frame = bpodqc.get_bpodqc_metrics_frame(eid, data=data, apply_criteria=False)
            bpod_pass = bpodqc.get_bpodqc_metrics_frame(eid, data=data, apply_criteria=True)

            # -- Show exec time
            end = perf_counter()
            execution_time = (end - start)
            print(execution_time)

            # -- Append and save variables
            app_token = {
                'eid': eid,
                'data': data,
                'bpod_frame': bpod_frame,
                'bpod_pass': bpod_pass,
                'ses_det': ses_det[i_eid]
            }

            outfile = Path.joinpath(outdir, f'{eid}__dataqc.npz')
            np.savez(outfile, dataqc=app_token)  # overwrite any existing file

            # -- Plot session metrics and save fig # TODO plot func not working yet
            # df = rearrange_metrics(bpod_frame)
            # savepath = ..
            # plot_metrics(df, ses_det[i_eid], save_path=None)

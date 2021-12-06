# RAW DATA
RAW_BEHAVIOUR = [
    ('_iblrig_ambientSensorData.raw', 'raw_behavior_data', True),
    ('_iblrig_encoderEvents.raw', 'raw_behavior_data', True),
    ('_iblrig_encoderPositions.raw', 'raw_behavior_data', True),
    ('_iblrig_encoderTrialInfo.raw', 'raw_behavior_data', True),
    ('_iblrig_micData.raw', 'raw_behavior_data', False),
    ('_iblrig_stimPositionScreen.raw', 'raw_behavior_data', False),
    ('_iblrig_syncSquareUpdate.raw', 'raw_behavior_data', False),
    ('_iblrig_taskData.raw', 'raw_behavior_data', True),
    ('_iblrig_taskSettings.raw', 'raw_behavior_data', True),
]

RAW_PASSIVE = [
    ('_iblrig_RFMapStim.raw', 'raw_passive_data', True),
    ('_iblrig_stimPositionScreen.raw', 'raw_passive_data', False),
    ('_iblrig_syncSquareUpdate.raw', 'raw_passive_data', False),
    ('_iblrig_taskSettings.raw', 'raw_passive_data', False),
    ('_iblrig_encoderEvents.raw', 'raw_passive_data', False),
    ('_iblrig_encoderPositions.raw', 'raw_passive_data', False),
    ('_iblrig_encoderTrialInfo.raw', 'raw_passive_data', False),
]

# Data common to both 3A and 3B
RAW_EPHYS = [
    ['ephysData.raw.ap', 'raw_ephys_data/XX', True],
    ['ephysData.raw.ch', 'raw_ephys_data/XX', True, ['ap', 'lf']],
    ['ephysData.raw.lf', 'raw_ephys_data/XX', True],
    ['ephysData.raw.meta', 'raw_ephys_data/XX', True, ['ap', 'lf']],
]

# These are for both 3B probes but only main probe for 3A
RAW_EPHYS_EXTRA = [
    ['ephysData.raw.sync', 'raw_ephys_data/XX', True],
    ['ephysData.raw.timestamps', 'raw_ephys_data/XX', True],
    ['ephysData.raw.wiring', 'raw_ephys_data/XX', False],
]

# Only for 3B
RAW_EPHYS_NIDAQ = [
    ('ephysData.raw.nidq', 'raw_ephys_data', True),
    ('ephysData.raw.meta', 'raw_ephys_data', True, ['nidq']),
    ('ephysData.raw.ch', 'raw_ephys_data', True, ['nidq']),
]


RAW_VIDEO = [
    ('_iblrig_Camera.raw', 'raw_video_data', True, ['left', 'right', 'body']),
    ('_iblrig_Camera.timestamps', 'raw_video_data', True, ['left', 'right', 'body']),
    ('_iblrig_Camera.GPIO', 'raw_video_data', False, ['left', 'right', 'body']),
    ('_iblrig_Camera.frame_counter', 'raw_video_data', False, ['left', 'right', 'body']),
]

# PROCESSED DATA
TRIALS = [
    ('trials.choice', 'alf', True),
    ('trials.contrastLeft', 'alf', True),
    ('trials.contrastRight', 'alf', True),
    ('trials.feedbackType', 'alf', True),
    ('trials.feedback_times', 'alf', True),
    ('trials.firstMovement_times', 'alf', True),
    ('trials.goCueTrigger_times', 'alf', True),
    ('trials.goCue_times', 'alf', True),
    ('trials.intervals', 'alf', True),
    ('trials.intervals', 'alf', True),
    ('trials.probabilityLeft', 'alf', True),
    ('trials.response_times', 'alf', True),
    ('trials.rewardVolume', 'alf', True),
    ('trials.stimOff_times', 'alf', True),
    ('trials.stimOn_times', 'alf', True),
]

WHEEL = [
    ('wheel.position', 'alf', True),
    ('wheel.timestamps', 'alf', True),
    ('wheelMoves.intervals', 'alf', True),
    ('wheelMoves.peakAmplitude', 'alf', True),
]

PASSIVE = [
    ('_ibl_passiveGabor.table', 'alf', True),
    ('_ibl_passivePeriods.intervalsTable', 'alf', True),
    ('_ibl_passiveRFM.times', 'alf', True),
    ('_ibl_passiveStims.table', 'alf', True),
]


DLC = [
    ('camera.dlc', 'alf', True, ['left', 'right', 'body']),
    ('camera.times', 'alf', True, ['left', 'right', 'body']),
    ('camera.ROIMotionEnergy', 'alf', False, ['left', 'right', 'body']),
    ('ROIMotionEnergy.position', 'alf', False, ['left', 'right', 'body']),
]

VIDEO = [
    ('camera.times', 'alf', True, ['left', 'right', 'body']),
]

# Data common to both 3A and 3B
EPHYS = [
    ['_spikeglx_sync.channels', 'raw_ephys_data/XX', True],
    ['_spikeglx_sync.polarities', 'raw_ephys_data/XX', True],
    ['_spikeglx_sync.times', 'raw_ephys_data/XX', True],
    ['_iblqc_ephysSpectralDensity.freqs', 'raw_ephys_data/XX', True, ['ap', 'lf']],
    ['_iblqc_ephysSpectralDensity.power', 'raw_ephys_data/XX', True, ['ap', 'lf']],
    ['_iblqc_ephysTimeRms.rms', 'raw_ephys_data/XX', True, ['ap', 'lf']],
    ['_iblqc_ephysTimeRms.timestamps', 'raw_ephys_data/XX', True, ['ap', 'lf']],
]

# Only for 3B
EPHYS_NIDAQ = [
    ('_spikeglx_sync.channels', 'raw_ephys_data', True),
    ('_spikeglx_sync.polarities', 'raw_ephys_data', True),
    ('_spikeglx_sync.times', 'raw_ephys_data', True),
]

SPIKE_SORTING = [
    ['_phy_spikes_subset.channels', 'alf/XX', False],
    ['_phy_spikes_subset.spikes', 'alf/XX', False],
    ['_phy_spikes_subset.waveforms', 'alf/XX', False],
    ['channels.brainLocationIds_ccf_2017', 'alf/XX', False],
    ['channels.mlapdv', 'alf/XX', False],
    ['channels.localCoordinates', 'alf/XX', True],
    ['channels.rawInd', 'alfa/XX', True],
    ['clusters.amps', 'alf/XX', True],
    ['clusters.brainLocationAcronyms_ccf_2017', 'alf/XX', False],
    ['clusters.brainLocationIds_ccf_2017', 'alf/XX', False],
    ['clusters.channels', 'alf/XX', True],
    ['clusters.depths', 'alf/XX', True],
    ['clusters.metrics', 'alf/XX', False],
    ['clusters.mlapdv', 'alf/XX', False],
    ['clusters.peakToTrough', 'alf/XX', True],
    ['clusters.uuids', 'alf/XX', True],
    ['clusters.waveforms', 'alf/XX', True],
    ['clusters.waveformsChannels', 'alf/XX', True],
    ['spikes.amps', 'alf/XX', True],
    ['spikes.clusters', 'alf/XX', True],
    ['spikes.depths', 'alf/XX', True],
    ['spikes.samples', 'alf/XX', True],
    ['spikes.templates', 'alf/XX', True],
    ['spikes.times', 'alf/XX', True],
    ['templates.amps', 'alf/XX', True],
    ['templates.waveforms', 'alf/XX', True],
    ['templates.waveformsChannels', 'alf/XX', True],
]


# RAW DATA TASKS
RAW_BEHAVIOUR_TASKS = ['TrainingRegisterRaw', 'EphysAudio']
RAW_PASSIVE_TASKS = ['TrainingRegisterRaw']
RAW_EPHYS_TASKS = ['EphysMtscomp']
RAW_VIDEO_TASKS = ['TrainingRegisterRaw', 'EphysVideoCompress']

# PROCESSED DATA TASKS
PASSIVE_TASKS = ['EphysPassive']
EPHYS_TASKS = ['EphysPulses', 'RawEphysQC']
VIDEO_TASKS = ['EphysVideoCompress']
TRIAL_TASKS = ['EphysTrials']
WHEEL_TASKS = ['EphysTrials']
SPIKE_SORTING_TASKS = ['SpikeSorting', 'EphysCellsQc']
DLC_TASKS = ['EphysDLC']

# DATA REQUIRED FOR TASKS
EPHYSTRIALS = [('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
               ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
               ('_spikeglx_sync.npy', 'raw_ephys_data*', True),
               ('_spikeglx_sync.polarities.npy', 'raw_ephys_data*', True),
               ('_spikeglx_sync.times.npy', 'raw_ephys_data*', True),
               ('_iblrig_encoderEvents.raw', 'raw_behavior_data', True),
               ('_iblrig_encoderPositions.raw', 'raw_behavior_data', True),
               ('*wiring.json', 'raw_ephys_data*', False),
               ('*.meta', 'raw_ephys_data*', True)]

EPHYSPASSIVE = [('_iblrig_taskSettings.raw*', 'raw_behavior_data', True),
                ('_spikeglx_sync.channels.*', 'raw_ephys_data*', True),
                ('_spikeglx_sync.polarities.*', 'raw_ephys_data*', True),
                ('_spikeglx_sync.times.*', 'raw_ephys_data*', True),
                ('*.meta', 'raw_ephys_data*', True),
                ('_iblrig_RFMapStim.raw*', 'raw_passive_data', True)]

RAWEPHYSQC = [('*.meta', 'raw_ephys_data/probe*', True),
              ('*.ch', 'raw_ephys_data/probe*', True),
              ('*.cbin', 'raw_ephys_data/probe*', True)]


def spike_sorting_signature(pname=None):
    pname = pname if pname is not None else "probe*"
    signature = [('*ap.meta', f'raw_ephys_data/{pname}', True),
                 ('*ap.ch', f'raw_ephys_data/{pname}', True),
                 ('*ap.cbin', f'raw_ephys_data/{pname}', True),
                 ('_spikeglx_sync.channels.*', 'raw_ephys_data*', True),
                 ('_spikeglx_sync.polarities.*', 'raw_ephys_data*', True),
                 ('_spikeglx_sync.times.*', 'raw_ephys_data*', True),
                 ('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                 ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True)
                 ]
    return signature


SPIKESORTING = [spike_sorting_signature()]

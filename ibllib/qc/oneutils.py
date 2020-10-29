import logging

from oneibl.one import ONE

log = logging.getLogger("ibllib")


def download_taskqc_raw_data(eid, one=None, fpga=False):
    """Download raw data required for performing task QC

    :param eid: A session UUID string
    :param one: An instance of ONE with which to download the data
    :param fpga: When True, downloads the raw ephys data required for extracting the FPGA task data
    :return: A list of file paths for the downloaded raw data
    """
    one = one or ONE()
    # Datasets required for extracting task data
    dstypes = [
        "_iblrig_taskData.raw",
        "_iblrig_taskSettings.raw",
        "_iblrig_encoderPositions.raw",
        "_iblrig_encoderEvents.raw",
        "_iblrig_stimPositionScreen.raw",
        "_iblrig_syncSquareUpdate.raw",
        "_iblrig_encoderTrialInfo.raw",
        "_iblrig_ambientSensorData.raw",
    ]
    # Extra files required for extracting task data from FPGA
    if fpga:
        dstypes.extend(['_spikeglx_sync.channels',
                        '_spikeglx_sync.polarities',
                        '_spikeglx_sync.times',
                        'ephysData.raw.meta',
                        'ephysData.raw.wiring'])
    # Download the data via ONE
    return one.load(eid, dataset_types=dstypes, download_only=True)

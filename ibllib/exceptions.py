class SyncBpodWheelException(ValueError):
    """
    When the bpod can't be synchronized with the data from Bonsai
    """
    pass


class SyncBpodFpgaException(ValueError):
    """
    When the bpod can't be synchronized with the FPGA because of overlap issue
    """
    pass

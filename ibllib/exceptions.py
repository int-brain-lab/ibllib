class IblError(BaseException):
    explanation = ''

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        return f"{self.message} \n {self.explanation} "


class SyncBpodWheelException(IblError):
    explanation = "The bpod can't be synchronized with the Rotary Encoder."


class SyncBpodFpgaException(IblError):
    explanation = "The bpod can't be synchronized with the FPGA."


class Neuropixel3BSyncFrontsNonMatching(IblError):
    explanation = (" When the npy files containing sync pulses for probes do not match with nidq."
                   "In 3B, this indicates that either the binary files is corrupt,"
                   "either the extracted sync files are corrupt.")

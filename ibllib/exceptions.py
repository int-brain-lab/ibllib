class IblError(Exception):
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


class AlyxSubjectNotFound(IblError):
    explanation = 'The subject was not found in Alyx database'


class ALFMultipleObjectsFound(IblError):
    explanation = ('The search object was not found.  ALF names have the pattern '
                   '(_namespace_)object.attribute(_timescale).extension, e.g. for the file '
                   '"_ibl_trials.intervals.npy" the object is "trials"')


class ALFMultipleCollectionsFound(IblError):
    explanation = ('The matching object/file(s) belong to more than one collection.  '
                   'ALF names have the pattern '
                   'collection/(_namespace_)object.attribute(_timescale).extension, e.g. for the '
                   'file "alf/probe01/spikes.times.npy" the collection is "alf/probe01"')


class ALFObjectNotFound(IblError):
    explanation = ('The ALF object was not found.  This may occur if the object or namespace or '
                   'incorrectly formatted e.g. the object "_ibl_trials.intervals.npy" would be '
                   'found with the filters `object="trials", namespace="ibl"`')


class NvidiaDriverNotReady(IblError):
    explanation = ('Nvidia driver does not respond. This usually means the GPU is inaccessible '
                   'and needs to be recovered through a system reboot.')

from collections import OrderedDict

import numpy as np
from ibllib.io.extractors import ephys_fpga
from ibllib.dsp.utils import sync_timestamps
from ibllib.plots import squares, vertical_lines
from ibllib.pipes import tasks

from ibllib.pipes.ephys_preprocessing import (
    EphysRegisterRaw, EphysPulses, RawEphysQC, EphysAudio, EphysMtscomp, EphysVideoCompress,
    EphysCellsQc, EphysDLC, SpikeSorting)

LASER_PULSE_DURATION_SECS = .5
LASER_PROBABILITY = .8
DISPLAY = True


class EphysPassiveOptoTrials(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 1

    def _run(self):
        sync, sync_map = ephys_fpga.get_main_probe_sync(self.session_path)
        bpod = ephys_fpga.get_sync_fronts(sync, sync_map['bpod'])
        laser_ttl = ephys_fpga.get_sync_fronts(sync, sync_map['laser_ttl'])
        t_bpod = bpod['times'][bpod['polarities'] == 1]
        t_laser = laser_ttl['times'][laser_ttl['polarities'] == 1]
        _, _, ibpod, ilaser = sync_timestamps(t_bpod, t_laser, return_indices=True)

        if DISPLAY:
            for ch in np.arange(3):
                ch0 = ephys_fpga.get_sync_fronts(sync, 16 + ch)
                squares(ch0['times'], ch0['polarities'], yrange=[-.5 + ch, .5 + ch])
            vertical_lines(t_bpod[ibpod], ymax=4)

        trial_starts = t_bpod
        trial_starts[ibpod] = t_laser[ilaser]
        ntrials = trial_starts.size

        laser_intervals = np.zeros((ntrials, 2)) * np.nan
        laser_intervals[ibpod, 0] = t_laser[ilaser]
        laser_intervals[ibpod, 1] = t_laser[ilaser] + LASER_PULSE_DURATION_SECS
        intervals = np.zeros((ntrials, 2)) * np.nan
        intervals[:, 0] = trial_starts
        intervals[:, 1] = np.r_[trial_starts[1:], np.nan]
        alf_path = self.session_path.joinpath('alf')
        alf_path.mkdir(parents=True, exist_ok=True)

        out_files = [
            alf_path.joinpath("_ibl_trials.laser_probability.npy"),
            alf_path.joinpath("_ibl_trials.laser_intervals.npy"),
            alf_path.joinpath("_ibl_trials.intervals.npy"),
        ]
        np.save(alf_path.joinpath("_ibl_trials.laser_intervals.npy"), laser_intervals)
        np.save(alf_path.joinpath("_ibl_trials.laser_probability.npy"), trial_starts * 0 + LASER_PROBABILITY)
        np.save(alf_path.joinpath("_ibl_trials.intervals.npy"), intervals)

        return out_files


class EphysPassiveOptoPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(EphysPassiveOptoPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = EphysRegisterRaw(self.session_path)
        tasks["EphysPulses"] = EphysPulses(self.session_path)
        tasks["EphysRawQC"] = RawEphysQC(self.session_path)
        tasks["EphysAudio"] = EphysAudio(self.session_path)
        tasks["EphysMtscomp"] = EphysMtscomp(self.session_path)
        # level 1
        tasks["SpikeSorting"] = SpikeSorting(
            self.session_path, parents=[tasks["EphysMtscomp"], tasks["EphysPulses"]])
        tasks["EphysPassiveOptoTrials"] = EphysPassiveOptoTrials(self.session_path, parents=[tasks["EphysPulses"]])
        tasks["EphysVideoCompress"] = EphysVideoCompress(
            self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        self.tasks = tasks


#
# from one.api import ONE
#
# one = ONE()
# session_path = Path("/mnt/s0/Data/Subjects/KS056/2021-07-18/001")
#
# session_path = "/mnt/iblserver2/ephys_recordings/KS056/2021-07-18/001"
#
# pipeline = EphysPassiveOptoPipeline(session_path, one=one)
# pipeline.run()

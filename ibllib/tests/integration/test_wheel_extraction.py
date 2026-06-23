import numpy as np

from scipy.signal import butter, filtfilt
import scipy.interpolate
import matplotlib.pyplot as plt

from one.api import ONE
from one.alf.path import get_session_path
from ibllib.io.extractors import training_wheel
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq

from ci.tests import base

DISPLAY = False


def compare_wheel_fpga_behaviour(session_path, display=DISPLAY):
    task = ChoiceWorldTrialsNidq(session_path, one=ONE(mode='local'),
                                 collection='raw_behavior_data', sync_collection='raw_ephys_data')
    fpga_trials, _ = task.extract_behaviour(save=False, tmax=None)
    bpod_trials = task.extractor.bpod_trials
    fpga_t, fpga_pos = fpga_trials['wheel_timestamps'], fpga_trials['wheel_position']
    bpod_t, bpod_pos = bpod_trials['wheel_timestamps'], bpod_trials['wheel_position']

    # resample both traces to the same rate and compute correlation coeff
    bpod_t = task.extractor.bpod2fpga(bpod_t)
    tmin = max(np.min(fpga_t), np.min(bpod_t))
    tmax = min(np.max(fpga_t), np.max(bpod_t))
    wheel = {'tscale': np.arange(tmin, tmax, 0.01)}
    wheel['fpga'] = scipy.interpolate.interp1d(
        fpga_t, fpga_pos)(wheel['tscale'])
    wheel['bpod'] = scipy.interpolate.interp1d(
        bpod_t, bpod_pos)(wheel['tscale'])
    if display:
        plt.figure()
        plt.plot(fpga_t - task.extractor.bpod2fpga(0), fpga_pos, '*')
        plt.plot(bpod_t - task.extractor.bpod2fpga(0), bpod_pos, '.')
    raw_wheel = {'fpga_t': fpga_t, 'fpga_pos': fpga_pos, 'bpod_t': bpod_t, 'bpod_pos': bpod_pos}
    return raw_wheel, wheel


class TestWheelExtractionSimpleEphys(base.IntegrationTest):

    required_files = ['wheel/ephys/three_clockwise_revolutions']

    def setUp(self) -> None:
        self.session_path = \
            self.data_path.joinpath('wheel', 'ephys', 'three_clockwise_revolutions')
        assert self.session_path.exists()

    def test_three_clockwise_revolutions_fpga(self):
        raw_wheel, wheel = compare_wheel_fpga_behaviour(self.session_path)
        self.assertTrue(np.all(np.abs(wheel['fpga'] - wheel['bpod']) < 0.1))
        # test that the units are in radians: we expect around 9 revolutions clockwise
        self.assertTrue(0.95 < raw_wheel['fpga_pos'][-1] / -(2 * 3.14 * 9) < 1.05)


class TestWheelExtractionSessionEphys(base.IntegrationTest):

    required_files = ['wheel/ephys/sessions']

    def setUp(self) -> None:
        self.root_path = self.data_path.joinpath('wheel', 'ephys', 'sessions')
        if not self.root_path.exists():
            return
        self.sessions = [f.parent for f in self.root_path.rglob('raw_behavior_data')]

    def test_wheel_extraction_session(self):
        for session_path in self.sessions:
            with self.subTest(msg=session_path):
                _, wheel = compare_wheel_fpga_behaviour(session_path)
                # makes sure that the HF component matches
                b, a = butter(3, 0.0001, btype='high', analog=False)
                fpga = filtfilt(b, a, wheel['fpga'])
                bpod = filtfilt(b, a, wheel['bpod'])
                # plt.figure()
                # plt.plot(wheel['tscale'], fpga)
                # plt.plot(wheel['tscale'], bpod)
                self.assertTrue(np.all(np.abs(fpga - bpod < 0.1)))


class TestWheelExtractionTraining(base.IntegrationTest):

    required_files = ['wheel/training']

    def setUp(self) -> None:
        self.root_path = self.data_path.joinpath('wheel', 'training')
        assert self.root_path.exists()

    def test_wheel_extraction_training(self):
        for rbf in self.root_path.rglob('raw_behavior_data'):
            session_path = get_session_path(rbf)
            with self.subTest(msg=session_path):
                bpod_t, _ = training_wheel.get_wheel_position(session_path)
                self.assertTrue(bpod_t.size)

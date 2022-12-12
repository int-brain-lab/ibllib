from pathlib import Path, PureWindowsPath

from pkg_resources import parse_version
import numpy as np
from one.alf.io import AlfBunch

from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.training_trials import (
    Choice, FeedbackTimes, FeedbackType, GoCueTimes, GoCueTriggerTimes,
    IncludedTrials, Intervals, ProbabilityLeft, ResponseTimes, RewardVolume,
    StimOnTimes_deprecated, StimOnTriggerTimes, StimOnOffFreezeTimes, ItiInTimes,
    StimOffTriggerTimes, StimFreezeTriggerTimes, ErrorCueTriggerTimes, PhasePosQuiescence)
from ibllib.io.extractors.training_wheel import Wheel


class ContrastLR(BaseBpodTrialsExtractor):
    """
    Get left and right contrasts from raw datafile.
    """
    save_names = ('_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy')
    var_names = ('contrastLeft', 'contrastRight')

    def _extract(self, **kwargs):
        contrastLeft = np.array([t['contrast'] if np.sign(
            t['position']) < 0 else np.nan for t in self.bpod_trials])
        contrastRight = np.array([t['contrast'] if np.sign(
            t['position']) > 0 else np.nan for t in self.bpod_trials])
        return contrastLeft, contrastRight


class ProbaContrasts(BaseBpodTrialsExtractor):
    """
    Bpod pre-generated values for probabilityLeft, contrastLR, phase, quiescence
    """
    save_names = ('_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy', None, None,
                  '_ibl_trials.probabilityLeft.npy', None)
    var_names = ('contrastLeft', 'contrastRight', 'phase',
                 'position', 'probabilityLeft', 'quiescence')

    def _extract(self, **kwargs):
        """Extracts positions, contrasts, quiescent delay, stimulus phase and probability left
        from pregenerated session files.  Used in ephysChoiceWorld extractions.
        Optional: saves alf contrastLR and probabilityLeft npy files"""
        pe = self.get_pregenerated_events(self.bpod_trials, self.settings)
        return [pe[k] for k in sorted(pe.keys())]

    @staticmethod
    def get_pregenerated_events(bpod_trials, settings):
        num = settings.get("PRELOADED_SESSION_NUM", None)
        if num is None:
            num = settings.get("PREGENERATED_SESSION_NUM", None)
        if num is None:
            fn = settings.get('SESSION_LOADED_FILE_PATH', '')
            fn = PureWindowsPath(fn).name
            num = ''.join([d for d in fn if d.isdigit()])
            if num == '':
                raise ValueError("Can't extract left probability behaviour.")
        # Load the pregenerated file
        ntrials = len(bpod_trials)
        sessions_folder = Path(raw.__file__).parent.joinpath(
            "extractors", "ephys_sessions")
        fname = f"session_{num}_ephys_pcqs.npy"
        pcqsp = np.load(sessions_folder.joinpath(fname))
        pos = pcqsp[:, 0]
        con = pcqsp[:, 1]
        pos = pos[: ntrials]
        con = con[: ntrials]
        contrastRight = con.copy()
        contrastLeft = con.copy()
        contrastRight[pos < 0] = np.nan
        contrastLeft[pos > 0] = np.nan
        qui = pcqsp[:, 2]
        qui = qui[: ntrials]
        phase = pcqsp[:, 3]
        phase = phase[: ntrials]
        pLeft = pcqsp[:, 4]
        pLeft = pLeft[: ntrials]

        phase_path = sessions_folder.joinpath(f"session_{num}_stim_phase.npy")
        is_patched_version = parse_version(
            settings.get('IBLRIG_VERSION_TAG', 0)) > parse_version('6.4.0')
        if phase_path.exists() and is_patched_version:
            phase = np.load(phase_path)[:ntrials]

        return {'position': pos, 'quiescence': qui, 'phase': phase, 'probabilityLeft': pLeft,
                'contrastRight': contrastRight, 'contrastLeft': contrastLeft}


class TrialsTableBiased(BaseBpodTrialsExtractor):
    """
    Extracts the following into a table from Bpod raw data:
        intervals, goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight,
        feedback_times, feedbackType, rewardVolume, probabilityLeft, firstMovement_times
    Additionally extracts the following wheel data:
        wheel_timestamps, wheel_position, wheel_moves_intervals, wheel_moves_peak_amplitude
    """
    save_names = ('_ibl_trials.table.pqt', None, None, '_ibl_wheel.timestamps.npy', '_ibl_wheel.position.npy',
                  '_ibl_wheelMoves.intervals.npy', '_ibl_wheelMoves.peakAmplitude.npy', None, None)
    var_names = ('table', 'stimOff_times', 'stimFreeze_times', 'wheel_timestamps', 'wheel_position', 'wheel_moves_intervals',
                 'wheel_moves_peak_amplitude', 'peakVelocity_times', 'is_final_movement')

    def _extract(self, extractor_classes=None, **kwargs):
        base = [Intervals, GoCueTimes, ResponseTimes, Choice, StimOnOffFreezeTimes, ContrastLR, FeedbackTimes, FeedbackType,
                RewardVolume, ProbabilityLeft, Wheel]
        out, _ = run_extractor_classes(base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings,
                                       save=False, task_collection=self.task_collection)

        table = AlfBunch({k: out.pop(k) for k in list(out.keys()) if k not in self.var_names})
        assert len(table.keys()) == 12

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')


class TrialsTableEphys(BaseBpodTrialsExtractor):
    """
    Extracts the following into a table from Bpod raw data:
        intervals, goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight,
        feedback_times, feedbackType, rewardVolume, probabilityLeft, firstMovement_times
    Additionally extracts the following wheel data:
        wheel_timestamps, wheel_position, wheel_moves_intervals, wheel_moves_peak_amplitude
    """
    save_names = ('_ibl_trials.table.pqt', None, None, '_ibl_wheel.timestamps.npy', '_ibl_wheel.position.npy',
                  '_ibl_wheelMoves.intervals.npy', '_ibl_wheelMoves.peakAmplitude.npy', None, None, None, None, None)
    var_names = ('table', 'stimOff_times', 'stimFreeze_times', 'wheel_timestamps', 'wheel_position', 'wheel_moves_intervals',
                 'wheel_moves_peak_amplitude', 'peakVelocity_times', 'is_final_movement',
                 'phase', 'position', 'quiescence')

    def _extract(self, extractor_classes=None, **kwargs):
        base = [Intervals, GoCueTimes, ResponseTimes, Choice, StimOnOffFreezeTimes, ProbaContrasts,
                FeedbackTimes, FeedbackType, RewardVolume, Wheel]
        # Exclude from trials table
        out, _ = run_extractor_classes(base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings,
                                       save=False, task_collection=self.task_collection)
        table = AlfBunch({k: v for k, v in out.items() if k not in self.var_names})
        assert len(table.keys()) == 12

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')


def extract_all(session_path, save=False, bpod_trials=False, settings=False, extra_classes=None,
                task_collection='raw_behavior_data', save_path=None):
    """
    Same as training_trials.extract_all except...
     - there is no RepNum
     - ContrastLR is extracted differently
     - IncludedTrials is only extracted for 5.0.0 or greater

    :param session_path:
    :param save:
    :param bpod_trials:
    :param settings:
    :param extra_classes: additional BaseBpodTrialsExtractor subclasses for custom extractions
    :return:
    """
    if not bpod_trials:
        bpod_trials = raw.load_data(session_path, task_collection=task_collection)
    if not settings:
        settings = raw.load_settings(session_path, task_collection=task_collection)
    if settings is None:
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    if settings['IBLRIG_VERSION_TAG'] == '':
        settings['IBLRIG_VERSION_TAG'] = '100.0.0'

    base = [GoCueTriggerTimes]
    # Version check
    if parse_version(settings['IBLRIG_VERSION_TAG']) >= parse_version('5.0.0'):
        # We now extract a single trials table
        base.extend([
            StimOnTriggerTimes, ItiInTimes, StimOffTriggerTimes, StimFreezeTriggerTimes, ErrorCueTriggerTimes,
            TrialsTableBiased, IncludedTrials, PhasePosQuiescence
        ])
    else:
        base.extend([
            Intervals, Wheel, FeedbackType, ContrastLR, ProbabilityLeft, Choice,
            StimOnTimes_deprecated, RewardVolume, FeedbackTimes, ResponseTimes, GoCueTimes, PhasePosQuiescence
        ])

    if extra_classes:
        base.extend(extra_classes)

    out, fil = run_extractor_classes(base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings,
                                     task_collection=task_collection, path_out=save_path)
    return out, fil

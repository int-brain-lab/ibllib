import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np

from packaging import version
import ibllib.pipes.behavior_tasks as btasks
from ibllib.io.extractors.bpod_trials import TrainingTrials
import ibllib.io.raw_data_loaders as rawio
from one.api import One
import one.alf.io as alfio

from ci.tests import base

TRIAL_KEYS = ['goCue_times', 'probabilityLeft', 'intervals', 'goCueTrigger_times', 'quiescencePeriod',
              'response_times', 'feedbackType', 'contrastLeft', 'feedback_times',
              'rewardVolume', 'choice', 'contrastRight', 'stimOn_times', 'firstMovement_times',
              'stimOnTrigger_times', 'included', 'stimOffTrigger_times', 'stimOff_times']
TRIAL_KEYS_TRAINING = TRIAL_KEYS + ['repNum']
TRIAL_KEYS_TRAINING.pop(TRIAL_KEYS.index('included'))

WHEEL_KEYS = ['position', 'timestamps']


class TestHabituation(base.IntegrationTest):

    def test_legacy_habituation_session(self):
        session_path = self.data_path.joinpath('Subjects_init/ZM_1098/2019-01-25/001')
        job = btasks.HabituationTrialsBpod(session_path, one=One(mode='local'), collection='raw_behavior_data')
        status = job.run(update=False)
        self.assertEqual(0, status)
        self.assertIn('No extraction of legacy habituation sessions', job.log)


class TestSessions(base.IntegrationTest):
    required_files = [
        'training/CSHL_003/2019-04-05/001',
        'training/CSHL_007/2019-07-31/001',  # only the down-going front of a trial was detected
        'training/ZM_1150/2019-05-07/001',
        'Subjects_init/IBL_46/2019-02-19/001',  # timestamps a million years in future
        'Subjects_init/ZM_335/2018-12-13/001',  # rotary encoder ms instead of us
        'Subjects_init/ZM_1085/2019-02-12/002',  # rotary encoder corrupt
        'Subjects_init/ZM_1085/2019-07-01/001',  # training session rig version 5.0.0
    ]

    def setUp(self):
        self.INIT_FOLDERS = list(map(self.data_path.joinpath, self.required_files))
        if not all(map(Path.exists, self.INIT_FOLDERS)):
            raise FileNotFoundError('missing fixure folders')
        self.one = One(mode='local')

    def test_trials_extraction(self):
        # extract all sessions
        with tempfile.TemporaryDirectory() as tdir:
            subjects_path = Path(tdir).joinpath('Subjects')
            for init_folder in self.INIT_FOLDERS:
                shutil.copytree(init_folder, subjects_path.joinpath(*init_folder.parts[-4:]))
            for fil in subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
                # read task settings and determine iblrig version to throw into subtests
                session_path = fil.parents[1]
                settings = rawio.load_settings(session_path)
                iblrig_version = version.parse(settings['IBLRIG_VERSION'])
                with self.subTest(file=fil, iblrig_version=iblrig_version):
                    # task running part
                    job = btasks.ChoiceWorldTrialsBpod(session_path, one=self.one, collection='raw_behavior_data')
                    job.run(update=False)
                    # check the trials objects
                    trials = alfio.load_object(session_path / 'alf', 'trials')
                    self.assertTrue(alfio.check_dimensions(trials) == 0)
                    tkeys = TRIAL_KEYS_TRAINING if isinstance(job.extractor, TrainingTrials) else TRIAL_KEYS
                    self.assertEqual(set(trials.keys()), set(tkeys))
                    """
                    For CSHL_007/2019-07-31/001 only the down-going front of a trial was detected,
                    resulting in an error for the go cue time. The fix was to extract the down-going
                    front and subtract 100ms.
                    """
                    if 'CSHL_007' in session_path.as_posix():
                        self.assertTrue(np.all(np.logical_not(np.isnan(trials.goCue_times))))
                    # check the wheel object if the extraction didn't fail
                    if job.status != -1:
                        wheel = alfio.load_object(session_path / 'alf', 'wheel')
                        self.assertTrue(alfio.check_dimensions(wheel) == 0)
                        self.assertEqual(set(wheel.keys()), set(WHEEL_KEYS))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

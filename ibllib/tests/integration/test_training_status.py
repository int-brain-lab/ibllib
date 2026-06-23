from ci.tests import base
from one.api import One
import one.alf.io as alfio
import shutil
from ibllib.pipes import training_status
import numpy as np


class TestTrainingStatus(base.IntegrationTest):
    """Test training status computations."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.subj_path = cls.default_data_root().joinpath('training_status', 'PL023')
        print('Building ONE cache from filesystem...')
        cls.one = One.setup(cls.subj_path, silent=True)

        cls.session_path = cls.subj_path.joinpath('2021-08-03', '002')
        cls.temp_results = cls.subj_path.joinpath('training_results_temp')
        cls.new_session1 = cls.subj_path.joinpath('2021-08-05', '001')
        cls.new_session2 = cls.subj_path.joinpath('2021-09-08', '001')

    @classmethod
    def tearDownClass(cls) -> None:
        for file in cls.subj_path.glob('*.pqt'):
            file.unlink()

    def tearDown(self) -> None:
        if self.new_session1.exists():
            shutil.rmtree(self.new_session1.parent)

        if self.new_session2.exists():
            shutil.rmtree(self.new_session2.parent)

        if training_status.save_path(self.subj_path).exists():
            training_status.save_path(self.subj_path).unlink()

    def test_missing_dates(self):
        # When no dataframe present
        df = training_status.load_existing_dataframe(self.subj_path)
        missing_dates = training_status.check_up_to_date(self.subj_path, df)
        expected_dates = [date_path.stem for date_path in self.subj_path.glob('2021*')]
        self.assertCountEqual(missing_dates.date.unique(), expected_dates)

        # Add some new sessions
        shutil.copytree(self.session_path, self.new_session1)
        shutil.copytree(self.session_path, self.new_session2)
        shutil.copy(self.temp_results.joinpath('training.csv'), training_status.save_path(self.subj_path))
        df = training_status.load_existing_dataframe(self.subj_path)
        missing_dates = training_status.check_up_to_date(self.subj_path, df)
        expected_dates = [self.new_session1.parent.stem, self.new_session2.parent.stem]

        self.assertCountEqual(missing_dates.date.unique(), expected_dates)

    def test_recompute_date(self):
        training_status.load_existing_dataframe(self.subj_path)
        shutil.copy(self.temp_results.joinpath('training_missing_latest.csv'), training_status.save_path(self.subj_path))
        df = training_status.load_existing_dataframe(self.subj_path)
        recompute_date = training_status.find_earliest_recompute_date(df.drop_duplicates('date').reset_index(drop=True))

        assert np.array_equal(recompute_date, ['2021-09-07'])

        shutil.copy(self.temp_results.joinpath('training_missing.csv'), training_status.save_path(self.subj_path))
        df = training_status.load_existing_dataframe(self.subj_path)
        recompute_date = training_status.find_earliest_recompute_date(df.drop_duplicates('date').reset_index(drop=True))
        assert recompute_date[0] == '2021-08-20'
        assert recompute_date[-1] == '2021-09-07'

    def test_training_hierachy(self):
        status = training_status.pass_through_training_hierachy('trained 1b', 'trained 1a')
        assert status == 'trained 1b'

        status = training_status.pass_through_training_hierachy('trained 1a', 'trained 1a')
        assert status == 'trained 1a'

        status = training_status.pass_through_training_hierachy('ready4delay', 'trained 1b')
        assert status == 'ready4delay'

    def test_concatentate_trials(self):
        paths = list(self.subj_path.joinpath('2021-08-13').glob('*'))
        trials1 = alfio.load_object(paths[0].joinpath('alf'), 'trials')
        trials2 = alfio.load_object(paths[1].joinpath('alf'), 'trials')

        concat_trials = training_status.load_combined_trials(paths, self.one)
        for key in concat_trials.keys():
            np.testing.assert_equal(np.r_[trials1[key], trials2[key]], concat_trials[key])

    def test_training_computation(self):

        training_status.get_latest_training_information(self.session_path, self.one)
        df = training_status.load_existing_dataframe(self.subj_path)

        status = df.drop_duplicates(subset='training_status', keep='first')
        assert status.loc[status['training_status'] == 'trained 1a', 'date'].values[0] == '2021-08-04'
        assert status.loc[status['training_status'] == 'trained 1b', 'date'].values[0] == '2021-08-17'
        assert status.loc[status['training_status'] == 'ready4ephysrig', 'date'].values[0] == '2021-08-25'
        assert status.loc[status['training_status'] == 'ready4delay', 'date'].values[0] == '2021-08-26'
        assert status.loc[status['training_status'] == 'ready4recording', 'date'].values[0] == '2021-09-06'

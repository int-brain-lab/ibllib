import unittest
import numpy as np
import brainbox.task.passive as passive
from brainbox.numerical import ismember2d


class TestPassive(unittest.TestCase):
    def setUp(self):
        """

        """
        # Simulate fake rfmap data
        test_frames = np.full((60, 15, 15), 128, dtype='uint8')
        # Test on and off individually
        test_frames[10:20, 8, 8] = 0
        test_frames[25:35, 10, 13] = 255
        # Test that interleaved are detected correctly
        test_frames[40:50, 4, 9] = 0
        test_frames[42:52, 6, 10] = 255
        test_frames[42:55, 11, 4] = 0
        test_frames[50:60, 8, 8] = 0

        test_times = np.arange(60)
        rf_map = {}
        rf_map['times'] = test_times
        rf_map['frames'] = test_frames

        rf_map_times, rf_map_pos, rf_stim_frames = passive.get_on_off_times_and_positions(rf_map)

        assert(all(rf_map_times == test_times))
        assert(rf_map_pos.shape == (15*15, 2))
        assert(len(rf_stim_frames['on']) == 15*15)
        assert(len(rf_stim_frames['off']) == 15 * 15)

        # Off is for the 0 ones
        assert(all(rf_stim_frames['off'][ismember2d(rf_map_pos, np.array([[8, 8]]))[0]][0][0]
                   == [10, 50]))
        assert(rf_stim_frames['off'][ismember2d(rf_map_pos, np.array([[4, 9]]))[0]][0][0] == 40)
        assert(rf_stim_frames['off'][ismember2d(rf_map_pos, np.array([[11, 4]]))[0]][0][0] == 42)

        # On is for the 255 ones
        assert(rf_stim_frames['on'][ismember2d(rf_map_pos, np.array([[10, 13]]))[0]][0][0] == 25)
        assert(rf_stim_frames['on'][ismember2d(rf_map_pos, np.array([[6, 10]]))[0]][0][0] == 42)


        # Next test that the firing rate function works
        # Basically just make one square responsive
        spike_times = np.arange(25, 35, 0.1)
        spike_depths = 500 * np.ones_like(spike_times)

        rara, depths = passive.get_rf_map_over_depth(rf_map_times, rf_map_pos, rf_stim_frames,spike_times, spike_depths)


        pickle_file = Path(__file__).parent.joinpath('fixtures', 'trials_test.pickle')
        if not pickle_file.exists():
            self.trial_data = None
        else:
            with open(pickle_file, 'rb') as f:
                self.trial_data = pickle.load(f)
        np.random.seed(0)
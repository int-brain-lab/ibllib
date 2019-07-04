from oneibl.one import ONE
import unittest


class Tests_REST(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        one = ONE(username='test_user', password='TapetesBloc18',
                  base_url='https://test.alyx.internationalbrainlab.org')
        self.one = one

    def test_water_restriction(self):
        """
        Examples of how to list all water restrictions and water-restriction for a given
        subject.
        """
        one = self.one
        # get all the water restrictions from start
        all_wr = one.alyx.rest('water-restriction', 'list')
        # 2 different ways to  get water restrictions for one subject
        wr_sub1 = one.alyx.rest('water-restriction', 'list', '?subject=algernon')
        wr_sub2 = one.alyx.rest('water-restriction', 'list', subject='algernon')  # recommended

        # enforce test logic
        self.assertTrue(set({'end_time', 'reference_weight', 'start_time', 'subject',
                             'water_type'}) >= set(all_wr[0].keys()))
        self.assertEqual(wr_sub1, wr_sub2)

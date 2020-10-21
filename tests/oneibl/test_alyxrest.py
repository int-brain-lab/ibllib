from oneibl.one import ONE
import unittest

one = ONE(username='test_user', password='TapetesBloc18',
          base_url='https://test.alyx.internationalbrainlab.org')


class Tests_REST(unittest.TestCase):

    def test_water_restriction(self):
        """
        Examples of how to list all water restrictions and water-restriction for a given
        subject.
        """
        # get all the water restrictions from start
        all_wr = one.alyx.rest('water-restriction', 'list')
        # 2 different ways to  get water restrictions for one subject
        wr_sub2 = one.alyx.rest('water-restriction', 'list', subject='algernon')  # recommended
        # enforce test logic
        self.assertTrue(set({'end_time', 'reference_weight', 'start_time', 'subject',
                             'water_type'}) >= set(all_wr[0].keys()))
        self.assertTrue(len(all_wr) > len(wr_sub2))

    def test_list_pk_query(self):
        """
        It's a bit stupid but the rest endpoint can't forward a direct query of the uuid via
        the pk keywork. The alyxclient has already an id parameter, which on the list method
        is used as a pk identifier. This special case is tested here
        :return:
        """
        ses = one.alyx.rest('sessions', 'list')[0]
        ses_ = one.alyx.rest('sessions', 'list', id=ses['url'][-36:])[0]
        self.assertEqual(ses, ses_)

from pathlib import Path
import unittest
import tempfile

import matplotlib.image
import numpy as np

from oneibl.one import ONE


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

    def test_note_with_picture_upload(self):
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        my_note = {'user': 'olivier',
                   'content_type': 'session',
                   'object_id': eid,
                   'text': "gnagnagna"}

        with tempfile.NamedTemporaryFile(mode="wb", suffix='.png') as png:
            matplotlib.image.imsave(png.name, np.random.random((500, 500)))
            files = {'image': open(Path(png.name), 'rb')}
            ar_note = one.alyx.rest('notes', 'create', data=my_note, files=files)

        self.assertTrue(len(ar_note['image']))
        self.assertTrue(ar_note['content_type'] == 'actions.session')
        one.alyx.rest('notes', 'delete', id=ar_note['id'])

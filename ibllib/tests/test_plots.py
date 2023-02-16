import unittest
import tempfile
import uuid

from pathlib import Path
from PIL import Image
from urllib.parse import urlparse
import datetime
import numpy as np

from one.api import ONE
from one.webclient import http_download_file

from ibllib.tests import TEST_DB
from ibllib.plots.snapshot import Snapshot
from ibllib.plots.figures import dlc_qc_plot

WIDTH, HEIGHT = 1000, 100


class TestSnapshot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Make a small image an store in tmp file
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.img_file = Path(cls.tmp_dir.name).joinpath('test.png')
        image = Image.new('RGBA', size=(WIDTH, HEIGHT), color=(155, 0, 0))
        image.save(cls.img_file, 'png')
        image.close()
        # set up ONE
        cls.one = ONE(**TEST_DB)
        # Collect all notes to delete them later
        cls.notes = []

        # make a new test session
        date = str(datetime.date(2018, np.random.randint(1, 12), np.random.randint(1, 28)))
        from one.registration import RegistrationClient
        _, eid = RegistrationClient(cls.one).create_new_session('ZM_1150', date=date)
        cls.eid = str(eid)

    def _get_image(self, url):
        # This is a bit of a hack because when running a the server locally, the request to the media folder fail
        rel_path = urlparse(url).path[1:]
        try:
            img_file = list(Path('/var/www/').rglob(rel_path))[0]
        except IndexError:
            img_file = http_download_file(url, target_dir=Path(self.tmp_dir.name), username=TEST_DB['username'],
                                          password=TEST_DB['password'], silent=True)
        return img_file

    def test_class_setup(self):
        # Tests that the creation of the class works and that the defaults are working
        object_id = str(uuid.uuid4())
        snp = Snapshot(object_id, one=self.one)
        self.assertEqual(snp.object_id, object_id)
        self.assertEqual(snp.content_type, 'session')  # default
        self.assertTrue(len(snp.images) == 0)
        # Test with different content type
        snp = Snapshot(object_id, content_type='probeinsertion', one=self.one)
        self.assertEqual(snp.object_id, object_id)
        self.assertEqual(snp.content_type, 'probeinsertion')
        self.assertTrue(len(snp.images) == 0)

    def test_snapshot_default(self):
        # Test a case where object_id and content type match
        object_id = self.one.alyx.rest('subjects', 'list', limit=1)[0]['id']
        snp = Snapshot(object_id, content_type='subject', one=self.one)
        with self.assertLogs('ibllib', 'INFO'):
            self.notes.append(snp.register_image(self.img_file, text='default size'))
        # Check image size is scaled to default width (defined in alyx settings.py)
        img_db = self._get_image(self.notes[-1]['image'])
        with Image.open(img_db) as im:
            self.assertEqual(im.size, (800, HEIGHT * 800 / WIDTH))
        # Test a case where they don't match
        snp = Snapshot(str(uuid.uuid4()), content_type='session', one=self.one)
        with self.assertLogs('ibllib', 'ERROR'):
            note = snp.register_image(self.img_file, text='default size')
            self.assertIsNone(note)

    def test_image_scaling(self):
        # make a new session
        object_id = self.eid
        snp = Snapshot(object_id, content_type='session', one=self.one)
        # Image in original size
        self.notes.append(snp.register_image(self.img_file, text='original size', width='orig'))
        img_db = self._get_image(self.notes[-1]['image'])
        with Image.open(img_db) as im:
            self.assertEqual(im.size, (WIDTH, HEIGHT))
        # Scale to width 100
        self.notes.append(snp.register_image(self.img_file, text='original size', width=100))
        img_db = self._get_image(self.notes[-1]['image'])
        with Image.open(img_db) as im:
            self.assertEqual(im.size, (100, HEIGHT * 100 / WIDTH))

    def test_register_multiple(self):
        expected_texts = ['first', 'second', 'third']
        expected_sizes = [(800, HEIGHT * 800 / WIDTH), (WIDTH, HEIGHT), (200, HEIGHT * 200 / WIDTH)]
        object_id = self.one.alyx.rest('datasets', 'list', limit=1)[0]['url'][-36:]
        snp = Snapshot(object_id, content_type='dataset', one=self.one)
        # Register multiple figures by giving a list
        self.notes.extend(snp.register_images([self.img_file, self.img_file, self.img_file],
                                              texts=['first', 'second', 'third'], widths=[None, 'orig', 200]))
        for i in range(3):
            self.assertEqual(self.notes[i - 3]['text'], expected_texts[i])
            img_db = self._get_image(self.notes[i - 3]['image'])
            with Image.open(img_db) as im:
                self.assertEqual(im.size, expected_sizes[i])
        # Registering multiple figures by adding to self.figures
        self.assertEqual(len(snp.images), 0)
        with self.assertLogs('ibllib', 'WARNING'):
            out = snp.register_images()
            self.assertIsNone(out)
        snp.images.extend([self.img_file, self.img_file, self.img_file])
        self.notes.extend(snp.register_images(texts=['always the same'], widths=[200]))
        for i in range(3):
            self.assertEqual(self.notes[i - 3]['text'], 'always the same')
            img_db = self._get_image(self.notes[i - 3]['image'])
            with Image.open(img_db) as im:
                self.assertEqual(im.size, expected_sizes[2])

    def test_generate_image(self):
        snp = Snapshot(str(uuid.uuid4()), one=self.one)

        def make_img(size, out_path):
            image = Image.new('RGBA', size=size, color=(100, 100, 100))
            image.save(out_path, 'png')
            return out_path

        out_path = Path(self.tmp_dir.name).joinpath('test_generate.png')
        snp.generate_image(make_img, {'size': (WIDTH, HEIGHT), 'out_path': out_path})
        self.assertEqual(len(snp.images), 1)
        self.assertEqual(snp.images[0], out_path)
        with Image.open(out_path) as im:
            self.assertEqual(im.size, (WIDTH, HEIGHT))

    @classmethod
    def tearDownClass(cls):
        # Clean up tmp dir
        cls.tmp_dir.cleanup()
        # Delete all notes
        for note in cls.notes:
            cls.one.alyx.rest('notes', 'delete', id=note['id'])
        # Delete the new session that was made
        cls.one.alyx.rest('sessions', 'delete', id=cls.eid)


class TestDlcQcPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.one = ONE(**TEST_DB)

    @classmethod
    def tearDownClass(cls):
        # Clean up tmp dir
        cls.tmp_dir.cleanup()

    def test_without_inputs(self):
        eid = '3473f9d2-aa5d-41a6-9048-c65d0b7ab97c'
        with self.assertRaises(AssertionError):
            dlc_qc_plot(self.one.eid2path(eid), self.one)
            # fig = dlc_qc_plot(self.one.eid2path(eid), self.one)
        # fig_path = (Path(self.tmp_dir.name).joinpath('dlc_qc_plot.png'))
        # fig.savefig(fig_path)
        # with Image.open(fig_path) as im:
        #     self.assertEqual(im.size, (1700, 1000))

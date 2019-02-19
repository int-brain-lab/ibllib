# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, February 19th 2019, 11:45:24 am
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 19-02-2019 11:46:07.077
import os
import shutil
import unittest
from pathlib import Path

from alf.one_iblrig import compress_video, create, extract, register
from alf.transfer_rig_data import main as transfer
from oneibl.one import ONE


class TestFlagOperations(unittest.TestCase):

    def setUp(self):
        # Set ONE to use the test database
        self.one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
                       username='test_user', password='TapetesBloc18')
        # Create fresh test database
        #
        # Folders
        self.init_folder = Path('/mnt/s0/IntegrationTests/Subjects_init')
        self.sessions = [x.parent for x in self.init_folder.rglob(
            'create_me.flag')]
        self.rig_folder = self.init_folder.parent / 'RigSubjects'
        self.server_folder = self.init_folder.parent / 'ServerSubjects'
        # Init rig_folder
        shutil.copytree(self.init_folder, self.rig_folder)

    def _create(self):
        create(self.rig_folder, one=self.one)
        # Check for deletion of create_me.flag
        cflags = list(self.server_folder.rglob('create_me.flag'))
        self.assertTrue(cflags == [])

    def _transfer(self):
        transfer(self.rig_folder, self.server_folder)

        # Check for deletion of transfer_me.flag
        tflags = list(self.server_folder.rglob('transfer_me.flag'))
        self.assertTrue(tflags == [])
        # Check creation of extract_me.flag and compress_video.flag in folders
        eflags = list(self.server_folder.rglob('extract_me.flag'))
        cflags = list(self.server_folder.rglob('compress_video.flag'))
        self.assertTrue(eflags != [])
        self.assertTrue(cflags != [])
        # for all sessions OK?
        self.assertTrue(
            all([x.parent == y for x, y in zip(eflags, self.sessions)]))
        self.assertTrue(
            all([x.parent == y for x, y in zip(cflags, self.sessions)]))

    def _extraction(self):
        extract(self.server_folder)

        # Check for deletion of extract_me.flag
        eflags = list(self.server_folder.rglob('extract_me.flag'))
        self.assertTrue(eflags == [])
        # Check for creation of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags != [])
        # Check for flag in all sessions
        self.assertTrue(
            all([x.parent == y for x, y in zip(rflags, self.sessions)]))

    def _data_qa(self):
        pass

    def _registration(self):
        register(self.server_folder, one=self.one)

        # Check for deletion of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags == [])

    def _compression(self):
        compress_video(self.server_folder)

        # Check for deletion of compress_video.flag
        cflags = list(self.server_folder.rglob('compress_video.flag'))
        self.assertTrue(cflags == [])
        # Check for creation of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags != [])
        # Check for flag in all sessions
        self.assertTrue(
            all([x.parent == y for x, y in zip(rflags, self.sessions)]))

    def test_all(self):
        self._create()
        self._transfer()
        self._extraction()
        self._data_qa()
        self._registration()
        self._compression()

    def tearDown(self):
        shutil.rmtree(self.rig_folder, ignore_errors=True)
        shutil.rmtree(self.server_folder, ignore_errors=True)
        os.system("ssh -i ~/.ssh/alyx.internationalbrainlab.org.pem ubuntu@test.alyx.internationalbrainlab.org './02_rebuild_from_cache.sh'")  # noqa


if __name__ == "__main__":
    unittest.main(exit=False)

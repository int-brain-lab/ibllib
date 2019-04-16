# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, February 19th 2019, 11:45:24 am
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 19-02-2019 11:46:07.077
import shutil
import unittest
from pathlib import Path

import alf.one_iblrig as iblrig_pipeline
from alf.transfer_rig_data import main as transfer
from oneibl.one import ONE


class TestFlagOperations(unittest.TestCase):

    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/Subjects_init')
        if not self.init_folder.exists():
            return
        # Set ONE to use the test database
        self.one = ONE(base_url='https://test.alyx.internationalbrainlab.org',  # testdev
                       username='test_user', password='TapetesBloc18')

        self.sessions = [x.parent for x in self.init_folder.rglob(
            'create_me.flag')]
        self.rig_folder = self.init_folder.parent / 'RigSubjects'
        self.server_folder = self.init_folder.parent / 'ServerSubjects'
        self.vidfiles = list(self.init_folder.rglob('*.avi'))
        # Init rig_folder
        if self.rig_folder.exists():
            shutil.rmtree(self.rig_folder)
        shutil.copytree(self.init_folder, self.rig_folder)

    def _create(self):
        iblrig_pipeline.create(self.rig_folder, one=self.one)
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
        self.assertTrue(len(self.sessions) == len(eflags))
        self.assertTrue(len(cflags) == len(self.vidfiles))

    def _extraction(self):
        iblrig_pipeline.extract(self.server_folder)

        # Check for deletion of extract_me.flag
        eflags = list(self.server_folder.rglob('extract_me.flag'))
        self.assertTrue(eflags == [])
        # Check for creation of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags != [])
        # Check for flag in all sessions
        self.assertTrue(len(rflags) == len(self.sessions))

    def _data_qa(self):
        pass

    def _registration(self):
        iblrig_pipeline.register(self.server_folder, one=self.one)

        # Check for deletion of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags == [])

    def _compression(self):
        iblrig_pipeline.compress_video(self.server_folder)

        # Check for deletion of compress_video.flag
        cflags = list(self.server_folder.rglob('compress_video.flag'))
        self.assertTrue(cflags == [])
        # Check for creation of register_me.flag
        rflags = list(self.server_folder.rglob('register_me.flag'))
        self.assertTrue(rflags != [])
        # Check for flag in all sessions
        self.assertTrue(len(rflags) == len(self.vidfiles))

    def test_all(self):
        if not self.init_folder.exists():
            return
        self._create()
        self._transfer()
        self._extraction()
        self._data_qa()
        self._registration()
        self._compression()
        self._registration()

    def tearDown(self):
        if not self.init_folder.exists():
            return
        shutil.rmtree(self.rig_folder, ignore_errors=True)
        shutil.rmtree(self.server_folder, ignore_errors=True)
        # os.system("ssh -i ~/.ssh/alyx.internationalbrainlab.org.pem ubuntu@test.alyx.internationalbrainlab.org './02_rebuild_from_cache.sh'")  # noqa


if __name__ == "__main__":
    unittest.main(exit=False)

import unittest
from unittest import mock
import json
import random
import string
import datetime
import numpy as np

import requests
from one.api import ONE

from ibllib.tests import TEST_DB
import ibllib.qc.critical_reasons as usrpmt
from one.registration import RegistrationClient

one = ONE(**TEST_DB)


def mock_input(prompt):
    if "Select from this list the reason(s)" in prompt:
        return "1,3"
    elif "Explain why you selected" in prompt:
        return "estoy un poco preocupada"
    elif "You are about to delete" in prompt:
        return "y"


class TestUserPmtSess(unittest.TestCase):

    def setUp(self) -> None:
        # Make sure tests use correct session ID
        one.alyx.clear_rest_cache()
        # Create new session on database with a random date to avoid race conditions
        date = str(datetime.date(2022, np.random.randint(1, 12), np.random.randint(1, 28)))
        from one.registration import RegistrationClient
        _, eid = RegistrationClient(one).create_new_session('ZM_1150', date=date)
        eid = str(eid)
        # Currently the task protocol of a session must contain 'ephys' in order to create an insertion!
        one.alyx.rest('sessions', 'partial_update', id=eid, data={'task_protocol': 'ephys'})
        self.sess_id = eid

        # Make new insertion with random name
        data = {'name': ''.join(random.choices(string.ascii_letters, k=5)),
                'session': self.sess_id,
                'model': '3A',
                'json': None,
                'datasets': []}
        one.alyx.rest('insertions', 'create', data=data)
        # 3. Save ins id in global variable for test access
        self.ins_id = one.alyx.rest('insertions', 'list', session=self.sess_id, name=data['name'], no_cache=True)[0]['id']

    def test_userinput_sess(self):
        eid = self.sess_id  # sess id
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}', no_cache=True)
        critical_dict = json.loads(note[0]['text'])
        print(critical_dict)
        expected_dict = {
            'title': '=== EXPERIMENTER REASON(S) FOR MARKING THE SESSION AS CRITICAL ===',
            'reasons_selected': ['synching impossible', 'essential dataset missing'],
            'reason_for_other': []}
        assert expected_dict == critical_dict

    def test_userinput_ins(self):
        eid = self.ins_id  # probe id
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}', no_cache=True)
        critical_dict = json.loads(note[0]['text'])
        expected_dict = {
            'title': '=== EXPERIMENTER REASON(S) FOR MARKING THE INSERTION AS CRITICAL ===',
            'reasons_selected': ['Track not visible on imaging data', 'Drift'],
            'reason_for_other': []}
        assert expected_dict == critical_dict

    def test_note_already_existing(self):
        eid = self.sess_id  # sess id
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}', no_cache=True)
        original_note_id = note[0]['id']

        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid, one=one)

        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}', no_cache=True)
        assert len(note) == 1
        assert original_note_id != note[0]['id']

    def test_guiinput_ins(self):
        eid = self.ins_id  # probe id
        str_notes_static = '=== EXPERIMENTER REASON(S) FOR MARKING THE INSERTION AS CRITICAL ==='
        notes = one.alyx.rest('notes', 'list',
                              django=f'text__icontains,{str_notes_static},object_id,{eid}',
                              no_cache=True)
        # delete any previous notes
        for note in notes:
            one.alyx.rest('notes', 'delete', id=note['id'])

        # write a new note and make sure it is found
        usrpmt.main_gui(eid, reasons_selected=['Drift'], one=one)
        note = one.alyx.rest('notes', 'list',
                             django=f'text__icontains,{str_notes_static},object_id,{eid}',
                             no_cache=True)
        assert len(note) == 1
        critical_dict = json.loads(note[0]['text'])
        expected_dict = {
            'title': '=== EXPERIMENTER REASON(S) FOR MARKING THE INSERTION AS CRITICAL ===',
            'reasons_selected': ['Drift'], 'reason_for_other': []}
        assert expected_dict == critical_dict

    def test_note_probe_ins(self):
        # Note: this test is redundant with the above, but it tests specifically whether
        # the nomenclature of writing notes in insertion is correct.
        eid = self.ins_id  # probe id
        content_type = 'probeinsertion'
        note_text = 'USING A FAKE SINGLE STRING HERE KSROI283IF982HKJFHWRY'

        notes = one.alyx.rest('notes', 'list',
                              django=f'text__icontains,{note_text},object_id,{eid}',
                              no_cache=True)
        # delete any previous notes
        for note in notes:
            one.alyx.rest('notes', 'delete', id=note['id'])

        # create new note
        my_note = {'user': one.alyx.user,
                   'content_type': content_type,
                   'object_id': eid,
                   'text': f'{note_text}'}

        one.alyx.rest('notes', 'create', data=my_note)

        notes = one.alyx.rest('notes', 'list',
                              django=f'text__icontains,{note_text},object_id,{eid}',
                              no_cache=True)
        assert len(notes) == 1

    def tearDown(self) -> None:
        try:
            one.alyx.rest('insertions', 'delete', id=self.ins_id)
        except requests.HTTPError as ex:
            if ex.errno != 404:
                raise ex
        text = '"title": "=== EXPERIMENTER REASON(S)'
        notes = one.alyx.rest('notes', 'list', django=f'text__icontains,{text}', no_cache=True)
        for n in notes:
            one.alyx.rest('notes', 'delete', n['id'])
        text = 'USING A FAKE SINGLE STRING HERE KSROI283IF982HKJFHWRY'
        notes = one.alyx.rest('notes', 'list', django=f'text__icontains,{text}', no_cache=True)
        for n in notes:
            one.alyx.rest('notes', 'delete', n['id'])

        note = one.alyx.rest('notes', 'list', django=f'object_id,{self.sess_id}', no_cache=True)
        for no in note:
            one.alyx.rest('notes', 'delete', id=no['id'])


class TestSignOffNote(unittest.TestCase):
    def setUp(self) -> None:
        path, eid = RegistrationClient(one).create_new_session('ZM_1743')
        self.eid = str(eid)
        self.sign_off_keys = ['biasedChoiceWorld_00', 'passiveChoiceWorld_01']
        data = {'sign_off_checklist': dict.fromkeys(map(lambda x: f'{x}', self.sign_off_keys)),
                'lala': 'blabla',
                'fafa': 'gaga'}
        one.alyx.json_field_update("sessions", self.eid, data=data)

    def test_sign_off(self):
        sess = one.alyx.rest('sessions', 'read', id=self.eid, no_cache=True)

        note = usrpmt.TaskSignOffNote(self.eid, one, sign_off_key=self.sign_off_keys[0])
        note.sign_off()

        sess = one.alyx.rest('sessions', 'read', id=self.eid, no_cache=True)
        assert sess['json']['sign_off_checklist'][self.sign_off_keys[0]]['date'] == note.datetime_key.split('_')[0]
        assert sess['json']['sign_off_checklist'][self.sign_off_keys[0]]['user'] == note.datetime_key.split('_')[1]
        # Make sure other json fields haven't been changed
        assert sess['json']['lala'] == 'blabla'
        assert sess['json']['fafa'] == 'gaga'

    def test_upload_note_prompt(self):
        with mock.patch('builtins.input', mock_input):
            note = usrpmt.TaskSignOffNote(self.eid, one, sign_off_key=self.sign_off_keys[0])
            note.upload_note()

        notes = one.alyx.rest('notes', 'list', django=f'text__icontains,{note.note_title},object_id,{self.eid}',
                              no_cache=True)

        assert len(notes) == 1
        note_dict = json.loads(notes[0]['text'])
        expected_dict = {
            'title': f'{note.note_title}',
            f'{note.datetime_key}': {'reasons_selected': ['wheel data corrupt', 'Other'],
                                     'reason_for_other': "estoy un poco preocupada"}
        }
        assert expected_dict == note_dict

        sess = one.alyx.rest('sessions', 'read', id=self.eid, no_cache=True)
        print(sess['json'])
        assert sess['json']['sign_off_checklist'][self.sign_off_keys[0]]['date'] == note.datetime_key.split('_')[0]
        assert sess['json']['sign_off_checklist'][self.sign_off_keys[0]]['user'] == note.datetime_key.split('_')[1]

    def test_upload_existing_note(self):
        # Make first note
        with mock.patch('builtins.input', mock_input):
            note = usrpmt.TaskSignOffNote(self.eid, one, sign_off_key=self.sign_off_keys[0])
            note.datetime_key = '2022-11-10_user'
            note.upload_note()

        # Make note again
        note = usrpmt.TaskSignOffNote(self.eid, one, sign_off_key=self.sign_off_keys[0])
        note.upload_note(nums='0')

        notes = one.alyx.rest('notes', 'list', django=f'text__icontains,{note.note_title},object_id,{self.eid}',
                              no_cache=True)
        assert len(notes) == 1
        note_dict = json.loads(notes[0]['text'])

        expected_dict = {
            'title': f'{note.note_title}',
            '2022-11-10_user': {'reasons_selected': ['wheel data corrupt', 'Other'],
                                'reason_for_other': "estoy un poco preocupada"},
            f'{note.datetime_key}': {'reasons_selected': ['raw trial data does not exist'],
                                     'reason_for_other': []}
        }

        assert expected_dict == note_dict

    def tearDown(self) -> None:
        one.alyx.rest('sessions', 'delete', id=self.eid)


if __name__ == '__main__':
    unittest.main()

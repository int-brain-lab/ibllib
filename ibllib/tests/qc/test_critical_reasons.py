import unittest
from unittest import mock
import json
import random
import string

import requests
from one.api import ONE

from ibllib.tests import TEST_DB
import ibllib.qc.critical_reasons as usrpmt

one = ONE(**TEST_DB)


def mock_input(prompt):
    if "Select from this list the reason(s)" in prompt:
        return "1,3"
    elif "Explain why you selected" in prompt:
        return "Estoy un poco preocupada"
    elif "You are about to delete" in prompt:
        return "y"


class TestUserPmtSess(unittest.TestCase):

    def setUp(self) -> None:
        # Make sure tests use correct session ID
        one.alyx.clear_rest_cache()
        self.sess_id = one.alyx.rest('sessions', 'list', task_protocol='ephys')[0]['url'][-36:]

        # Make new insertion with random name
        data = {'name': ''.join(random.choices(string.ascii_letters, k=5)),
                'session': self.sess_id,
                'model': '3A',
                'json': None,
                'datasets': []}
        one.alyx.rest('insertions', 'create', data=data)
        # 3. Save ins id in global variable for test access
        self.ins_id = one.alyx.rest('insertions', 'list', session=self.sess_id, name=data['name'], no_cache=True)[0]['id']

    def test_reason_addnumberstr(self):
        outstr = usrpmt._reason_addnumberstr(reason_list=['a', 'b'])
        self.assertEqual(outstr, ['0) a', '1) b'])

    def test_userinput_sess(self):
        eid = self.sess_id  # sess id
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid=eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}', no_cache=True)
        critical_dict = json.loads(note[0]['text'])
        expected_dict = {
            'title': '=== EXPERIMENTER REASON(S) FOR MARKING THE SESSION AS CRITICAL ===',
            'reasons_selected': ['synching impossible', 'essential dataset missing'],
            'reason_for_other': []}
        assert expected_dict == critical_dict

    def test_userinput_ins(self):
        eid = self.ins_id  # probe id
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid=eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}', no_cache=True)
        critical_dict = json.loads(note[0]['text'])
        expected_dict = {
            'title': '=== EXPERIMENTER REASON(S) FOR MARKING THE INSERTION AS CRITICAL ===',
            'reasons_selected': ['Track not visible on imaging data', 'Drift'],
            'reason_for_other': []}
        assert expected_dict == critical_dict

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
        usrpmt.main_gui(eid=eid, reasons_selected=['Drift'], one=one)
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


if __name__ == '__main__':
    unittest.main()

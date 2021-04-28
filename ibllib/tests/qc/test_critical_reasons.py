import unittest
from unittest import mock
import json
from oneibl.one import ONE

import ibllib.qc.critical_reasons as usrpmt

one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
          username='test_user',
          password='TapetesBloc18')


def mock_input(prompt):
    if "Select from this list the reason(s)" in prompt:
        return "1,3"
    elif "Explain why you selected" in prompt:
        return "Estoy un poco preocupada"
    elif "You are about to delete" in prompt:
        return "y"


class TestUserPmtSess(unittest.TestCase):
    def test_reason_addnumberstr(self):
        outstr = usrpmt._reason_addnumberstr(reason_list=['a', 'b'])
        self.assertEqual(outstr, ['0) a', '1) b'])

    def test_userinput_sess(self):
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'  # sess id
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid=eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}')
        critical_dict = json.loads(note[0]['text'])
        expected_dict = {
            'title': '=== EXPERIMENTER REASON(S) FOR MARKING THE SESSION AS CRITICAL ===',
            'reasons_selected': ['synching impossible', 'essential dataset missing'],
            'reason_for_other': []}
        assert expected_dict == critical_dict

    def test_userinput_ins(self):
        eid = '440d02a4-b6dc-4de0-b487-ed64f7a59375'  # probe id
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid=eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}')
        critical_dict = json.loads(note[0]['text'])
        expected_dict = {
            'title': '=== EXPERIMENTER REASON(S) FOR MARKING THE INSERTION AS CRITICAL ===',
            'reasons_selected': ['Track not visible on imaging data', 'Drift'],
            'reason_for_other': []}
        assert expected_dict == critical_dict

    def test_note_probe_ins(self):
        eid = '440d02a4-b6dc-4de0-b487-ed64f7a59375'  # probe id
        content_type = 'probeinsertion'
        note_text = 'USING A FAKE SINGLE STRING HERE KSROI283IF982HKJFHWRY'

        my_note = {'user': one._par.ALYX_LOGIN,
                   'content_type': content_type,
                   'object_id': eid,
                   'text': f'{note_text}'}

        one.alyx.rest('notes', 'create', data=my_note)

        notes = one.alyx.rest('notes', 'list',
                              django=f'text__icontains,{note_text},'
                                     f'object_id,{eid}')
        assert len(notes) == 1


if __name__ == '__main__':
    unittest.main()

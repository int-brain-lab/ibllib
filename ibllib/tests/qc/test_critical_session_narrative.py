import unittest
from unittest import mock
import json
from oneibl.one import ONE

import ibllib.qc.critical_session_narrative as usrpmt

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

    def test_userinput(self):
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        with mock.patch('builtins.input', mock_input):
            usrpmt.main(eid=eid, one=one)
        note = one.alyx.rest('notes', 'list', django=f'object_id,{eid}')
        critical_dict = json.loads(note[0]['text'])
        expected_dict = {
            'title': usrpmt.STR_NOTES_STATIC,
            'reasons_selected': ['synching impossible', 'essential dataset missing'],
            'reason_for_other': []}
        assert expected_dict == critical_dict


if __name__ == '__main__':
    unittest.main()

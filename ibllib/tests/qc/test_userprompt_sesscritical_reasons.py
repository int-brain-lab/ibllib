import unittest
import ibllib.qc.userprompt_sesscritical_reasons as usrpmt
from oneibl.one import ONE


# def fake_input(prompt=None, responses=None):
#     if responses:
#         # Store responses list and reset counter
#         fake_input.responses = responses
#         fake_input.count = 0
#     else:  # Return stored response
#         index = min(fake_input.count, len(fake_input.responses) - 1)
#         fake_input.count += 1  # Increase count
#         return fake_input.responses[index]


class TestUserPmtSess(unittest.TestCase):
    def test_reason_addnumberstr(self):
        outstr = usrpmt.reason_addnumberstr(reason_list=['a', 'b'])
        self.assertEqual(outstr, ['0) a', '1) b'])

    # def test_userinput(self):
    #     # Add my responses for test
    #     usrpmt.input = fake_input(responses=['1,3', 'n'])
    #     # todo use test DB
    #     usrpmt.main_function(eid='2ffd3ed5-477e-4153-9af7-7fdad3c6946b',
    #                          one=ONE(base_url='https://dev.alyx.internationalbrainlab.org'))


if __name__ == '__main__':
    unittest.main()

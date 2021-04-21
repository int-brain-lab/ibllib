import unittest
import ibllib.qc.userprompt_sesscritical_reasons as usrpmt


class TestUserPmtSess(unittest.TestCase):
    def test_reason_addnumberstr(self):
        outstr = usrpmt.reason_addnumberstr(reason_list=['a', 'b'])
        self.assertEqual(outstr, ['0) a', '1) b'])


if __name__ == '__main__':
    unittest.main()

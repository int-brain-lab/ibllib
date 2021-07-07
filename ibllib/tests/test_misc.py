import unittest
import logging
import time

from ibllib.misc import (version, print_progress)


class TestPrintProgress(unittest.TestCase):

    def test_simple_print(self):
        print('waitbar')
        for p in range(10):
            time.sleep(0.05)
            print_progress(p, 9)


class TestVersionTags(unittest.TestCase):

    def test_compare_version_tags(self):
        self.assert_eq('3.2.3', '3.2.3')
        self.assert_eq('3.2.3', '3.2.03')
        self.assert_g('3.2.3', '3.2.1')
        self.assert_l('3.2.1', '3.2.3')
        self.assert_g('3.2.11', '3.2.2')
        self.assert_l('3.2.1', '3.2.11')

    def assert_eq(self, v0, v_):
        self.assertTrue(version.eq(v0, v_))
        self.assertTrue(version.ge(v0, v_))
        self.assertTrue(version.le(v0, v_))
        self.assertFalse(version.gt(v0, v_))
        self.assertFalse(version.lt(v0, v_))

    def assert_l(self, v0, v_):
        self.assertFalse(version.eq(v0, v_))
        self.assertFalse(version.ge(v0, v_))
        self.assertTrue(version.le(v0, v_))
        self.assertFalse(version.gt(v0, v_))
        self.assertTrue(version.lt(v0, v_))

    def assert_g(self, v0, v_):
        self.assertFalse(version.eq(v0, v_))
        self.assertTrue(version.ge(v0, v_))
        self.assertFalse(version.le(v0, v_))
        self.assertTrue(version.gt(v0, v_))
        self.assertFalse(version.lt(v0, v_))


class TestLoggingSystem(unittest.TestCase):

    def test_levels(self):
        # logger = logger_config('ibllib')
        logger = logging.getLogger('ibllib')
        logger.critical('IBLLIB This is a critical message')
        logger.error('IBLLIB This is an error message')
        logger.warning('IBLLIB This is a warning message')
        logger.info('IBLLIB This is an info message')
        logger.debug('IBLLIB This is a debug message')
        logger = logging.getLogger()
        logger.critical('ROOT This is a critical message')
        logger.error('ROOT This is an error message')
        logger.warning('ROOT This is a warning message')
        logger.info('ROOT This is an info message')
        logger.debug('ROOT This is a debug message')


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

import unittest
import logging
from pathlib import Path

from ibllib.misc import version, log_to_file


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
    log_name = '_foobar'

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

    def test_file_handler(self):
        """Test for ibllib.misc.log_to_file"""
        log_path = Path.home().joinpath('.ibl_logs', self.log_name)
        log_path.unlink(missing_ok=True)
        test_log = log_to_file(self.log_name, log=self.log_name)
        test_log.info('foobar')

        # Should have created a log file and written to it
        self.assertTrue(log_path.exists())
        with open(log_path, 'r') as f:
            logged = f.read()
        self.assertIn('foobar', logged)

    def tearDown(self) -> None:
        # Before we can delete the test log file we must close the file handler
        test_log = logging.getLogger(self.log_name)
        for handler in test_log.handlers:
            handler.close()
            test_log.removeHandler(handler)
        Path.home().joinpath('.ibl_logs', self.log_name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

import unittest
import datetime


class TestRefactoringSpikeGlxDsp(unittest.TestCase):

    def test_old_imports(self):
        """
        Test old imports for compatibility
        This function will break on purpose at a given date
        :return:
        """
        from ibllib.io.spikeglx import read_meta_data  # noqa
        import ibllib.dsp as dsp
        from ibllib.dsp import smooth  # noqa
        from ibllib.dsp.utils import parabolic_max  # noqa
        from ibllib.ephys.neuropixel import trace_header  # noqa

        assert dsp.fourier.__file__ is not None

        if datetime.datetime.now() > datetime.datetime(2022, 11, 18):
            raise NotImplementedError
        """
        When this happens it means the deprecation period has elapsed for the ugly imports above
        At this date take the following simple steps:
        -   delete all import wrappers:
            -   delete the ibllib.io.spikeglx module
            -   delete the ibllib.ephys.neuropixel module
            -   delete the ibllib.dsp folder
        -   delete this test
        Thank you, you're good to go !
        """

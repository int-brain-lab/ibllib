import tempfile
import unittest
import logging

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader


_logger = logging.getLogger('ibllib')
_logger.setLevel(10)

ba = AllenAtlas()
one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    silent=True,
    password='international'
)


class TestReadSpikeSorting(unittest.TestCase):

    def test_spike_sorting_loader(self):
        # insertions = one.alyx.rest('insertions', 'list')
        pid = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd'
        ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = ssl.load_spike_sorting()
        SpikeSortingLoader.merge_clusters(spikes, clusters, channels)

        self.assertEqual({'depths', 'clusters', 'amps', 'times'}, set(spikes.keys()))
        expected = 'alf/probe00/pykilosort/#2024-05-06#'
        self.assertEqual(ssl.spike_sorting_path.relative_to(ssl.session_path).as_posix(), expected)
        self.assertTrue(expected.startswith(ssl.collection))
        self.assertEqual('alf', ssl.histology)
        self.assertEqual(2, len(ssl.collections))


class TestStreamData(unittest.TestCase):

    def test_streamer_object_with_spike_sorting_loader(self):
        pid = '675952a4-e8b3-4e82-a179-cc970d5a8b01'
        t0 = 50
        self.td = tempfile.TemporaryDirectory()
        tmp_one = ONE(
            base_url='https://openalyx.internationalbrainlab.org',
            password='international',
            silent=True,
            cache_dir=self.td.name)

        sl = SpikeSortingLoader(pid=pid, one=tmp_one)
        sr = sl.raw_electrophysiology(stream=True)
        # read once to download the data
        raw_ = sr[int(t0 * 2500):int((t0 + 1) * 2500), :]  # noqa
        # second read to use the local cache
        raw_ = sr[int(t0 * 2500):int((t0 + 1) * 2500), :]
        sl_raw = sr[int(t0 * 2500):int((t0 + 1) * 2500), :-1]
        assert sl_raw.shape == (2500, 384)
        assert sr.nc == 385
        assert sr.nsync == 1
        assert sr.rl == 6085.7024
        assert (raw_.shape == (2500, 385))
        assert sr.target_dir.exists()
        assert sr.geometry.keys()

    def tearDown(self) -> None:
        self.td.cleanup()

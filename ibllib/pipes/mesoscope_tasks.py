"""The mesoscope data extraction pipeline.

The mesoscope pipeline currently comprises raw image registration and timestamps extraction.  In
the future there will be compression (and potential cropping), FOV metadata extraction, and ROI
extraction.

Pipeline:
    1. Data renamed to be ALF-compliant and registered
"""
import logging
import subprocess
from pathlib import Path
from itertools import chain

from ibllib.io.extractors.mesoscope import TimelineTrials
from ibllib.pipes import base_tasks

_logger = logging.getLogger(__name__)


class MesoscopeRegisterSnapshots(base_tasks.RegisterRawDataTask):

    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.tif', f'{self.device_collection}/snapshots', False),
                            ('*.png', f'{self.device_collection}/snapshots', False),
                            ('*.jpg', f'{self.device_collection}/snapshots', False)],
            'output_files': []
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self):
        out_files = super()._run()
        self.register_snapshots()
        return out_files


class MesoscopeCompress(base_tasks.DynamicTask):

    priority = 90
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.tif', self.device_collection, True)],
            'output_files': [('imaging.frames.tar.bz', self.device_collection, True)]
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self, remove_uncompressed=False, verify_output=True, **kwargs):
        in_dir = self.session_path.joinpath(self.device_collection or '')
        outfile = self.session_path.joinpath(*filter(None, reversed(self.output_files[0][:2])))
        infiles = list(chain(*map(lambda x: in_dir.glob(x[0]), self.input_files)))  # glob for all input patterns
        if not infiles:
            _logger.info('No image files found; returning')
            return []

        uncompressed_size = sum(x.stat().st_size for x in infiles)
        _logger.info('Compressing %i files', len(infiles))
        cmd = 'tar -I lbzip2 -cvf "{output}" "{input}"'.format(
            output=outfile.relative_to(in_dir), input='" "'.join(str(x.relative_to(in_dir)) for x in infiles))
        _logger.debug(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
        info, error = process.communicate()
        assert process.returncode == 0, f'compression failed: {error}'

        # Check the output
        assert outfile.exists(), 'output file missing'
        compressed_size = outfile.stat().st_size
        _logger.info('Compression ratio = %.3f, saving %.2f pct (%.2f MB)',
                     uncompressed_size / compressed_size,
                     round((1 - (compressed_size / uncompressed_size)) * 10000) / 100,
                     (uncompressed_size - compressed_size) / 1024 / 1024)

        if verify_output:
            cmd = f'tar -tzf "{outfile.relative_to(in_dir)}"'
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
            info, error = process.communicate()
            assert process.returncode == 0, f'failed to read compressed file: {error}'
            # TODO Assert number of files in tar match input files

        if remove_uncompressed:
            for file in infiles:
                file.unlink()

        return [outfile]


#  level 1
class MesoscopePreprocess(base_tasks.DynamicTask):

    priority = 80
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('imaging.frames.*', self.device_collection, True),
                            ('mesoscopeEvents.raw.*', self.device_collection, True)],
            'output_files': [('mesoscopeChannels.frameAverage.npy', 'alf/mesoscope', True),
                             ('mesoscopeU.images.npy', 'alf/mesoscope', True),
                             ('mesoscopeSVT.uncorrected.npy', 'alf/mesoscope', True),
                             ('mesoscopeSVT.haemoCorrected.npy', 'alf/mesoscope', True)]
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self, **kwargs):
        self.wf = base_tasks.DynamicTask(self.session_path)
        _, out_files = self.wf.extract(save=True, extract_timestamps=False)
        return out_files

    def tearDown(self):
        super(MesoscopePreprocess, self).tearDown()
        self.wf.remove_files()


class MesoscopeSync(base_tasks.DynamicTask):

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('imaging.raw.mov', self.device_collection, True),
                            ('mesoscopeEvents.raw.camlog', self.device_collection, True),
                            (f'_{self.sync_namespace}_sync.channels.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.polarities.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.times.npy', self.sync_collection, True)],
            'output_files': [('imaging.times.npy', 'alf/mesoscope', True),
                             ('imaging.imagingLightSource.npy', 'alf/mesoscope', True),
                             ('imagingLightSource.properties.htsv', 'alf/mesoscope', True)]
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self):

        self.wf = base_tasks.DynamicTask(self.session_path)
        save_paths = [self.session_path.joinpath(sig[1], sig[0]) for sig in self.signature['output_files']]
        out_files = self.wf.sync_timestamps(bin_exists=False, save=True, save_paths=save_paths)

        # TODO QC

        return out_files


class MesoscopeFOV(base_tasks.DynamicTask):

    priority = 40
    job_size = 'small'

    signature = {
        'input_files': [('mesoscopeLandmarks.dorsalCortex.json', 'alf', True),
                        ('mesoscopeSVT.uncorrected.npy', 'alf', True),
                        ('mesoscopeSVT.haemoCorrected.npy', 'alf', True)],
        'output_files': []
    }

    def _run(self):
        # TODO make task that computes location

        return []

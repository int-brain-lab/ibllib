import logging
import numpy as np

from ibllib.io.extractors import biased_trials
from ibllib.io.extractors.base import BaseBpodTrialsExtractor

_logger = logging.getLogger('ibllib')


class LaserBool(BaseBpodTrialsExtractor):
    """
    Extracts the laser probabilities from the bpod jsonable
    """
    save_names = ('_ibl_trials.laser_stimulation.npy', '_ibl_trials.laser_probability.npy')
    var_names = ('laser_stimulation', 'laser_probability')

    def _extract(self, **kwargs):
        lstim = np.array([float(t.get('laser_stimulation', np.NaN)) for t in self.bpod_trials])
        lprob = np.array([float(t.get('laser_probability', np.NaN)) for t in self.bpod_trials])
        _logger.info('Extracting laser datasets')
        if np.all(np.isnan(lprob)):
            # this prevents the file from being saved when no data
            self.save_names = ('_ibl_trials.laser_stimulation.npy', None)
            _logger.warning('No laser probability found in bpod data')
        if np.all(np.isnan(lstim)):
            # this prevents the file from being saved when no data
            self.save_names = (None, '_ibl_trials.laser_probability.npy')
            _logger.warning('No laser stimulation found in bpod data')
        return lstim, lprob


def extract_all(*args, extra_classes=None, **kwargs):
    """
    Extracts the biased trials for a training session
    """
    if extra_classes is not None:
        extra_classes.append(LaserBool)
    else:
        extra_classes = [LaserBool]
    return biased_trials.extract_all(*args, **kwargs, extra_classes=extra_classes)

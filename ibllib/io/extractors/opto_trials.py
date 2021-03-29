import logging
import numpy as np

from ibllib.io.extractors import biased_trials
from ibllib.io.extractors.base import BaseBpodTrialsExtractor

_logger = logging.getLogger('ibllib')


class LaserBool(BaseBpodTrialsExtractor):
    save_names = ['_ibl_trials.laser_stimulation.npy', '_ibl_trials.laser_probability.npy']
    var_names = ['laser_stimulation', 'laser_probability']

    def _extract(self):
        lstim = np.array([np.float(t.get('laser_stimulation', np.NaN)) for t in self.bpod_trials])
        lprob = np.array([np.float(t.get('laser_probability', np.NaN)) for t in self.bpod_trials])
        _logger.info('Extracting laser datasets')
        if np.all(np.isnan(lprob)):
            self.save_names[1] = None  # this prevents the file from being saved when no data
            _logger.info('No laser probability found in bpod data')
        if np.all(np.isnan(lstim)):
            self.save_names[0] = None  # this prevents the file from being saved when no data
            _logger.info('No laser stimulation found in bpod data')
        return lstim, lprob


def extract_all(*args, extra_classes=None, **kwargs):
    if extra_classes is not None:
        extra_classes.append(LaserBool)
    return biased_trials.extract_all(*args, **kwargs, extra_classes=extra_classes)

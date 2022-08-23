import logging
import numpy as np

from ibllib.io.extractors import biased_trials
from ibllib.io.extractors.base import BaseBpodTrialsExtractor

_logger = logging.getLogger(__name__)


class LaserBool(BaseBpodTrialsExtractor):
    """
    Extracts the laser probabilities from the bpod jsonable
    """
    save_names = ('_ibl_trials.laserStimulation.npy', '_ibl_trials.laserProbability.npy')
    var_names = ('laserStimulation', 'laserProbability')

    def _extract(self, **kwargs):
        _logger.info('Extracting laser datasets')
        # reference pybpod implementation
        lstim = np.array([float(t.get('laser_stimulation', np.NaN)) for t in self.bpod_trials])
        lprob = np.array([float(t.get('laser_probability', np.NaN)) for t in self.bpod_trials])

        # Karolina's choice world legacy implementation - from Slack message:
        # it is possible that some versions I have used:
        # 1) opto_ON_time (NaN - no laser or some number-laser)
        # opto_ON_time=~isnan(opto_ON_time)
        # laserON_trials=(opto_ON_time==1);
        # laserOFF_trials=(opto_ON_time==0);
        # 2) optoOUT (0 - no laser or 255 - laser):
        # laserON_trials=(optoOUT ==255);
        # laserOFF_trials=(optoOUT ==0);
        if 'PROBABILITY_OPTO' in self.settings.keys() and np.all(np.isnan(lstim)):
            lprob = np.zeros_like(lprob) + self.settings['PROBABILITY_OPTO']
            lstim = np.array([float(t.get('opto_ON_time', np.NaN)) for t in self.bpod_trials])
            if np.all(np.isnan(lstim)):
                lstim = np.array([float(t.get('optoOUT', np.NaN)) for t in self.bpod_trials])
                lstim[lstim == 255] = 1
            else:
                lstim[~np.isnan(lstim)] = 1
                lstim[np.isnan(lstim)] = 0

        if np.all(np.isnan(lprob)):
            # this prevents the file from being saved when no data
            self.save_names = ('_ibl_trials.laserStimulation.npy', None)
            _logger.warning('No laser probability found in bpod data')
        if np.all(np.isnan(lstim)):
            # this prevents the file from being saved when no data
            self.save_names = (None, '_ibl_trials.laserProbability.npy')
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

"""Mesoscope quality control.

This module runs a list of quality control metrics on the extracted imaging and cell detection
data.
"""
import logging
from collections import defaultdict
from inspect import getmembers, isfunction
import unittest
from pathlib import Path, PurePosixPath
from functools import wraps

import numpy as np
import scipy.stats
from one.alf import spec
from one.alf.spec import is_uuid

from . import base

_log = logging.getLogger(__name__)


def get_neural_quality_metrics(F, Fneu, badframes=None, iscell=None, F0_percentile=20, neuropil_factor=.7, **kwargs):
    """Compute neural quality metrics based on raw fluorescence traces.

    Parameters
    ----------
    F : numpy.array
        Raw fluorescence trace (nROIs by nTimepoints), e.g. mpci.ROINeuropilActivityF.
    Fneu : numpy.array
        Raw neuropil trace (nROIs by nTimepoints), e.g. mpci.ROIActivityF.
    badframes : numpy.array
        Indices of frames that should be excluded (default = all false).
    iscell : numpy.array
        Boolean array with true for cells, and false for not cells (default = all true).
    F0_percentile : int
        Percentile to be used for computing baseline fluorescence F0 (default = 20).
    neuropil_factor : float
        Factor to multiply neuropil with to get neuropil-corrected trace (default = 0.7).
        Must be between 0 and 1.
    times : numpy.array
        An array of frame times used to infer the frame rate, e.g. mpci.times.
    frame_rate : float
        The known frame rate of acquisition in Hz.  This value takes precedence when times
        also passed in. If neither times nor frame_rate passed, defaults to 7 Hz.

    Returns
    -------
    dict
        A dictionary with the following keys:
            noiseLevel: standardized shot noise level
            mean: time-averaged raw fluorescence (proxy for overall brightness)
            std: standard deviation of neuropil-corrected activity
            skew: skewness of neuropil-corrected activity
        Each averaged across all ROIs.
    numpy.array
        A structured numpy array with the fields noise_level, mean, std, and skew. One value
        per ROI.
    """
    if 'frame_rate' in kwargs:
        if (frame_rate := kwargs['frame_rate']) <= 0:
            raise ValueError('frame_rate must be positive')
        _log.info('Frame rate: %.2f Hz', frame_rate)
    elif 'times' in kwargs:
        frame_rate = 1 / np.median(np.diff(kwargs['times']))
        _log.info('Inferred frame rate of %.2f Hz from frame times', frame_rate)
    else:
        frame_rate = 7
        _log.warning('Assuming frame rate of %.2f Hz', frame_rate)
    if badframes is None:
        badframes = np.zeros(F.shape[0], dtype=bool)
    elif not isinstance(badframes, np.ndarray):
        raise TypeError(f'expected `badframes` to by numpy array, got `{type(badframes)}` instead')
    if iscell is None:
        iscell = np.ones(F.shape[0], dtype=bool)
    if neuropil_factor <= 0 or neuropil_factor > 1:
        raise ValueError('neuropil_factor must be between zero and one')

    # only take the good frames
    F = F[~badframes, :]
    Fneu = Fneu[~badframes, :]

    # F_npc is neuropil corrected trace
    F_npc = F - neuropil_factor * Fneu

    # dFF is deltaF / F0 in %, i.e. baseline-normalized fluorescence trace
    if F0_percentile is None:
        dFF = F
    else:
        F0 = np.percentile(F, F0_percentile, axis=0)  # F0 is some percentile of full trace
        F0_ = np.tile(F0, (F.shape[0], 1))
        dFF = (F - F0_) / F0_ * 100

    # compute noise level
    noise_levels = np.nanmedian(np.abs(np.diff(dFF, axis=0)), axis=0) / np.sqrt(frame_rate)  # Rupprecht et al. 2021

    # compute time-averaged raw fluorescence (proxy for overall brightness)
    means = np.mean(F, axis=0)

    # compute standard deviation of neuropil-corrected activity
    stds = np.std(F_npc, axis=0, ddof=0)

    # compute skewness of neuropil-corrected activity
    skews = scipy.stats.skew(F_npc, axis=0, bias=True)

    metrics = {'noise_level': noise_levels, 'mean': means, 'std': stds, 'skew': skews}

    # return all the neural quality metrics in a struct
    fov_metrics = np.rec.fromarrays(metrics.values(), dtype=np.dtype([(k, F.dtype) for k in metrics]))

    # return all the FOV-wide mean quality metrics in a struct
    neural_metrics = {k: np.nanmean(v[iscell]) for k, v in metrics.items()}
    return neural_metrics, fov_metrics


class TestQM(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(r'E:\integration\mesoscope\SP037\2023-03-23\002\alf\suite2p\plane2')

    def test_neural_qm(self):
        F = np.load(self.data_path / 'F.npy').T
        Fneu = np.load(self.data_path / 'Fneu.npy').T
        iscell = np.load(self.data_path / 'iscell.npy')
        neural_metrics, fov_metrics = get_neural_quality_metrics(F, Fneu, iscell=iscell[:, 0].astype(bool))
        expected = {'noise_level', 'mean', 'std', 'skew'}
        self.assertCountEqual(expected, neural_metrics)
        np.testing.assert_approx_equal(neural_metrics['noise_level'], 3.8925, significant=5)
        np.testing.assert_approx_equal(neural_metrics['mean'], 983.0024)
        np.testing.assert_approx_equal(neural_metrics['std'], 120.6435)
        np.testing.assert_approx_equal(neural_metrics['skew'], 1.0478, significant=5)
        self.assertCountEqual(expected, fov_metrics.dtype.names)
        self.assertTrue(all(fov_metrics[x].size == F.shape[1] for x in expected))


def return_qc_datasets(datasets):
    """
    Decorator that allows a function to return a predefined list of dataset names
    when the `only_dsets` keyword argument is set to True.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            only_dsets = kwargs.pop('only_dsets', False)
            if only_dsets:
                return datasets
            return func(*args, **kwargs)
        return wrapper
    return decorator


class MesoscopeQC(base.QC):
    """A class for computing mesoscope QC FOV metrics."""

    def _confirm_endpoint_id(self, endpoint_id):
        """Confirm the endpoint ID and set the name attribute.

        If the endpoint ID is a UUID, the name attribute is set to the name of the FOV from Alyx.
        Otherwise, the name attribute is set to the endpoint ID (assumed to be either FOV_XX or planeX).
        """
        if not is_uuid(endpoint_id, versions=(4,)):
            self.log.debug('Offline mode; skipping endpoint_id check')
            self.name = endpoint_id
            self.eid = None
            return
        super()._confirm_endpoint_id(endpoint_id)
        self.name = self.one.alyx.rest('field-of-views', 'read', id=endpoint_id)['name']

    def run(self, update: bool = False, **kwargs):
        """
        Run mesoscope QC checks and return outcome.

        Parameters
        ----------
        update : bool
            If true, updates the session QC fields on Alyx.

        Returns
        -------
        str
            The overall outcome.
        dict
            A map of checks and their outcomes.
        dict
            A dict of datasets and their outcomes
        """
        _log.info(f'Computing QC outcome for FOV {self.eid}')

        namespace = 'mesoscope'
        if not getattr(self, 'data', {}):
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(self.__class__, is_metric)

        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}
        self.datasets = {f'_{namespace}_' + k[6:]: fn(self, only_dsets=True) for k, fn in checks}

        values = [x if isinstance(x, spec.QC) else x[0] for x in self.metrics.values()]
        outcome = self.overall_outcome(values)

        # For each dataset get a list of qc values from relevant qc tests
        datasets = defaultdict(list)
        for check, dsets in self.datasets.items():
            qc_val = self.metrics[check] if isinstance(self.metrics[check], spec.QC) else self.metrics[check][0]
            for dset in dsets:
                datasets[dset].append(qc_val)
        # Compute the final qc for each dataset, takes the worst value
        dataset_qc = {k: self.overall_outcome(v) for k, v in datasets.items()}

        if update:
            extended = {
                k: 'NOT_SET' if v is None else v
                for k, v in self.metrics.items()
            }
            self.update_extended_qc(extended)
            self.update(outcome, namespace)

        return outcome, self.metrics, dataset_qc

    def load_data(self):
        """Load the data required for QC checks."""
        self.data = {}
        if self.name.startswith('FOV_'):
            # Load mpci objects
            alf_path = self.session_path.joinpath('alf', self.name)
            self.data['F'] = np.load(alf_path.joinpath('mpci.ROIActivityF.npy'))
            self.data['Fneu'] = np.load(alf_path.joinpath('mpci.ROINeuropilActivityF.npy'))
            self.data['iscell'] = np.load(alf_path.joinpath('mpciROIs.mpciROITypes.npy'))
            self.data['badframes'] = np.load(alf_path.joinpath('mpci.badFrames.npy'))
            s2pdata = np.load(alf_path.joinpath('_suite2p_ROIData.raw.zip'), allow_pickle=True)  # lazy load from zip
            self.data['ops'] = s2pdata['ops'].item()
            self.data['times'] = np.load(alf_path.joinpath('mpci.times.npy'))
        elif self.name.startswith('plane'):
            # Load suite2p objects
            alf_path = self.session_path.joinpath('suite2p', self.name)
            self.data['F'] = np.load(alf_path.joinpath('F.npy'))
            self.data['Fneu'] = np.load(alf_path.joinpath('Fneu.npy'))
            self.data['iscell'] = np.load(alf_path.joinpath('iscell.npy'))
            self.data['badframes'] = np.load(alf_path.joinpath('mpci.badFrames.npy.npy'))
            self.data['ops'] = np.load(alf_path.joinpath('ops.npy'), allow_pickle=True).item()
            self.data['times'] = None
        else:
            raise ValueError(f'Invalid session identifier: {self.eid}')

    @return_qc_datasets(['mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy', 'mpciROIs.mpciROITypes.npy',
                         'mpci.times.npy'])
    def check_neural_quality(self, **kwargs):
        """Check the neural quality metrics."""
        neural_metrics, fov_metrics = get_neural_quality_metrics(**self.data, **kwargs)

        # TODO Apply thresholds
        return spec.QC.NOT_SET, neural_metrics  # Don't return fov_metric as this is one per cell which is large

    @return_qc_datasets(['mpci.times.npy'])
    def check_timestamps_consistency(self, **kwargs):
        """Check the timestamps are the same length as the fluorescence data"""

        if self.data['times'] is None:
            return spec.QC.NOT_SET

        metrics = spec.QC.PASS if self.data['times'].size == self.data['F'].shape[0] else spec.QC.FAIL

        return metrics

    @staticmethod
    def qc_session(eid, one=None, **kwargs):
        """Run mesoscope QC checks on a session.

        This instantiates a MesoscopeQC object and runs the checks for each FOV in the session.
        It's not ideal to have one QC object per FOV - this could also be a single class that updates
        both the session endpoint and the FOV endpoints.

        The QC may be run on a local session before the suite2p outputs have been renamed to ALF format,
        however to update the FOV endpoint, this relies on the MesoscopeFOV task having been run, and for the
        data, the MesoscopePreprocess task.
        """
        update = kwargs.pop('update', False)
        session_qc = base.QC(eid, one=one)
        one = session_qc.one
        remote = session_qc.one and not session_qc.one.offline
        fov_qcs = dict()
        if remote:
            collections = session_qc.one.list_collections(eid, collection='alf/FOV_??')
            collections = sorted(map(session_qc.session_path.joinpath, collections))
            FOVs = one.alyx.rest('fields-of-view', 'list', session=session_qc.eid)
            for collection in collections:
                endpoint_id = next((x['id'] for x in FOVs if x['name'] == collection.name), None)
                if not endpoint_id:
                    _log.warning(f'No Alyx record for FOV {collection.name}')
                    continue
                qc = MesoscopeQC(endpoint_id, one=one, endpoint='fields-of-view')
                qc.session_path = session_qc.session_path
                fov_qcs[f'alf/{collection.name}'] = qc.run(update=update)
        else:
            collections = sorted(session_qc.session_path.glob('alf/FOV_??'))
            if not collections:
                collections = sorted(session_qc.session_path.glob('suite2p/plane*'))
            for collection in collections:
                qc = MesoscopeQC(collection.name, one=one, endpoint='fields-of-view')
                qc.session_path = session_qc.session_path
                fov_qcs[f'alf/{collection.name}'] = qc.run(update=update)

        # TODO Log or store outcomes for each FOV
        return fov_qcs


def update_dataset_qc_for_session(eid, qc, registered_datasets, one, override=False):
    """
    Update QC values for individual datasets associated to a session

    Parameters
    ----------
    eid: str or UUID
        The session identifier
    qc : dict
        Output from running MesoscopeQC.qc_session
    registered_datasets : list of dict
        A list of Alyx dataset records.
    one : one.api.OneAlyx
        An online instance of ONE.
    override : bool
        If True the QC field is updated even if new value is better than previous.

    Returns
    -------
    dict of lists
        For each collection in the qc returns a list of associated datasets that had their qc updated
    """
    datasets = {}
    for collection in qc.keys():
        reg_dsets = [d for d in registered_datasets if d['collection'] == collection]
        dsets = update_dataset_qc_for_collection(eid, collection, qc[collection][3], reg_dsets, one, override=override)
        datasets[collection] = dsets

    return datasets


def update_dataset_qc_for_collection(eid, collection, dataset_qc, registered_datasets, one, override=False):
    """
    Update QC values for individual datasets associated to a collection (normally 'alf/FOV_??')

    Parameters
    ----------
    eid: str or UUID
        The session identifier
    collection: str
        Collection that the datasets belong to. Assumes the collection is the same for all datasets provided in
        registered_datasets and dataset_qc.
    dataset_qc : dict
        A dictionary containing dataset name as keys and the associated qc as values
    registered_datasets : list of dict
        A list of Alyx dataset records.
    one : one.api.OneAlyx
        An online instance of ONE.
    override : bool
        If True the QC field is updated even if new value is better than previous.

    Returns
    -------
    list of dict
        The list of datasets with qcs updated with the 'qc' fields updated.
    """

    datasets = registered_datasets.copy()

    # Create map of dataset name, sans extension, to dataset id
    stem2id = {PurePosixPath(dset['name']).stem: dset.get('id') for dset in registered_datasets}
    # Ensure dataset stems are unique
    assert len(stem2id) == len(registered_datasets), 'ambiguous dataset names'

    # If the dataset that we are updating is not part of the registered datasets we need to fetch the dataset info
    extra_dsets = [k for k in dataset_qc if PurePosixPath(k).stem not in stem2id.keys()]
    for extra in extra_dsets:
        dset = one.alyx.rest('datasets', 'list', session=eid, name=extra, collection=collection)
        if len(dset) == 0:
            # If the dataset doesn't exist we continue
            _log.debug('dataset %s not registered, skipping', extra)
            continue
        if len(dset) > 0:
            # If multiple datasets find the default dataset
            dset = next(d for d in dset if d['default_dataset'])

        dset['id'] = dset['url'][-36:]
        stem2id[PurePosixPath(dset['name']).stem] = dset['id']

        datasets.append(dset)

    # Work over map of dataset name to outcome and update the dataset qc
    for name, outcome in dataset_qc.items():
        # Check if dataset was registered to Alyx
        if not (did := stem2id.get(PurePosixPath(name).stem)):
            _log.debug('dataset %s not registered, skipping', name)
            continue
        # Update the dataset QC value on Alyx
        if outcome > spec.QC.NOT_SET or override:
            dset_qc = base.QC(did, one=one, log=_log, endpoint='datasets')
            dset = next(x for x in datasets if did == x.get('id'))
            dset['qc'] = dset_qc.update(outcome, namespace='', override=override).name

    return datasets


if __name__ == '__main__':
    unittest.main()

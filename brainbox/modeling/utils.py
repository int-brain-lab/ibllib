import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.matlib import repmat


def raised_cosine(duration, nbases, binfun):
    nbins = binfun(duration)
    ttb = repmat(np.arange(1, nbins + 1).reshape(-1, 1), 1, nbases)
    dbcenter = nbins / nbases
    cwidth = 4 * dbcenter
    bcenters = 0.5 * dbcenter + dbcenter * np.arange(0, nbases)
    x = ttb - repmat(bcenters.reshape(1, -1), nbins, 1)
    bases = (np.abs(x / cwidth) < 0.5) * (np.cos(x * np.pi * 2 / cwidth) * 0.5 + 0.5)
    return bases


def full_rcos(duration, nbases, binfun, n_before=1):
    if not isinstance(n_before, int):
        n_before = int(n_before)
    nbins = binfun(duration)
    ttb = repmat(np.arange(1, nbins + 1).reshape(-1, 1), 1, nbases)
    dbcenter = nbins / (nbases - 2)
    cwidth = 4 * dbcenter
    bcenters = 0.5 * dbcenter + dbcenter * np.arange(-n_before, nbases - n_before)
    x = ttb - repmat(bcenters.reshape(1, -1), nbins, 1)
    bases = (np.abs(x / cwidth) < 0.5) * (np.cos(x * np.pi * 2 / cwidth) * 0.5 + 0.5)
    return bases


def neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    nzidx = f != 0
    if np.any(y[~nzidx] != 0):
        return np.inf
    return -y[nzidx].reshape(1, -1) @ xproj[nzidx] + np.sum(f)


class SequentialSelector:
    def __init__(self, model, n_features_to_select=None, direction='forward', scoring=None):
        """
        Sequential feature selection for neural models

        Parameters
        ----------
        model : brainbox.modeling.neural_model.NeuralModel
            Any class which inherits NeuralModel and has already been instantiated.
        n_features_to_select : int, optional
            Number of covariates to select. When None, will sequentially fit all parameters and
            store the associated scores. By default None
        direction : str, optional
            Direction of sequential selection. 'forward' indicates model will be built from 1
            regressor up, while 'backward' indicates regrssors will be removed one at a time until
            n_features_to_select is reached or 1 regressor remains. By default 'forward'
        scoring : str, optional
            Scoring function to use. Must be a valid argument to the subclass of NeuralModel passed
            to SequentialSelector. By default None
        """
        self.model = model
        self.design = model.design
        if n_features_to_select:
            self.n_features_to_select = int(n_features_to_select)
        else:
            self.n_features_to_select = len(self.design.covar)
        self.direction = direction
        self.scoring = scoring
        self.delta_scores = pd.DataFrame(index=self.model.clu_ids)
        self.trlabels = self.design.trlabels
        self.train = np.isin(self.trlabels, self.model.traininds).flatten()
        self.test = ~self.train
        self.features = np.array(list(self.design.covar.keys()))

    def fit(self, progress=False):
        """
        Fit the sequential feature selection

        Parameters
        ----------
        progress : bool, optional
            Whether to show a progress bar, by default False
        """
        n_features = len(self.features)
        maskdf = pd.DataFrame(index=self.model.clu_ids, columns=self.features, dtype=bool)
        maskdf.loc[:, :] = False
        seqdf = pd.DataFrame(index=self.model.clu_ids, columns=range(self.n_features_to_select))
        scoredf = pd.DataFrame(index=self.model.clu_ids, columns=range(self.n_features_to_select))

        if not 0 < self.n_features_to_select <= n_features:
            raise ValueError('n_features_to_select is not a valid number in the context'
                             ' of the model.')

        n_iterations = (
            self.n_features_to_select if self.direction == 'forward'
            else n_features - self.n_features_to_select
        )
        for i in tqdm(range(n_iterations), desc='step', leave=False, disable=not progress):
            masks_set = maskdf.groupby(self.features.tolist()).groups
            for current_mask in tqdm(masks_set, desc='feature subset', leave=False):
                cells = masks_set[current_mask]
                new_feature_idx, nf_score = self._get_best_new_feature(current_mask, cells)
                for cell in cells:
                    maskdf.at[cell, self.features[new_feature_idx.loc[cell]]] = True
                    seqdf.loc[cell, i] = self.features[new_feature_idx]
                    scoredf.loc[cell, i] = nf_score.loc[cell]
        self.support_ = maskdf
        self.sequences_ = seqdf
        self.scores_ = scoredf

    def _get_best_new_feature(self, mask, cells):
        mask = np.array(mask)
        candidate_features = np.flatnonzero(~mask)
        my = self.model.binnedspikes[self.train]
        scores = pd.DataFrame(index=cells, columns=candidate_features, dtype=float)
        for feature_idx in candidate_features:
            candidate_mask = mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == 'backward':
                candidate_mask = ~candidate_mask
            fitfeatures = self.features[candidate_mask]
            feat_idx = np.hstack([self.design.covar[feat]['dmcol_idx'] for feat in fitfeatures])
            mdm = self.design[np.ix_(self.train, feat_idx)]
            coefs, intercepts = self.model._fit(mdm, my, cells=cells)
            for i, cell in enumerate(cells):
                scores.at[cell, feature_idx] = self.model._scorer(coefs.loc[cell],
                                                                  intercepts.loc[cell],
                                                                  mdm, my[:, i])
        return scores.idxmax(axis=1), scores.max(axis=1)

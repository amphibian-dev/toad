import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .transform import WOETransformer, Combiner
from .utils import to_ndarray, bin_by_splits


class ScoreCard(BaseEstimator):
    def __init__(self, pdo = 60, rate = 2, base_odds = 35, base_score = 750):
        self.pdo = pdo
        self.rate = rate
        self.base_odds = base_odds
        self.base_score = base_score

        self.factor = pdo / np.log(rate)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, X, y, model = None, combiner = None):
        """
        Args:
            X (2D array-like)
            Y (array-like)
            model (estimator): sklearn logistic regression estimator
            combiner (dict)

        """
        if model is None:
            raise ValueError('model must be setted!')

        if combiner is None:
            raise ValueError('combiner must be setted!')

        self.weight = model.coef_[0]
        self.bias = model.intercept_[0]
        self.n_features_ = len(self.weight)

        self.set_combiner(combiner)

        bins = self.combine(X)
        transer = WOETransformer().fit(bins, y)
        self.score_map_ = self.score_map(transer)

        return self


    def set_combiner(self, combiner):
        """set combiner
        """
        if isinstance(combiner, Combiner):
            combiner = combiner.export()

        cb = dict()
        for key in combiner:
            c = combiner[key]
            if isinstance(c, np.ndarray):
                cb[key] = np.copy(c)
            else:
                cb[key] = np.array(c)

        self.combiner = combiner


    def predict(self, X, **kwargs):
        """predict score
        Args:
            X (2D array-like): X to predict
            return_sub (bool): if need to return sub score, default `False`

        Returns:
            array-like: predicted score
            DataFrame: sub score for each feature
        """
        bins = self.combine(X)
        return self.bin_to_score(bins, **kwargs)

    def proba_to_score(self, prob):
        """covert probability to score
        """
        odds = (1 - prob) / prob
        return self.factor * np.log(odds) + self.offset

    def combine(self, X):
        res = X.copy()

        for col in self.combiner:
            groups = self.combiner[col]
            val = res[col].values
            if np.issubdtype(groups.dtype, np.number):
                val = bin_by_splits(val, groups)
            else:
                # set default group to -1
                b = np.full(val.shape, -1)
                for i in range(len(groups)):
                    b[np.isin(val, groups[i])] = i
                val = b

            res[col] = val

        return res

    def bin_to_score(self, bins, return_sub = False):
        """predict score from bins
        """
        res = bins.copy()
        for col in bins:
            s_map = self.score_map_[col]
            b = bins[col].values
            # set default group to min score
            b[b == -1] = np.argmin(s_map)
            # replace score
            res[col] = s_map[b]

        score = np.sum(res.values, axis = 1)

        if return_sub is False:
            return score

        return score, res

    def woe_to_score(self, woe, weight = None):
        """calculate score by woe
        """
        woe = to_ndarray(woe)

        if weight is None:
            weight = self.weight

        b = self.offset - self.factor * self.bias
        s = -self.factor * weight * woe

        return s + b / self.n_features_

    def score_map(self, transer):
        """calculate score map by woe
        """
        keys = list(transer.values_.keys())

        s_map = dict()
        for i, k in enumerate(keys):
            weight = self.weight[i]
            woe = transer.woe_[k]
            s_map[k] = self.woe_to_score(woe, weight = weight)

        return s_map

    def export_map(self):
        """generate a scorecard object

        Returns:
            dict
        """
        card = dict()
        for col in self.combiner:
            group = self.combiner[col]
            card[col] = dict()

            if not np.issubdtype(group.dtype, np.number):
                for i, v in enumerate(group):
                    card[col][','.join(v)] = self.score_map_[col][i]
            else:
                sp_l = [-np.inf] + group.tolist() + [np.inf]
                for i in range(len(sp_l) - 1):
                    card[col]['['+str(sp_l[i])+' ~ '+str(sp_l[i+1])+')'] = self.score_map_[col][i]

        return card

import math
import numpy as np
import pandas as pd
from .stats import WOE
from sklearn.base import TransformerMixin

from .utils import to_ndarray, np_count, bin_by_splits
from .merge import DTMerge, ChiMerge, StepMerge, QuantileMerge, KMeansMerge

ELSE_GROUP = 'else'


class WOETransformer(TransformerMixin):
    """WOE transformer
    """
    def fit(self, X, y, **kwargs):
        """
        """
        if not isinstance(X, pd.DataFrame):
            self.values_, self.woe_ = self._fit_woe(X, y, **kwargs)
            return self

        if isinstance(y, str):
            y = X.pop(y)

        self.values_ = dict()
        self.woe_ = dict()
        for col in X:
            self.values_[col], self.woe_[col] = self._fit_woe(X[col], y)

        return self

    def _fit_woe(self, X, y):
        t_counts_0 = np_count(y, 0, default = 1)
        t_counts_1 = np_count(y, 1, default = 1)

        values = np.unique(X)
        l = len(values)
        woe = np.zeros(l)

        for i in range(l):
            sub_target = y[X == values[i]]

            sub_0 = np_count(sub_target, 0, default = 1)
            sub_1 = np_count(sub_target, 1, default = 1)

            y_prob = sub_1 / t_counts_1
            n_prob = sub_0 / t_counts_0

            woe[i] = WOE(y_prob, n_prob)

        return values, woe


    def transform(self, X, **kwargs):
        if not isinstance(self.values_, dict):
            return self._transform_apply(X, self.values_, self.woe_, **kwargs)

        res = X.copy()
        for col in X:
            if col in self.values_:
                res[col] = self._transform_apply(X[col], self.values_[col], self.woe_[col], **kwargs)

        return res


    def _transform_apply(self, X, value, woe, default = 'min'):
        X = to_ndarray(X)
        res = np.zeros(len(X))

        if default is 'min':
            default = np.min(woe)
        elif default is 'max':
            default = np.max(woe)

        # replace unknown group to default value
        res[np.isin(X, value, invert = True)] = default

        for i in range(len(value)):
            res[X == value[i]] = woe[i]

        return res


class Combiner(TransformerMixin):
    def fit(self, X, y = None, **kwargs):
        if not isinstance(X, pd.DataFrame):
            self.splits_ = self._merge(X, y = y, **kwargs)
            return self

        if isinstance(y, str):
            y = X.pop(y)

        self.splits_ = dict()
        for col in X:
            self.splits_[col] = self._merge(X[col], y = y, **kwargs)

        return self

    def _merge(self, X, y = None, method = 'chi', **kwargs):
        X = to_ndarray(X)

        if y is not None:
            y = to_ndarray(y)

        uni_val = False
        if not np.issubdtype(X.dtype, np.number):
            # transform raw data by woe
            transer = WOETransformer()
            woe = transer.fit_transform(X, y)
            uni_woe, ix_woe = np.unique(woe, return_index = True)
            # sort value by woe
            ix = np.argsort(uni_woe)
            uni_val = X[ix_woe[ix]]
            # replace X by sorted index
            X = self._raw_to_bin(X, uni_val)

        if method is 'dt':
            splits = DTMerge(X, y, **kwargs)
        elif method is 'chi':
            splits = ChiMerge(X, y, **kwargs)
        elif method is 'quantile':
            splits = QuantileMerge(X, **kwargs)
        elif method is 'step':
            splits = StepMerge(X, **kwargs)
        elif method is 'kmeans':
            splits = KMeaMerge(X, target = y, **kwargs)

        return self._covert_splits(uni_val, splits)

    def transform(self, X):
        if not isinstance(self.splits_, dict):
            return self._transform_apply(X, self.splits_)

        res = X.copy()
        for col in X:
            if col in self.splits_:
                res[col] = self._transform_apply(X[col], self.splits_[col])

        return res

    def _transform_apply(self, X, splits):
        X = to_ndarray(X)

        # if is not continuous
        if not np.issubdtype(splits.dtype, np.number):
            return self._raw_to_bin(X, splits)

        if len(splits):
            bins = bin_by_splits(X, splits)
        else:
            bins = np.zeros(len(X))

        return bins

    def _raw_to_bin(self, X, splits):
        # set default group to -1
        bins = np.full(X.shape, -1)
        for i in range(len(splits)):
            group = splits[i]
            # if group is else, set all empty group to it
            if isinstance(group, str) and group == ELSE_GROUP:
                bins[bins == -1] = i
            else:
                bins[np.isin(X, group)] = i

        return bins

    def set_rules(self, map):
        if not isinstance(map, dict):
            self.splits_ = np.array(map)

        self.splits_ = dict()
        for col in map:
            self.splits_[col] = np.array(map[col])

        return self

    def export(self):
        """export combine rules for score card
        """
        return self.splits_

    def _covert_splits(self, value, splits):
        """covert combine rules to array
        """
        if value is False:
            return splits

        start = 0
        l = list()
        for i in splits:
            i = math.ceil(i)
            l.append(np.array(value[start:i]))
            start = i

        l.append(value[start:])

        return np.array(l)

import numpy as np
from .stats import WOE
from sklearn.base import TransformerMixin

from .utils import to_ndarray, np_count, bin_by_splits
from .merge import DTMerge, ChiMerge, StepMerge, QuantileMerge, KMeansMerge


def trans_woe(X, y):
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


# TODO use nd array replace dataframe
class WOETransformer(TransformerMixin):

    def fit(self, X, y, ix = 0):
        X = to_ndarray(X)
        y = to_ndarray(y)

        self.values_ = list()
        self.woe_ = list()

        if X.ndim == 1:
            X = X.reshape((-1, 1))

        for col in X.T:
            val, woe = trans_woe(col, y)
            self.values_.append(val)
            self.woe_.append(woe)

        return self


    def transform(self, X, ix = 0):
        X = to_ndarray(X)

        if X.ndim == 1:
            return self._transfrom_apply(X, self.values_[0], self.woe_[0])

        _, n_col = X.shape
        woe = np.zeros(X.shape)
        for i in range(n_col):
            woe[:, i] = self._transfrom_apply(X[:, i], self.values_[i], self.woe_[i])

        return woe

    def _transfrom_apply(self, X, value, woe):
        res = np.zeros(len(X))

        for i in range(len(value)):
            res[X == value[i]] = woe[i]

        return res


class Combiner(TransformerMixin):
    def fit(self, X, y = None, method = 'chi', **kwargs):
        X = to_ndarray(X)

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

        self.splits_ = splits

        return self

    def transform(self, X):
        if len(self.splits_):
            bins = bin_by_splits(X, self.splits_)
        else:
            bins = np.zeros(len(X))

        return bins

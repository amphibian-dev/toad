import numpy as np
from .stats import WOE

from .utils import to_ndarray, np_count, bin_by_splits
from .merge import DTMerge, ChiMerge, StepMerge, QuantileMerge, KMeansMerge


class WOETransformer:
    def fit(self, feature, target):
        feature = to_ndarray(feature)
        target = to_ndarray(target)

        t_counts_0 = np_count(target, 0, default = 1)
        t_counts_1 = np_count(target, 1, default = 1)

        self.values_ = np.unique(feature)
        l = len(self.values_)
        woe = np.zeros(l)

        for i in range(l):
            sub_target = target[feature == self.values_[i]]

            sub_0 = np_count(sub_target, 0, default = 1)
            sub_1 = np_count(sub_target, 1, default = 1)

            y_prob = sub_1 / t_counts_1
            n_prob = sub_0 / t_counts_0

            woe[i] = WOE(y_prob, n_prob)

        self.woe_ = woe

        return self


    def transform(self, feature):
        """
        """
        feature = to_ndarray(feature)

        woe = np.zeros(len(feature))
        for i in range(len(self.values_)):
            woe[feature == self.values_[i]] = self.woe_[i]

        return woe


    def fit_transform(self, feature, target):
        self.fit(feature, target)

        return self.transform(feature)


class Combiner:
    def fit(self, feature, target = None, method = 'chi', **kwargs):
        feature = to_ndarray(feature)

        if method is 'dt':
            splits = DTMerge(feature, target, **kwargs)
        elif method is 'chi':
            splits = ChiMerge(feature, target, **kwargs)
        elif method is 'quantile':
            splits = QuantileMerge(feature, **kwargs)
        elif method is 'step':
            splits = StepMerge(feature, **kwargs)
        elif method is 'kmeans':
            splits = KMeaMerge(feature, target = target, **kwargs)

        self.splits_ = splits

        return self

    def transform(self, feature):
        if len(self.splits_):
            bins = bin_by_splits(feature, self.splits_)
        else:
            bins = np.zeros(len(feature))

        return bins

    def fit_transform(self, feature, target = None, **kwargs):
        self.fit(feature, target = target, **kwargs)

        return self.transform(feature)

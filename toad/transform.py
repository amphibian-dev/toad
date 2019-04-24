import math
import copy
import numpy as np
import pandas as pd
from functools import wraps
from sklearn.base import TransformerMixin

from .stats import WOE, probability
from .merge import merge
from .utils import to_ndarray, np_count, bin_by_splits, save_json

EMPTY_BIN = -1
ELSE_GROUP = 'else'


def support_select_dtypes(fn):

    @wraps(fn)
    def func(self, X, *args, select_dtypes = None, **kwargs):
        if select_dtypes is not None and isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include = select_dtypes)

        return fn(self, X, *args, **kwargs)

    return func


def support_exclude(fn):
    @wraps(fn)
    def func(self, X, *args, exclude = None, **kwargs):
        if exclude is not None and isinstance(X, pd.DataFrame):
            X = X.drop(columns = exclude)

        return fn(self, X, *args, **kwargs)

    return func


class WOETransformer(TransformerMixin):
    """WOE transformer
    """
    @support_exclude
    @support_select_dtypes
    def fit(self, X, y, **kwargs):
        """fit WOE transformer

        Args:
            X (DataFrame|array-like)
            y (str|array-like)
            select_dtypes (str|numpy.dtypes): `'object'`, `'number'` etc. only selected dtypes will be transform,
        """
        if not isinstance(X, pd.DataFrame):
            self.values_, self.woe_ = self._fit_woe(X, y, **kwargs)
            return self

        if isinstance(y, str):
            X = X.copy()
            y = X.pop(y)

        self.values_ = dict()
        self.woe_ = dict()

        for col in X:
            self.values_[col], self.woe_[col] = self._fit_woe(X[col], y)

        return self

    def _fit_woe(self, X, y):
        X = to_ndarray(X)

        values = np.unique(X)
        l = len(values)
        woe = np.zeros(l)

        for i in range(l):
            y_prob, n_prob = probability(y, mask = (X == values[i]))

            woe[i] = WOE(y_prob, n_prob)

        return values, woe


    def transform(self, X, **kwargs):
        """transform woe

        Args:
            X (DataFrame|array-like)
            default (str): 'min'(default), 'max' - the strategy to be used for unknown group

        Returns:
            array-like
        """
        if not isinstance(self.values_, dict):
            return self._transform_apply(X, self.values_, self.woe_, **kwargs)

        res = X.copy()
        for col in X:
            if col in self.values_:
                res[col] = self._transform_apply(X[col], self.values_[col], self.woe_[col], **kwargs)

        return res


    def _transform_apply(self, X, value, woe, default = 'min'):
        """transform function for single feature

        Args:
            X (array-like)
            value (array-like)
            woe (array-like)
            default (str): 'min'(default), 'max' - the strategy to be used for unknown group

        Returns:
            array-like
        """
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
    @support_exclude
    @support_select_dtypes
    def fit(self, X, y = None, **kwargs):
        """fit combiner

        Args:
            X (DataFrame|array-like): features to be combined
            y (str|array-like): target data or name of target in `X`
            method (str): the strategy to be used to merge `X`, same as `.merge`, default is `chi`
            n_bins (int): counts of bins will be combined

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            self.splits_ = self._merge(X, y = y, **kwargs)
            return self

        if isinstance(y, str):
            X = X.copy()
            y = X.pop(y)

        self.splits_ = dict()
        for col in X:
            self.splits_[col] = self._merge(X[col], y = y, **kwargs)

        return self

    def _merge(self, X, y = None, method = 'chi', **kwargs):
        """merge function for fit

        Args:
            X (DataFrame|array-like): features to be combined
            y (str|array-like): target data or name of target in `X`
            method (str): the strategy to be used to merge `X`, same as `.merge`, `chi` by default

        Returns:
            array-like: array of splits
        """
        X = to_ndarray(X)

        if y is not None:
            y = to_ndarray(y)

        uni_val = False
        if not np.issubdtype(X.dtype, np.number):
            # transform raw data by woe
            transer = WOETransformer()
            woe = transer.fit_transform(X, y)
            # find unique value and its woe value
            uni_val, ix_val = np.unique(X, return_index = True)
            uni_woe = woe[ix_val]
            # sort value by woe
            ix = np.argsort(uni_woe)
            # sort unique value
            uni_val = uni_val[ix]
            # replace X by sorted index
            X = self._raw_to_bin(X, uni_val)

        _, splits = merge(X, target = y, method = method, return_splits = True, **kwargs)

        return self._covert_splits(uni_val, splits)

    def transform(self, X, **kwargs):
        """transform X by combiner

        Args:
            X (DataFrame|array-like): features to be transformed
            labels (bool): if need to use labels for resulting bins, `False` by default

        Returns:
            array-like
        """
        if not isinstance(self.splits_, dict):
            return self._transform_apply(X, self.splits_, **kwargs)

        res = X.copy()
        for col in X:
            if col in self.splits_:
                res[col] = self._transform_apply(X[col], self.splits_[col], **kwargs)

        return res

    def _transform_apply(self, X, splits, labels = False):
        """transform function for single feature

        Args:
            X (array-like): feature to be transformed
            splits (array-like): splits of `X`
            labels (bool): if need to use labels for resulting bins, `False` by default

        Returns:
            array-like
        """
        X = to_ndarray(X)

        # if is not continuous
        if splits.ndim > 1 or not np.issubdtype(splits.dtype, np.number):
            bins = self._raw_to_bin(X, splits)

        else:
            if len(splits):
                bins = bin_by_splits(X, splits)
            else:
                bins = np.zeros(len(X), dtype = int)

        if labels:
            formated = self._format_splits(splits, index = True)
            empty_mask = (bins == EMPTY_BIN)
            bins = formated[bins]
            bins[empty_mask] = EMPTY_BIN

        return bins

    def _raw_to_bin(self, X, splits):
        """bin by splits

        Args:
            X (array-like): feature to be combined
            splits (array-like): splits of `X`

        Returns:
            array-like
        """
        # set default group to EMPTY_BIN
        bins = np.full(X.shape, EMPTY_BIN)
        for i in range(len(splits)):
            group = splits[i]
            # if group is else, set all empty group to it
            if isinstance(group, str) and group == ELSE_GROUP:
                bins[bins == EMPTY_BIN] = i
            else:
                bins[np.isin(X, group)] = i

        return bins

    def _format_splits(self, splits, index = False):
        l = list()
        if np.issubdtype(splits.dtype, np.number):
            sp_l = [-np.inf] + splits.tolist() + [np.inf]
            for i in range(len(sp_l) - 1):
                l.append('['+str(sp_l[i])+' ~ '+str(sp_l[i+1])+')')
        else:
            for keys in splits:
                if isinstance(keys, str) and keys == ELSE_GROUP:
                    l.append(keys)
                else:
                    l.append(','.join(keys))

        if index:
            indexes = [i for i in range(len(l))]
            l = ["{}.{}".format(ix, lab) for ix, lab in zip(indexes, l)]

        return np.array(l)

    def set_rules(self, map):
        """set rules for combiner

        Args:
            map (dict|array-like): map of splits

        Returns:
            self
        """
        if not isinstance(map, dict):
            self.splits_ = np.array(map)

        self.splits_ = dict()
        for col in map:
            self.splits_[col] = np.array(map[col])

        return self


    @property
    def dtypes(self):
        """get the dtypes which is combiner used

        Returns:
            (str|dict)
        """
        if not isinstance(self.splits_, dict):
            return self._get_dtype(self.splits_)

        t = {}
        for n, v in self.splits_.items():
            t[n] = self._get_dtype(v)
        return t

    def _get_dtype(self, split):
        if np.issubdtype(split.dtype, np.number):
            return 'numeric'

        return 'object'


    def export(self, format = False, to_json = None):
        """export combine rules for score card

        Args:
            format (bool): if True, bins will be replace with string label for values
            to_json (str|IOBase): io to write json file

        Returns:
            dict
        """
        splits = copy.deepcopy(self.splits_)

        if format:
            if not isinstance(splits, dict):
                splits = self._format_splits(splits)
            else:
                for col in splits:
                    splits[col] = self._format_splits(splits[col])

        if not isinstance(splits, dict):
            bins = splits.tolist()
        else:
            bins = {k: v.tolist() for k, v in splits.items()}

        if to_json is None:
            return bins

        save_json(bins, to_json)


    def _covert_splits(self, value, splits):
        """covert combine rules to array
        """
        if value is False:
            return splits

        if isinstance(value, np.ndarray):
            value = value.tolist()

        start = 0
        l = list()
        for i in splits:
            i = math.ceil(i)
            l.append(value[start:i])
            start = i

        l.append(value[start:])

        return np.array(l)

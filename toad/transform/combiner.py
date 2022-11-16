import math
import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator, 
    TransformerMixin
)

from .base import Transformer
from ..utils.mixin import BinsMixin
from ..utils.func import to_ndarray


class Combiner(Transformer, BinsMixin):
    """Combiner for merge data
    """
    def fit_(self, X, y = None, method = 'chi', empty_separate = False, **kwargs):
        """fit combiner

        Args:
            X (DataFrame|array-like): features to be combined
            y (str|array-like): target data or name of target in `X`
            method (str): the strategy to be used to merge `X`, same as `.merge`, default is `chi`
            n_bins (int): counts of bins will be combined
            empty_separate (bool): if need to combine empty values into a separate group
        """
        from ..merge import merge

        X = to_ndarray(X)

        if y is not None:
            y = to_ndarray(y)


        if not np.issubdtype(X.dtype, np.number):
            if y is None:
                raise ValueError("Can not combine `{dtype}` type in X, if you want to combine this type columns, please pass argument `y` to deal with it".format(dtype = X.dtype))

            from .woe_transformer import WOETransformer
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
        

        mask = pd.isna(X)
        if mask.any() and empty_separate:
            X = X[~mask]
            y = y[~mask]
        
        _, splits = merge(X, target = y, method = method, return_splits = True, **kwargs)

        if mask.any() and empty_separate:
            splits = np.append(splits, np.nan)
        
        return splits


    def transform_(self, rule, X, labels = False, ellipsis = 16, **kwargs):
        """transform X by combiner

        Args:
            X (DataFrame|array-like): features to be transformed
            labels (bool): if need to use labels for resulting bins, `False` by default
            ellipsis (int): max length threshold that labels will not be ellipsis, `None` for skipping ellipsis

        Returns:
            array-like
        """

        X = to_ndarray(X)

        # if is not continuous
        if rule.ndim > 1 or not np.issubdtype(rule.dtype, np.number):
            bins = self._raw_to_bin(X, rule)

        else:
            bins = np.zeros(X.shape, dtype = int)

            if len(rule):
                from ..utils.func import bin_by_splits

                # empty to a separate group
                if np.isnan(rule[-1]):
                    mask = pd.isna(X)
                    bins[~mask] = bin_by_splits(X[~mask], rule[:-1])
                    bins[mask] = len(rule)
                else:
                    bins = bin_by_splits(X, rule)

        if labels:
            formated = self.format_bins(rule, index = True, ellipsis = ellipsis)
            empty_mask = (bins == self.EMPTY_BIN)
            bins = formated[bins]
            bins[empty_mask] = self.EMPTY_BIN

        return bins


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

        return np.array(l, dtype = object)


    def _raw_to_bin(self, X, splits):
        """bin by splits

        Args:
            X (array-like): feature to be combined
            splits (array-like): splits of `X`

        Returns:
            array-like
        """
        # set default group to EMPTY_BIN
        bins = np.full(X.shape, self.EMPTY_BIN)
        for i in range(len(splits)):
            group = splits[i]
            # if group is else, set all empty group to it
            if isinstance(group, str) and group == self.ELSE_GROUP:
                bins[bins == self.EMPTY_BIN] = i
            else:
                bins[np.isin(X, group)] = i

        return bins


    def set_rules(self, map, reset = False):
        """set rules for combiner

        Args:
            map (dict|array-like): map of splits
            reset (bool): if need to reset combiner

        Returns:
            self
        """
        import warnings
        warnings.warn(
            """`combiner.set_rules` will be deprecated soon,
                use `combiner.load(rules, update = False)` instead!
            """,
            DeprecationWarning,
        )


        self.load(map, update = not reset)

        return self

    def _parse_rule(self, rule):
        return np.array(rule)

    def _format_rule(self, rule, format = False):
        if format:
            rule = self.format_bins(rule)

        return rule.tolist()

# make a transformer for combiner, therefore the combiner could participate in pipeline as well as GridSearchCV for combiner's parameters tuning
class CombinerTransformer4pipe(BaseEstimator, TransformerMixin):
    """ A Transformer spcifically for combiner, which make it more flexible
    """

    def __init__(
        self,
        method='chi', 
        empty_separate=False, 
        min_samples=0.05,
        n_bins=None,
        update_rules={},
        exclude=None,
        **kwargs
    ):
        """_summary_

        Args:
            method (str): the strategy to be used to merge `X`, same as `.merge`, default is `chi`
            empty_separate (bool): if need to combine empty values into a separate group
            min_samples (float): threshold of percentage of each bins
            n_bins (int): counts of bins will be combined
            update_rules (dict): fixed bin rules from prior experience
            exclude (array-like): list of feature name that will not be dropped
        """
        super().__init__()
        self.combiner = Combiner()
        # setting up all necessary parameters for the combiner
        self.model_params = {
            'method' : method,
            'empty_separate' : empty_separate,
            'min_samples' : min_samples,
            'n_bins' : n_bins,
            'exclude' : exclude
        }
        self.model_params.update(kwargs)

        # set all incoming parameters as properties of self class
        for k, v in self.model_params.items():
            setattr(self, k, v)
        
        self.update_rules = update_rules

    def fit(self, X, y):
        """fit combiner

        Args:
            X (DataFrame): features to be selected, and note X only contains features, no labels
            y (array-like): Label of the sample

        Returns:
            self
        """        
        # reset self properties because there might be combiner.set_params() before fit
        for key in self.model_params.keys():
            self.model_params[key] = getattr(self, key) 

        # if the input rules are not {}, this implies that the features and its bins in the dict are already set and should be fixed.
        # Therefore the combiner should not re-calculated the bins for these features, which could much more efficient. 
        # The achieve this fucntion, one simple method is to first append these features into the exclude parameter, and update the dict after the fit progress
        if len(self.update_rules) > 0:
            if self.model_params['exclude'] is None:
                self.model_params['exclude'] = list(self.update_rules.keys())
            else:
                self.model_params['exclude'] += list(self.update_rules.keys())

        self.combiner = self.combiner.fit(X, y, **self.model_params)

        if len(self.update_rules) > 0:
            self.combiner.update(self.update_rules)

        return self

    def transform(self, X, y=None):
        """transform X by combiner

        Args:
            X (DataFrame): features to be transformed

        Returns:
            DataFrame
        """
        return self.combiner.transform(X)
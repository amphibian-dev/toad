import numpy as np

from .base import Transformer
from ..utils.func import to_ndarray


class WOETransformer(Transformer):
    """WOE transformer
    """

    def fit_(self, X, y):
        """fit WOE transformer

        Args:
            X (DataFrame|array-like)
            y (str|array-like)
            select_dtypes (str|numpy.dtypes): `'object'`, `'number'` etc. only selected dtypes will be transform
        """
        from ..stats import WOE, probability

        X = to_ndarray(X)

        value = np.unique(X)
        l = len(value)
        woe = np.zeros(l)

        for i in range(l):
            y_prob, n_prob = probability(y, mask = (X == value[i]))

            woe[i] = WOE(y_prob, n_prob)

        return {
            'value': value,
            'woe': woe,
        }

    def transform_(self, rule, X, default = 'min'):
        """transform function for single feature

        Args:
            X (array-like)
            default (str): 'min'(default), 'max' - the strategy to be used for unknown group

        Returns:
            array-like
        """
        X = to_ndarray(X)
        res = np.zeros(X.shape)

        value = rule['value']
        woe = rule['woe']

        if default == 'min':
            default = np.min(woe)
        elif default == 'max':
            default = np.max(woe)

        # replace unknown group to default value
        res[np.isin(X, value, invert = True)] = default

        for i in range(len(value)):
            res[X == value[i]] = woe[i]

        return res

    def _format_rule(self, rule):
        return dict(zip(rule['value'], rule['woe']))

    def _parse_rule(self, rule):
        return {
            'value': np.array(list(rule.keys())),
            'woe': np.array(list(rule.values())),
        }

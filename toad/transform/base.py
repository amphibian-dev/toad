import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


from ..utils.func import split_target
from ..utils.decorator import frame_exclude, select_dtypes
from ..utils.mixin import RulesMixin


class Transformer(TransformerMixin, RulesMixin):
    """Base class for transformers
    """

    _fit_frame = False

    @property
    def _fitted(self):
        return len(self.rules) > 0


    @frame_exclude(is_class = True)
    @select_dtypes(is_class = True)
    def fit(self, X, *args, update = False, **kwargs):
        """fit method, see details in `fit_` method
        """
        dim = getattr(X, 'ndim', 1)

        rules = {}

        if self._fit_frame:
            rules = self.fit_(X, *args, **kwargs)

        elif dim == 1:
            name = getattr(X, 'name', self._default_name)
            rules[name] = self.fit_(X, *args, **kwargs)

        else:
            if len(args) > 0:
                X, y = split_target(X, args[0])
                args = (y, *args[1:])
            if 'y' in kwargs:
                X, kwargs['y'] = split_target(X, kwargs['y'])

            self._check_duplicated_keys(X)

            for col in X:
                name = X[col].name
                rules[name] = self.fit_(X[col], *args, **kwargs)

        if update:
            self.rules.update(rules)
        else:
            self.rules = rules

        return self


    def transform(self, X, *args, **kwargs):
        """transform method, see details in `transform_` method
        """
        if not self._fitted:
            return self._raise_unfitted()


        if self._fit_frame:
            return self.transform_(self.rules, X, *args, **kwargs)

        if getattr(X, 'ndim', 1) == 1 and not isinstance(X, dict):
            if len(self.rules) == 1:
                return self.transform_(self.default_rule(), X, *args, **kwargs)
            elif hasattr(X, 'name') and X.name in self:
                return self.transform_(self.rules[X.name], X, *args, **kwargs)
            else:
                return X

        self._check_duplicated_keys(X)

        res = X.copy()
        for key in X:
            if key in self.rules:
                try:
                    res[key] = self.transform_(self.rules[key], X[key], *args, **kwargs)
                except Exception as e:
                    e.args += ('on column "{key}"'.format(key = key),)
                    raise e

        return res


    def _raise_unfitted(self):
        raise Exception('transformer is unfitted yet!')
    

    def _check_duplicated_keys(self, X):
        if isinstance(X, pd.DataFrame) and X.columns.has_duplicates:
            keys = X.columns[X.columns.duplicated()].values
            raise Exception("X has duplicate keys `{keys}`".format(keys = str(keys)))
        
        return True









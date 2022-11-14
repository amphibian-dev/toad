from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

from .base import Transformer

class GBDTTransformer(Transformer):
    """GBDT transformer
    """
    _fit_frame = True

    def __init__(self):
        self.gbdt = None
        self.onehot = None


    def fit_(self, X, y, **kwargs):
        """fit GBDT transformer

        Args:
            X (DataFrame|array-like)
            y (str|array-like)
            select_dtypes (str|numpy.dtypes): `'object'`, `'number'` etc. only selected dtypes will be transform,
        """

        if isinstance(y, str):
            X = X.copy()
            y = X.pop(y)

        gbdt = GradientBoostingClassifier(**kwargs)
        gbdt.fit(X, y)

        X = gbdt.apply(X)
        X = X.reshape(-1, X.shape[1])

        onehot = OneHotEncoder().fit(X)

        return {
            'gbdt': gbdt,
            'onehot': onehot,
        }


    def transform_(self, rules, X):
        """transform woe

        Args:
            X (DataFrame|array-like)

        Returns:
            array-like
        """
        X = rules['gbdt'].apply(X)
        X = X.reshape(-1, X.shape[1])
        res = rules['onehot'].transform(X).toarray()
        return res

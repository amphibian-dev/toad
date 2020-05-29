import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder



def impute(df):
    imputer = Imputer(
        estimator = RandomForestRegressor(),
        random_state = 1,
    )

    return imputer.fit_transform(df)


class Imputer(IterativeImputer):
    def __init__(self, missing_values = np.nan, **kwargs):
        super().__init__(missing_values = np.nan, **kwargs)

        if not isinstance(missing_values, list):
            missing_values = [missing_values]
        
        self.missing_values_list = missing_values
        self.encoder_dict = dict()

    def _impute_one_feature(self, X_filled, mask_missing_values, feat_idx,
                            neighbor_feat_idx, **kwargs):
        
        return super()._impute_one_feature(X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, **kwargs)

    def fit_transform(self, X, **kwargs):
        X, mask = self._replace_empty(X)
        X = self._fit_encode(X, mask)

        res = super().fit_transform(X, **kwargs)
        res = pd.DataFrame(res, columns = X.columns)
        return self._decode(res)

    
    def transform(self, X, **kwargs):
        X, mask = self._replace_empty(X)
        X = self._encode(X, mask)

        res = super().transform(X, **kwargs)
        res = pd.DataFrame(res, columns = X.columns)
        return self._decode(res)
    

    def _replace_empty(self, X):
        mask = X.isin(self.missing_values_list)
        X = X.where(~mask, np.nan)
        return X, mask

    def _fit_encode(self, X, mask):
        """fit encoder for object data

        Args:
            X (DataFrame)
            mask (Mask): empty mask for X
        """
        category_data = X.select_dtypes(exclude = np.number).columns
        
        for col in category_data:
            unique, X[col].loc[~mask[col]] = np.unique(X[col][~mask[col]], return_inverse = True)

            self.encoder_dict[col] = unique
        
        return X

    def _encode(self, X, mask):
        """encode object data to number

        Args:
            X (DataFrame)
            mask (Mask): empty mask for X
        """
        for col, unique in self.encoder_dict.items():
            table = dict(zip(unique, np.arange(len(unique))))
            X[col].loc[~mask[col]] = np.array([table[v] for v in X[col][~mask[col]]])
        
        return X
    
    def _decode(self, X):
        """decode object data from number to origin data

        Args:
            X (DataFrame)
            mask (Mask): empty mask for X
        """
        for col, unique in self.encoder_dict.items():
            ix = X[col].values.astype(int)
            X[col] = unique[ix]
        
        return X

    



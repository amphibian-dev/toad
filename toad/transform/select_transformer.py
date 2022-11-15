import pandas as pd
from ..selection import select
from sklearn.base import (
    BaseEstimator, 
    TransformerMixin
)

class SelectTransformer4pipe(BaseEstimator, TransformerMixin):
    """ Specific transformer for toad selection function
    """

    def __init__(
        self,
        empty=0.9,
        iv=0.02,
        corr=0.99,
        exclude=None
    ):
        super().__init__()
        self.model_params = {
            'empty' : empty,
            'iv' : iv,
            'corr' : corr,
            'return_drop' : True,
            'exclude' : exclude
        }

        for k, v in self.model_params.items():
            setattr(self, k, v)
        
    def fit(self, X, y):
        # First could to reload all attributes
        for key in self.model_params.keys():
            self.model_params[key] = getattr(self, key)

        # setup a series to save which columns are kept and which aren't after using the selection.select functions
        # At default, All values are False
        cols = X.columns
        self.col2select_ = pd.Series([False] * cols.size, index=cols)
        
        selected, drop_list = select(
            X, target=y, **self.model_params
        )

        # Update all freatures from selected into col2select to value True
        for i in selected.columns:
            self.col2select_[i] = True
        
        return self
    
    def transform(self, X, y=None):
        return X.loc[:, list(self.col2select_[self.col2select_].index)] 

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
        """_summary_

        Args:
            empty (number):drop the features which empty num is greater than threshold. if threshold is less than `1`, it will be use as percentage. 
            iv (float): drop the features whose IV is less than threshold. 
            corr (float): drop features that has the smallest IV in each groups which correlation is greater than threshold
            exclude (array-like): list of feature name that will not be dropped
        """
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
        """Starting Fitting 

        Args:
            X (DataFrame): features to be selected, and note X only contains features, no labels
            y (array-like): Label of the sample

        Returns:
            self
        """
        # Reset All self properties because there might be a set_params() before fit
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
        """transform X by select

        Args:
            X (DataFrame): features to be transformed

        Returns:
            DataFrame
        """
        return X.loc[:, list(self.col2select_[self.col2select_].index)] 

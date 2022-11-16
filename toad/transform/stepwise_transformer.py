import pandas as pd
from sklearn.base import (
    BaseEstimator, 
    TransformerMixin
)
from sklearn.feature_selection import VarianceThreshold

from ..selection import drop_corr, stepwise

class StepwiseTransformer4pipe(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        skip = False,
        estimator = 'ols', 
        direction = 'both', 
        criterion = 'aic',
        p_enter = 0.01, 
        p_remove = 0.01, 
        p_value_enter = 0.2, 
        intercept = False,
        max_iter = 10000,
        return_drop = False, 
        exclude = None,
        corr = 0.9       
    ):
        """Specific transformer for toad stepwise function

        Args:
            skip (bool): whether to skip this part in pipeline
            estimator (str): model to use for stats
            direction (str): direction of stepwise, support 'forward', 'backward' and 'both', suggest 'both'
            criterion (str): criterion to statistic model, support 'aic', 'bic'
            p_enter (float): threshold that will be used in 'forward' and 'both' to keep features
            p_remove (float): threshold that will be used in 'backward' to remove features
            p_value_enter (float): threshold that will be used in 'both' to remove features
            intercept (bool): if have intercept
            max_iter (int): maximum number of iterate
            return_drop (bool): if need to return features' name who has been dropped
            exclude (array-like): list of feature names that will not be dropped
            corr (float): used for drop_corr in order to avoid high correlated bined features before the stepwise
        """
        super().__init__()
        self.exclude = exclude
        # If the user has already found the necessary features and do wish all these features are kept after combiner step followed participating into the LogisticRegression Model, then there is a need to set a skip function
        self.skip = skip
        self.model_params = {
            'estimator' : estimator, 
            'direction' : direction, 
            'criterion' : criterion,
            'p_enter' : p_enter, 
            'p_remove' : p_remove , 
            'p_value_enter' : p_value_enter, 
            'intercept' : intercept,
            'max_iter' : max_iter, 
            'return_drop' : return_drop,
            'exclude' : exclude
        }
        for k, v in self.model_params.items():
            setattr(self, k, v)
        self.corr = corr
    
    def fit(self, X, y=None):
        """fit stepwise

        Args:
            X (DataFrame): features to be selected, and note X only contains features, no labels
            y (array-like): Label of the sample

        Returns:
            self
        """          
        cols = X.columns
        # if skip, then set col2select_ to True for each features
        if self.skip:
            self.col2select_ = pd.Series([True] * cols.size, index=cols)
            return self

        for key in self.model_params.keys():
            self.model_params[key] = getattr(self, key)        

        self.col2select_ = pd.Series([False] * cols.size, index=cols)    
        
        # Warning, the following step might not be very necessary, and might be deleted in future versions
        # Before Stepwise, there might be a need to do a variance threshold to filter small variance features, this might due to an insufficient bins in combiner.rule
        # This might also cause the singular-matrix ValueError in the later stepwise

        VT = VarianceThreshold(0)
        VT = VT.fit(X)
        valid_cols_from_VT = VT.get_support()
        X = X.loc[:, valid_cols_from_VT]

        # Warning, the following step might not be very necessary, and might be deleted in future versions
        # There might still be high correlated features after WOEtransformers, therefor set a drop_corr to fix this
        X, corr_drop = drop_corr(
            frame=X,
            target=y,
            threshold=self.corr,
            by='IV',
            return_drop=True,
            exclude=self.exclude
        )

        selected = stepwise(
            X, target=y, **self.model_params
        )

        for i in selected.columns:
            self.col2select_[i] = True
        
        return self        

    def transform(self, X, y = None):
        """transform X by woe

        Args:
            X (DataFrame): features to be transformed

        Returns:
            DataFrame
        """        
        return X.loc[:, list(self.col2select_[self.col2select_].index)]

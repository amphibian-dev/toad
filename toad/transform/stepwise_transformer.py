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
        super().__init__()
        self.exclude = exclude
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
        cols = X.columns
        if self.skip:
            self.col2select_ = pd.Series([True] * cols.size, index=cols)
            return self

        for key in self.model_params.keys():
            self.model_params[key] = getattr(self, key)        

        self.col2select_ = pd.Series([False] * cols.size, index=cols)    
        # 注意，在做stepwise之前，可能需要选做一次方差过滤，优先的将那些方差为0的删掉
        # 不然后面在stepwise的t值中，算XTX会出现singular-matrix的报错信息
        VT = VarianceThreshold(0)
        VT = VT.fit(X)
        valid_cols_from_VT = VT.get_support()
        X = X.loc[:, valid_cols_from_VT]

        # 接着这边尝试将特别特别高的相关性的特征给删掉，优先删掉iv小的，这里可以尝试使用toad里面的小函数
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
        return X.loc[:, list(self.col2select_[self.col2select_].index)]

import pytest
import numpy as np
import pandas as pd

from .gbdt_transformer import GBDTTransformer

np.random.seed(1)

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size = 500)
target = np.random.randint(2, size = 500)
str_feat = ab[np.random.choice(7, 500)]
uni_feat = np.ones(500)
empty_feat = feature.astype(float)
empty_feat[np.random.choice(500, 50, replace = False)] = np.nan

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
    'C': uni_feat,
    'D': empty_feat,
    'target': target,
})


def test_gbdt_transformer():
    np.random.seed(1)

    df = pd.DataFrame({
        'A': np.random.rand(500),
        'B': np.random.randint(10, size = 500),
    })
    f = GBDTTransformer().fit_transform(df, target, n_estimators = 10, max_depth = 2)
    assert f.shape == (500, 40)


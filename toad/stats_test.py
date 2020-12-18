import pytest
import numpy as np
import pandas as pd

from .stats import IV, WOE, gini, gini_cond, entropy_cond, quality, _IV, VIF


np.random.seed(1)

feature = np.random.rand(500)
target = np.random.randint(2, size = 500)
A = np.random.randint(100, size = 500)
B = np.random.randint(100, size = 500)
mask = np.random.randint(8, size = 500)

df = pd.DataFrame({
    'feature': feature,
    'target': target,
    'A': A,
    'B': B,
})


def test_woe():
    value = WOE(0.2, 0.3)
    assert value == -0.4054651081081643

def test_iv_priv():
    value, _ = _IV(df['feature'], df['target'])
    assert value == 0.010385942643745403

def test_iv():
    value = IV(df['feature'], df['target'], n_bins = 10, method = 'dt')
    assert value == 0.2735917707743619

def test_iv_return_sub():
    _, sub = IV(mask, df['target'], return_sub = True, n_bins = 10, method = 'dt')
    assert len(sub) == 8
    assert sub[4] == 0.006449386778057019

def test_iv_frame():
    res = IV(df, 'target', n_bins = 10, method = 'chi')
    assert res.loc[0, 'A'] == 0.226363832867123

def test_gini():
    value = gini(df['target'])
    assert value == 0.499352

def test_gini_cond():
    value = gini_cond(df['feature'], df['target'])
    assert value == 0.4970162601626016

def test_entropy_cond():
    value = entropy_cond(df['feature'], df['target'])
    assert value == 0.6924990371522171

def test_quality():
    result = quality(df, 'target')
    assert result.loc['feature', 'iv'] == 0.2735917707743619
    assert result.loc['A', 'gini'] == 0.49284164671885444
    assert result.loc['B', 'entropy'] == 0.6924956879070063
    assert result.loc['feature', 'unique'] == 500

def test_quality_iv_only():
    result = quality(df, 'target', iv_only = True)
    assert np.isnan(result.loc['feature', 'gini'])

def test_quality_with_merge():
    result = quality(df, 'target', n_bins = 5, method = 'chi')
    assert result.loc['feature', 'iv'] == 0.13367825777558

def test_quality_object_type_array_with_nan():
    feature = np.array([np.nan, 'A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype = 'O')[mask]

    df = pd.DataFrame({
        'feature': feature,
        'target': target,
    })
    result = quality(df)
    assert result.loc['feature', 'iv'] == 0.016379338180530334

def test_vif():
    vif = VIF(df)
    assert vif['A'] == 2.969336442640111

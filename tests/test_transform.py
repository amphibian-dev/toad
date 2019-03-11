import pytest
import numpy as np
import pandas as pd

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from toad.transform import WOETransformer, Combiner

np.random.seed(1)

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size = 500)
target = np.random.randint(2, size = 500)
str_feat = ab[np.random.choice(7, 500)]
uni_feat = np.ones(500)

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
    'C': uni_feat,
})



def test_woe_transformer():
    f = WOETransformer().fit_transform(feature, target)
    assert f[451] == -0.17061154127869285

def test_woe_transformer_with_str():
    f = WOETransformer().fit_transform(str_feat, target)
    assert f[451] == -0.2198594761130199

def test_woe_transformer_with_unknown_group():
    transer = WOETransformer().fit(str_feat, target)
    res = transer.transform(['Z'], default = 'min')
    assert res[0] == -0.2198594761130199

def test_woe_transformer_frame():
    res = WOETransformer().fit_transform(df, target)
    assert res.iloc[451, 1] == -0.2198594761130199

def test_woe_transformer_select_dtypes():
    res = WOETransformer().fit_transform(df, target, select_dtypes = 'object')
    assert res.loc[451, 'A'] == 3

def test_woe_transformer_exclude():
    res = WOETransformer().fit_transform(df, target, exclude = 'A')
    assert res.loc[451, 'A'] == 3

def test_combiner():
    f = Combiner().fit_transform(feature, target, method = 'chi')
    assert f[451] == 3

def test_combiner_with_str():
    f = Combiner().fit_transform(str_feat, target, method = 'chi')
    assert f[451] == 0

def test_combiner_unique_feature():
    f = Combiner().fit_transform(uni_feat, target, method = 'chi')
    assert f[451] == 0

def test_combiner_frame():
    res = Combiner().fit_transform(df, target)
    assert res.iloc[404, 1] == 2

def test_combiner_select_dtypes():
    res = Combiner().fit_transform(df, target, select_dtypes = 'number')
    assert res.loc[451, 'B'] == 'G'

def test_combiner_exclude():
    res = Combiner().fit_transform(df, target, exclude = 'B')
    assert res.loc[451, 'B'] == 'G'

def test_combiner_labels():
    combiner = Combiner().fit(df, target)
    res = combiner.transform(df, labels = True)
    assert res.loc[451, 'A'] == '3.[3 ~ 4)'

def test_combiner_export():
    combiner = Combiner().fit(df, target, method = 'chi', n_bins = 4)
    bins = combiner.export()
    assert isinstance(bins['B'][0], list)

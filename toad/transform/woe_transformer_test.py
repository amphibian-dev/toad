import pytest
import numpy as np
import pandas as pd

from .woe_transformer import WOETransformer, WOETransformer4pipe

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



def test_duplicated_keys():
    dup_df = df.rename(columns = {"C": "A"})
    with pytest.raises(Exception, match=r"X has duplicate keys `.*`"):
        WOETransformer().fit_transform(dup_df, target)

def test_woe_transformer():
    f = WOETransformer().fit_transform(feature, target)
    assert f[451] == pytest.approx(-0.17061154127869285)

def test_woe_transformer_with_str():
    f = WOETransformer().fit_transform(str_feat, target)
    assert f[451] == pytest.approx(-0.2198594761130199)

def test_woe_transformer_with_unknown_group():
    transer = WOETransformer().fit(str_feat, target)
    res = transer.transform(['Z'], default = 'min')
    assert res[0] == pytest.approx(-0.2198594761130199)

def test_woe_transformer_frame():
    res = WOETransformer().fit_transform(df, target)
    assert res.iloc[451, 1] == pytest.approx(-0.2198594761130199)

def test_woe_transformer_dict():
    transer = WOETransformer().fit(df, 'target')
    res = transer.transform({
        "A": 6,
        "B": "C",
        "C": 1,
        "D": 2,
    })
    assert res['B'].item() == pytest.approx(-0.09149433112609942)

def test_woe_transformer_select_dtypes():
    res = WOETransformer().fit_transform(df, target, select_dtypes = 'object')
    assert res.loc[451, 'A'] == 3

def test_woe_transformer_exclude():
    res = WOETransformer().fit_transform(df, target, exclude = 'A')
    assert res.loc[451, 'A'] == 3

def test_woe_transformer_export_single():
    transer = WOETransformer().fit(feature, target)
    t = transer.export()
    assert t[transer._default_name][5] == pytest.approx(0.3938235330926786)

def test_woe_transformer_export():
    transer = WOETransformer().fit(df, target)
    t = transer.export()
    assert t['C'][1] == 0

def test_woe_transformer_load():
    rules = {
        'A': {
            1: 0.1,
            2: 0.2,
            3: 0.3,
        }
    }

    transer = WOETransformer().load(rules)
    assert transer._rules['A']['woe'][1] == 0.2


Y = df.pop('target')
X = df.copy()


def test_duplicated_keys_transformer():
    dup_df = X.rename(columns = {"C": "A"})
    with pytest.raises(Exception, match=r"X has duplicate keys `.*`"):
        WOETransformer4pipe().fit_transform(dup_df, Y)

def test_woe_transformer_transformer():
    f = WOETransformer4pipe().fit_transform(feature, target)
    assert f[451] == pytest.approx(-0.17061154127869285)

def test_woe_transformer_with_str_transformer():
    f = WOETransformer4pipe().fit_transform(str_feat, target)
    assert f[451] == pytest.approx(-0.2198594761130199)

def test_woe_transformer_with_unknown_group_transformer():
    transer = WOETransformer4pipe(default = 'min').fit(str_feat, target)
    res = transer.transform(['Z'])
    assert res[0] == pytest.approx(-0.2198594761130199)

def test_woe_transformer_frame_transformer():
    res = WOETransformer4pipe().fit_transform(X, Y)
    assert res.iloc[451, 1] == pytest.approx(-0.2198594761130199)

def test_woe_transformer_dict_transformer():
    transer = WOETransformer4pipe().fit(X, Y)
    res = transer.transform({
        "A": 6,
        "B": "C",
        "C": 1,
        "D": 2,
    })
    assert res['B'].item() == pytest.approx(-0.09149433112609942)

def test_woe_transformer_select_dtypes_transformer():
    res = WOETransformer4pipe(select_dtypes = 'object').fit_transform(X, Y)
    assert res.loc[451, 'A'] == 3

def test_woe_transformer_exclude_transformer():
    res = WOETransformer4pipe(exclude = 'A').fit_transform(X, Y)    
    assert res.loc[451, 'A'] == 3

def test_woe_transformer_export_single_transformer():
    transer = WOETransformer4pipe().fit(feature, target)
    t = transer.export()
    assert t[transer._default_name][5] == pytest.approx(0.3938235330926786)

def test_woe_transformer_export_transformer():
    transer = WOETransformer4pipe().fit(X, Y)
    t = transer.export()
    assert t['C'][1] == 0

def test_woe_transformer_load_transformer():
    rules = {
        'A': {
            1: 0.1,
            2: 0.2,
            3: 0.3,
        }
    }

    transer = WOETransformer4pipe().woe.load(rules)
    assert transer._rules['A']['woe'][1] == 0.2
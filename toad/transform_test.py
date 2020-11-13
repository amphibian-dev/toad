import pytest
import numpy as np
import pandas as pd

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from .transform import WOETransformer, Combiner, GBDTTransformer

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

def test_woe_transformer_export_single():
    transer = WOETransformer().fit(feature, target)
    t = transer.export()
    assert t[transer._default_name][5] == 0.3938235330926786

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
    assert res.loc[451, 'A'] == '03.[3 ~ 4)'

def test_combiner_single_feature():
    combiner = Combiner().fit(df['A'], method = 'step', n_bins = 5)
    res = combiner.transform(df['A'])
    assert res[451] == 1

def test_combiner_export():
    combiner = Combiner().fit(df, target, method = 'chi', n_bins = 4)
    bins = combiner.export()
    assert isinstance(bins['B'][0], list)

def test_combiner_update():
    combiner = Combiner().fit(df, target, method = 'chi', n_bins = 4)
    combiner.update({'A': [1,2,3,4,5,6]})
    bins = combiner.export()
    assert len(bins['A']) == 6

def test_combiner_step():
    combiner = Combiner().fit(df['A'], method = 'step', n_bins = 4)
    bins = combiner.export()
    assert bins['A'][1] == 4.5

def test_combiner_target_in_frame():
    combiner = Combiner().fit(df, 'target', n_bins = 4)
    bins = combiner.export()
    assert bins['A'][1] == 6

def test_combiner_target_in_frame_kwargs():
    combiner = Combiner().fit(df, y = 'target', n_bins = 4)
    bins = combiner.export()
    assert bins['A'][1] == 6

def test_combiner_empty_separate():
    combiner = Combiner()
    bins = combiner.fit_transform(df, 'target', n_bins = 4, empty_separate = True)
    mask = pd.isna(df['D'])
    assert (bins['D'][~mask] != 4).all()

def test_combiner_labels_with_empty():
    combiner = Combiner().fit(df, 'target', n_bins = 4, empty_separate = True)
    res = combiner.transform(df, labels = True)
    assert res.loc[2, 'D'] == '04.nan'

def test_gbdt_transformer():
    np.random.seed(1)

    df = pd.DataFrame({
        'A': np.random.rand(500),
        'B': np.random.randint(10, size = 500),
    })
    f = GBDTTransformer().fit_transform(df, target, n_estimators = 10, max_depth = 2)
    assert f.shape == (500, 40)

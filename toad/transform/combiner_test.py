import pytest
import numpy as np
import pandas as pd

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from .combiner import Combiner, CombinerTransformer4pipe

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

Y = df['target']
X = df.iloc[:, :-1]

def test_combiner_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi')
    f = combiner_T.fit_transform(feature, target)
    assert f[451] == 3

def test_combiner_with_str_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi')
    f = combiner_T.fit_transform(str_feat, target)
    assert f[451] == 0

def test_combiner_unique_feature_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi')
    f = combiner_T.fit_transform(uni_feat, target)
    assert f[451] == 0

def test_combiner_XY_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi')
    res = combiner_T.fit_transform(X, Y)
    assert res.iloc[404, 1] == 2

def test_combiner_select_dtypes_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', select_dtypes = 'number')
    res = combiner_T.fit_transform(X, Y)
    assert res.loc[451, 'B'] == 'G'

def test_combiner_exclude_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', exclude = 'B')
    res = combiner_T.fit_transform(X, Y)
    assert res.loc[451, 'B'] == 'G'

def test_combiner_labels_transformer():
    combiner_T = CombinerTransformer4pipe(labels=True)
    combiner_T = combiner_T.fit(X, Y)
    res = combiner_T.transform(X)
    assert res.loc[451, 'A'] == '03.[3 ~ 4)'

def test_combiner_single_feature_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'step', n_bins = 5)
    combiner_T = combiner_T.fit(X['A'], Y)
    res = combiner_T.transform(X['A'])
    assert res[451] == 1

def test_combiner_export_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', n_bins = 4)
    combiner_T = combiner_T.fit(X, Y)
    bins = combiner_T.export()
    assert isinstance(bins['B'][0], list)

def test_combiner_update_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', n_bins = 4)
    combiner_T = combiner_T.fit(X, Y)
    combiner_T.combiner.update({'A': [1,2,3,4,5,6]})
    bins = combiner_T.export()
    assert len(bins['A']) == 6

def test_combiner_step_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'step', n_bins = 4)
    combiner_T = combiner_T.fit(X['A'], Y)
    bins = combiner_T.export()
    assert bins['A'][1] == 4.5

def test_combiner_target_in_frame_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', n_bins = 4)
    combiner_T = combiner_T.fit(X, Y)
    bins = combiner_T.export()
    assert bins['A'][1] == 6

def test_combiner_target_in_frame_kwargs_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', n_bins = 4)
    combiner_T = combiner_T.fit(X, Y)
    bins = combiner_T.export()
    assert bins['A'][1] == 6

def test_combiner_empty_separate_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', n_bins = 4, empty_separate = True)
    bins = combiner_T.fit_transform(X, Y)
    mask = pd.isna(df['D'])
    assert (bins['D'][~mask] != 4).all()

def test_combiner_labels_with_empty_transformer():
    combiner_T = CombinerTransformer4pipe(method = 'chi', n_bins = 4, empty_separate = True, labels=True)
    combiner_T = combiner_T.fit(X, Y)
    res = combiner_T.transform(X)
    assert res.loc[2, 'D'] == '04.nan'
import pytest
import numpy as np
import pandas as pd

try:
    import pyximport
except ModuleNotFoundError:
    pyximport = None

if pyximport is not None:
    pyximport.install(setup_args={"include_dirs": np.get_include()})

from .merge import merge, ChiMerge, DTMerge, QuantileMerge, StepMerge, KMeansMerge

np.random.seed(1)
feature = np.random.rand(500)
target = np.random.randint(2, size = 500)
A = np.random.randint(100, size = 500)
B = np.random.randint(3, size = 500)

df = pd.DataFrame({
    'feature': feature,
    'target': target,
    'A': A,
})


def _bin_counts(feature, splits):
    bins = np.digitize(feature, splits, right = False)
    return np.bincount(bins.astype(int))


def _make_constraint_case_a():
    counts = np.array([41, 34, 22, 24, 10, 11, 8, 17, 50, 41, 55])
    probs = np.array([
        0.19128, 0.24629567, 0.21073853, 0.2514205, 0.37637079,
        0.50289282, 0.41724785, 0.64468458, 0.65278623, 0.76874129,
        0.86190187,
    ])
    rng = np.random.default_rng(0)

    feature = np.concatenate([
        np.full(count, i, dtype = float)
        for i, count in enumerate(counts)
    ])
    target = np.concatenate([
        rng.binomial(1, prob, size = count)
        for count, prob in zip(counts, probs)
    ])

    return feature, target


def _make_constraint_case_b():
    counts = np.array([26, 21, 12, 13, 3, 4, 2, 8, 32, 26, 36, 21])
    probs = np.array([
        0.23941619, 0.01, 0.01, 0.17079965, 0.38553792, 0.01,
        0.49715076, 0.37354508, 0.55809198, 0.67751184, 0.80492179,
        0.99,
    ])
    rng = np.random.default_rng(0)

    feature = np.concatenate([
        np.full(count, i, dtype = float)
        for i, count in enumerate(counts)
    ])
    target = np.concatenate([
        rng.binomial(1, prob, size = count)
        for count, prob in zip(counts, probs)
    ])

    return feature, target



def test_chimerge():
    splits = ChiMerge(feature, target, n_bins = 10)
    assert len(splits) == 9

def test_chimerge_bins_not_enough():
    splits = ChiMerge(B, target, n_bins = 10)
    assert len(splits) == 2

def test_chimerge_bins_with_min_samples():
    splits = ChiMerge(feature, target, min_samples = 0.02)
    assert len(splits) == 10

def test_chimerge_constraint_mode_all_matches_any_with_only_n_bins():
    case_feature, case_target = _make_constraint_case_a()
    splits_any = ChiMerge(case_feature, case_target, n_bins = 4, constraint_mode = 'any')
    splits_all = ChiMerge(case_feature, case_target, n_bins = 4, constraint_mode = 'all')

    np.testing.assert_array_equal(splits_any, splits_all)

def test_chimerge_constraint_mode_all_matches_any_with_only_min_samples():
    case_feature, case_target = _make_constraint_case_a()
    splits_any = ChiMerge(case_feature, case_target, min_samples = 0.051, constraint_mode = 'any')
    splits_all = ChiMerge(case_feature, case_target, min_samples = 0.051, constraint_mode = 'all')

    np.testing.assert_array_equal(splits_any, splits_all)

def test_chimerge_constraint_mode_all_requires_n_bins_and_min_samples():
    case_feature, case_target = _make_constraint_case_a()
    min_samples = 0.05
    splits_any = ChiMerge(case_feature, case_target, n_bins = 3, min_samples = min_samples, constraint_mode = 'any')
    splits_all = ChiMerge(case_feature, case_target, n_bins = 3, min_samples = min_samples, constraint_mode = 'all')

    counts_any = _bin_counts(case_feature, splits_any)
    counts_all = _bin_counts(case_feature, splits_all)
    threshold = len(case_feature) * min_samples

    assert len(splits_any) + 1 > 3
    assert counts_any.min() > threshold
    assert len(splits_all) + 1 <= 3
    assert counts_all.min() >= threshold

def test_chimerge_constraint_mode_all_keeps_merging_when_small_bins_remain():
    case_feature, case_target = _make_constraint_case_b()
    min_samples = 0.12
    splits_any = ChiMerge(case_feature, case_target, n_bins = 3, min_samples = min_samples, constraint_mode = 'any')
    splits_all = ChiMerge(case_feature, case_target, n_bins = 3, min_samples = min_samples, constraint_mode = 'all')

    counts_any = _bin_counts(case_feature, splits_any)
    counts_all = _bin_counts(case_feature, splits_all)
    threshold = len(case_feature) * min_samples

    assert len(splits_any) + 1 <= 3
    assert counts_any.min() < threshold
    assert len(splits_all) + 1 <= 3
    assert counts_all.min() >= threshold

def test_chimerge_constraint_mode_all_ignores_min_threshold_stop():
    case_feature, case_target = _make_constraint_case_a()
    splits_any = ChiMerge(
        case_feature,
        case_target,
        n_bins = 3,
        min_samples = 0.05,
        min_threshold = -1.0,
        constraint_mode = 'any',
    )
    splits_all = ChiMerge(
        case_feature,
        case_target,
        n_bins = 3,
        min_samples = 0.05,
        min_threshold = -1.0,
        constraint_mode = 'all',
    )

    assert len(splits_any) + 1 > 3
    assert len(splits_all) + 1 <= 3

def test_dtmerge():
    splits = DTMerge(feature, target, n_bins = 10)
    assert len(splits) == 9

def test_quantilemerge():
    splits = QuantileMerge(feature, n_bins = 10)
    assert len(splits) == 9

def test_quantilemerge_not_enough():
    splits = QuantileMerge(B, n_bins = 10)
    assert len(splits) == 2

def test_stepmerge():
    splits = StepMerge(feature, n_bins = 10)
    assert len(splits) == 9

def test_kmeansmerge():
    splits = KMeansMerge(feature, n_bins = 10)
    assert len(splits) == 9

def test_merge():
    res = merge(feature, target = target, method = 'chi', n_bins = 10)
    assert len(np.unique(res)) == 10

def test_merge_frame():
    res = merge(df, target = 'target', method = 'chi', n_bins = 10)
    assert len(np.unique(res['A'])) == 10

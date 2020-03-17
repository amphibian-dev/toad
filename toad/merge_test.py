import pytest
import numpy as np
import pandas as pd

import pyximport

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



def test_chimerge():
    splits = ChiMerge(feature, target, n_bins = 10)
    assert len(splits) == 9

def test_chimerge_bins_not_enough():
    splits = ChiMerge(B, target, n_bins = 10)
    assert len(splits) == 2

def test_chimerge_bins_with_min_samples():
    splits = ChiMerge(feature, target, min_samples = 0.02)
    assert len(splits) == 10

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

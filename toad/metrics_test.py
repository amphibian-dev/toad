import pytest
import numpy as np
import pandas as pd

from .metrics import KS, KS_bucket, F1, PSI, AUC, matrix

np.random.seed(1)

feature = np.random.rand(500)
target = np.random.randint(2, size = 500)
base_feature = np.random.rand(500)

test_df = pd.DataFrame({
    'A': np.random.rand(500),
    'B': np.random.rand(500),
})
base_df = pd.DataFrame({
    'A': np.random.rand(500),
    'B': np.random.rand(500),
})


def test_KS():
    result = KS(feature, target)
    assert result == 0.05536775661256989

def test_KS_bucket():
    result = KS_bucket(feature, target)
    assert result.loc[4, 'ks'] == -0.028036335090276976

def test_KS_bucket_use_step():
    result = KS_bucket(feature, target, method = 'step', clip_q = 0.01)
    assert result.loc[4, 'ks'] == -0.0422147102645028

def test_KS_bucket_for_all_score():
    result = KS_bucket(feature, target, bucket = False)
    assert len(result) == 500

def test_KS_bucket_return_splits():
    result, splits = KS_bucket(feature, target, return_splits = True)
    assert len(splits) == 9

def test_KS_bucket_use_split_pointers():
    result = KS_bucket(feature, target, bucket = [0.2, 0.6])
    assert len(result) == 3

def test_KS_bucket_with_lift():
    result = KS_bucket(feature, target)
    assert result.loc[3, 'lift'] == 1.003861003861004

def test_F1():
    result, split = F1(feature, target, return_split = True)
    assert result == 0.6844207723035951

def test_F1_split():
    result = F1(feature, target, split = 0.5)
    assert result == 0.51417004048583

def test_AUC():
    result = AUC(feature, target)
    assert result == 0.5038690142424582

def test_AUC_with_curve():
    auc, fpr, tpr, thresholds = AUC(feature, target, return_curve = True)
    assert thresholds[200] == 0.15773006987053328

def test_PSI():
    result = PSI(feature, base_feature, combiner = [0.3, 0.5, 0.7])
    assert result == 0.018630024627491467

def test_PSI_frame():
    result = PSI(
        test_df,
        base_df,
        combiner = {
            'A': [0.3, 0.5, 0.7],
            'B': [0.4, 0.8],
        },
    )

    assert result['B'] == 0.014528279995858708

def test_PSI_return_frame():
    result, frame = PSI(
        test_df,
        base_df,
        combiner = {
            'A': [0.3, 0.5, 0.7],
            'B': [0.4, 0.8],
        },
        return_frame = True,
    )

    assert frame.loc[4, 'test'] == 0.38

def test_matrix():
    df = matrix(feature, target, splits = 0.5)
    assert df.iloc[0,1] == 133
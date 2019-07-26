import pytest
import numpy as np
import pandas as pd

from toad.metrics import KS, KS_bucket, F1

np.random.seed(1)

feature = np.random.rand(500)
target = np.random.randint(2, size = 500)


def test_KS():
    result = KS(feature, target)
    assert result == 0.055367756612569874

def test_KS_bucket():
    result = KS_bucket(feature, target)
    assert result.loc[4, 'ks'] == 0.028036335090276976

def test_KS_bucket_use_step():
    result = KS_bucket(feature, target, method = 'step', clip_q = 0.01)
    assert result.loc[4, 'ks'] == 0.0422147102645028

def test_KS_bucket_for_all_score():
    result = KS_bucket(feature, target, bucket = False)
    assert len(result) == 500

def test_F1():
    result, split = F1(feature, target, return_split = True)
    assert result == 0.6844207723035951

def test_F1_split():
    result = F1(feature, target, split = 0.5)
    assert result == 0.51417004048583

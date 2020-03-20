import pytest
import numpy as np
import pandas as pd


from .partition import TimePartition, ValuePartition


np.random.seed(1)

ab = np.array(list('ABCDEFG'))

history = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, 400, size = 500)
open_time = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, size = 500)
A = ab[np.random.choice(7, 500)]
B = np.random.randint(10, size = 500).astype(float)
B[np.random.choice(500, 10)] = np.nan


df = pd.DataFrame({
    'history': history,
    'open_time': open_time,
    'A': A,
    'B': B,
})


def test_timepartition():
    tp = TimePartition('open_time', 'history', ['90d', '180d'])
    mask, suffix = next(tp.partition(df))
    assert mask.sum() == 93


def test_timepartition_all():
    tp = TimePartition('open_time', 'history', ['all'])
    mask, suffix = next(tp.partition(df))
    assert mask.sum() == 500

def test_valuepartition():
    vp = ValuePartition('A')
    mask, suffix = next(vp.partition(df))
    assert mask.sum() == 67

def test_valuepartition_with_na():
    vp = ValuePartition('B')
    s = 0
    for mask, suffix in vp.partition(df):
        s += mask.sum()
    
    assert s == 500
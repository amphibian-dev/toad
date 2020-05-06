import pytest
import numpy as np
import pandas as pd


from .process import Processing, Mask, F


np.random.seed(1)

history = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, 400, size = 500)
open_time = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, size = 500)
A = np.random.randint(10, size = 500)
B = np.random.rand(500)
B[np.random.choice(500, 10)] = np.nan


df = pd.DataFrame({
    'history': history,
    'open_time': open_time,
    'A': A,
    'B': B,
})


def test_mask():
    m = Mask('A') > 3
    assert m.replay(df).sum() == (A > 3).sum()


def test_mask_without_name():
    m = Mask() > 3
    assert m.replay(A).sum() == (A > 3).sum()

def test_mask_isin():
    m = Mask('A').isin([1,2,3])
    assert m.replay(df).sum() == df['A'].isin([1,2,3]).sum()

def test_mask_isna():
    m = Mask('A').isna()
    assert m.replay(df).sum() == df['A'].isna().sum()

def test_f():
    assert F(len)(A)[0] == 500

def test_processing():
    res = (
        Processing(df)
        .groupby('open_time')
        .apply({'A': ['min', 'mean']})
        .apply({'B': [
            {
                'f': 'size',
                'mask': Mask('A') > 1,
            },
            {
                'f': len,
            },
        ]})
        .exec()
    )
    
    assert res.size == 120 and res.loc['2020-02-29', 'B_size'] == 23


def test_processing_with_partition():
    from .partition import ValuePartition
    res = (
        Processing(df)
        .groupby('open_time')
        .partitionby(ValuePartition('A'))
        .apply({'B': ['mean', 'size']})
        .exec()
    )

    assert res.size == 600 and res.loc['2020-02-29', 'B_size_1'] == 2
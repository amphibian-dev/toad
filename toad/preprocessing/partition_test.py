import pytest
import numpy as np
import pandas as pd


from .partition import TimePartition


np.random.seed(1)

history = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, 400, size = 500)
open_time = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, size = 500)


df = pd.DataFrame({
    'history': history,
    'open_time': open_time,
})


def test_timepartition():
    tp = TimePartition('open_time', 'history', ['90d', '180d'])
    mask, suffix = next(tp.partition(df))
    assert mask.sum() == 93


def test_timepartition_all():
    tp = TimePartition('open_time', 'history', ['all'])
    mask, suffix = next(tp.partition(df))
    assert mask.sum() == 500

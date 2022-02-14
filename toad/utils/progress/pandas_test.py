import numpy as np
import pandas as pd
from .pandas import pandas_enable, pandas_disable



def test_pandas_with():
    assert hasattr(pd.DataFrame, 'progress') == False
    assert hasattr(pd.Series, 'progress') == False
    with pandas_enable():
        assert hasattr(pd.DataFrame, 'progress') == True
        assert hasattr(pd.Series, 'progress') == True
    assert hasattr(pd.DataFrame, 'progress') == False
    assert hasattr(pd.Series, 'progress') == False

def test_pandas_disable():
    assert hasattr(pd.DataFrame, 'progress') == False
    assert hasattr(pd.Series, 'progress') == False
    pandas_enable()
    assert hasattr(pd.DataFrame, 'progress') == True
    assert hasattr(pd.Series, 'progress') == True
    pandas_disable()
    assert hasattr(pd.DataFrame, 'progress') == False
    assert hasattr(pd.Series, 'progress') == False

def test_dataframe_apply():
    df = pd.DataFrame({
        "A": np.random.rand(1000),
        "B": np.random.randint(10, size = (1000,))
    })

    with pandas_enable():
        res = df.progress.apply(lambda x: x + 1)

def test_dataframe_apply_axis():
    df = pd.DataFrame({
        "A": np.random.rand(1000),
        "B": np.random.randint(10, size = (1000,))
    })

    with pandas_enable():
        res = df.progress.apply(lambda x: x + 1, axis = 1)
    

def test_series_apply():
    series = pd.Series(np.random.rand(2000))

    with pandas_enable():
        res = series.progress.apply(lambda x: x + 1)


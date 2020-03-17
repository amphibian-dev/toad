import pytest
import numpy as np
import pandas as pd


from .partition import TimePartition


np.random.seed(1)

history = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, 400, size = 500)
open_time = np.full(500, np.datetime64('2020-03-01')) - np.random.randint(30, size = 500)
A = np.random.randint(10, size = 500)
B = np.random.rand(500)


df = pd.DataFrame({
    'history': history,
    'open_time': open_time,
    'A': A,
    'B': B,
})
import pytest
import numpy as np
import pandas as pd

from .select_transformer import SelectTransformer4pipe

np.random.seed(1)

LENGTH = 500

A = np.random.rand(LENGTH)
A[np.random.choice(LENGTH, 20, replace = False)] = np.nan

B = np.random.randint(100, size = LENGTH)
C = A + np.random.normal(0, 0.2, LENGTH)
D = A + np.random.normal(0, 0.1, LENGTH)

E = np.random.rand(LENGTH)
E[np.random.choice(LENGTH, 480, replace = False)] = np.nan

F = B + np.random.normal(0, 10, LENGTH)

target = np.random.randint(2, size = LENGTH)

frame = pd.DataFrame({
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'F': F,
})

frame['target'] = target
Y = frame.pop('target')
X = frame.copy()

def test_select():
    select_T = SelectTransformer4pipe(empty = 0.8, iv = 0.2, corr = 0.7)
    df = select_T.fit_transform(X, Y)
    assert ['D', 'F'] == df.columns.tolist()

def test_select_exclude():
    select_T = SelectTransformer4pipe(empty = 0.8, iv = 0.2, corr = 0.7, exclude = ['A'])
    df = select_T.fit_transform(X, Y)
    assert ['A', 'D', 'F'] == df.columns.tolist()




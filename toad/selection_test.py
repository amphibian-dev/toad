import pytest
import numpy as np
import pandas as pd

from .selection import drop_empty, drop_var, drop_corr, drop_iv, drop_vif, select, stepwise

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


def test_drop_empty():
    df = drop_empty(frame, threshold = 0.8)
    assert 'E' not in df

def test_drop_var():
    df = drop_var(frame, threshold = 0.1)
    assert 'A' not in df

def test_drop_var_exclude():
    df = drop_var(frame, threshold = 0.1, exclude = 'A')
    assert 'A' in df

def test_drop_corr():
    df = drop_corr(frame, target = 'target')
    assert ['D', 'E', 'F', 'target'] == df.columns.tolist()

def test_drop_iv():
    df = drop_iv(frame, target = 'target', threshold = 0.25)
    assert 'B' not in df

def test_select():
    df = select(frame, target = 'target', empty = 0.8, iv = 0.2, corr = 0.7)
    assert ['D', 'F', 'target'] == df.columns.tolist()

def test_select_exclude():
    df = select(frame, target = 'target', empty = 0.8, iv = 0.2, corr = 0.7, exclude = ['A'])
    assert ['A', 'D', 'F', 'target'] == df.columns.tolist()

def test_stepwise():
    df = stepwise(frame.fillna(-1), target = 'target')
    assert ['C', 'E', 'F', 'target'] == df.columns.tolist()

def test_stepwise_backward():
    df = stepwise(frame.fillna(-1), target = 'target', direction = 'backward')
    assert ['C', 'E', 'F', 'target'] == df.columns.tolist()

def test_stepwise_forward():
    df = stepwise(frame.fillna(-1), target = 'target', direction = 'forward')
    assert ['C', 'E', 'F', 'target'] == df.columns.tolist()

def test_stepwise_exclude():
    df = stepwise(frame.fillna(-1), target = 'target', exclude = 'B')
    assert ['B', 'C', 'E', 'F', 'target'] == df.columns.tolist()

def test_stepwise_return_drop():
    df, drop_list = stepwise(frame.fillna(-1), target = 'target', return_drop = True)
    assert ['B', 'A', 'D'] == drop_list

def test_stepwise_lr():
    df = stepwise(frame.fillna(-1), target = 'target', estimator = 'lr', direction = 'forward')
    assert ['C', 'target'] == df.columns.tolist()

def test_stepwise_ks():
    df = stepwise(frame.fillna(-1), target = 'target', criterion = 'ks', direction = 'forward')
    assert ['A', 'C', 'target'] == df.columns.tolist()

def test_stepwise_zero():
    df = pd.DataFrame({
        'X': np.zeros(500),
        'Z': np.random.rand(500),
        'Y': np.random.randint(2, size = 500),
    })
    df = stepwise(df, target = 'Y')
    assert set(['Z', 'Y']) == set(df.columns.tolist())

def test_stepwise_forward_when_best_is_first():
    df = frame[['E', 'F', 'B', 'A', 'D', 'C', 'target']]
    df = stepwise(df.fillna(-1), target = 'target', direction = 'forward')
    assert ['E', 'F', 'C', 'target'] == df.columns.tolist()

def test_drop_vif():
    df = drop_vif(frame.fillna(-1), exclude = 'target')
    assert ['C', 'F', 'target'] == df.columns.tolist()

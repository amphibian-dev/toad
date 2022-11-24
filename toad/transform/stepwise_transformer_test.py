import pytest
import numpy as np
import pandas as pd

from .stepwise_transformer import StepwiseTransformer4pipe

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
Y = frame['target']
X = frame.iloc[:, :-1]


def test_stepwise():
    stepwise_T = StepwiseTransformer4pipe()
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    assert ['C', 'E', 'F'] == df.columns.tolist()

def test_stepwise_backward():
    stepwise_T = StepwiseTransformer4pipe(direction = 'backward')
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    assert ['C', 'E', 'F'] == df.columns.tolist()

def test_stepwise_forward():
    stepwise_T = StepwiseTransformer4pipe(direction = 'backward')
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    assert ['C', 'E', 'F'] == df.columns.tolist()

def test_stepwise_exclude():
    stepwise_T = StepwiseTransformer4pipe(exclude = 'B')
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    assert ['B', 'C', 'E', 'F'] == df.columns.tolist()

def test_stepwise_return_drop():
    stepwise_T = StepwiseTransformer4pipe(return_drop = True)
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    drop_list = list(stepwise_T.col2select_[~stepwise_T.col2select_].index)
    assert ['A', 'B', 'D'] == drop_list

def test_stepwise_lr():
    stepwise_T = StepwiseTransformer4pipe(estimator = 'lr', direction = 'forward')
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    assert ['C'] == df.columns.tolist()

def test_stepwise_ks():
    stepwise_T = StepwiseTransformer4pipe(criterion = 'ks', direction = 'forward')
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    assert ['A', 'C'] == df.columns.tolist()

def test_stepwise_zero():
    df = pd.DataFrame({
        'X': np.zeros(500),
        'Z': np.random.rand(500),
        'Y': np.random.randint(2, size = 500),
    })
    stepwise_T = StepwiseTransformer4pipe()
    df = stepwise_T.fit_transform(df[['X', 'Z']], df['Y'])
    assert set(['Z']) == set(df.columns.tolist())

def test_stepwise_forward_when_best_is_first():
    X = frame[['E', 'F', 'B', 'A', 'D', 'C']]
    stepwise_T = StepwiseTransformer4pipe(direction = 'forward')
    df = stepwise_T.fit_transform(X.fillna(-1), Y)
    assert ['E', 'F', 'C'] == df.columns.tolist()
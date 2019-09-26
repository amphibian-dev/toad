import pytest
import numpy as np
import pandas as pd

from toad.selection import drop_empty, drop_corr, drop_iv, drop_vif, select, stepwise

from generate_data import frame



def test_drop_empty():
    df = drop_empty(frame, threshold = 0.8)
    assert 'E' not in df

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

def test_drop_vif():
    df = drop_vif(frame.fillna(-1), exclude = 'target')
    assert ['C', 'F', 'target'] == df.columns.tolist()

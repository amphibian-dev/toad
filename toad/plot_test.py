import pytest
import numpy as np
import pandas as pd

from .plot import (
    badrate_plot,
    corr_plot,
    proportion_plot,
    roc_plot,
    bin_plot,
)

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


def test_badrate_plot():
    g = badrate_plot(
        frame,
        x = 'A',
        target = 'target',
        return_counts = True,
        return_proportion = True,
    )

def test_badrate_plot_y_axis():
    g = badrate_plot(
        frame,
        x = 'A',
        target = 'target',
    )
    bottom, _ = g.get_ylim()
    assert bottom == 0

def test_corr_plot():
    g = corr_plot(frame)


def test_proportion_plot():
    g = proportion_plot(x = frame['target'])


def test_roc_plot():
    g = roc_plot(frame['B'], frame['target'])


def test_bin_plot():
    g = bin_plot(frame, x = 'B', target = 'target')
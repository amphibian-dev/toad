import pytest
import numpy as np
import pandas as pd

from toad.plot import (
    badrate_plot,
    corr_plot,
    proportion_plot,
    roc_plot,
    bin_plot,
)

from generate_data import frame


def test_badrate_plot():
    g = badrate_plot(
        frame,
        x = 'A',
        target = 'target',
        return_counts = True,
        return_proportion = True,
    )

def test_corr_plot():
    g = corr_plot(frame)


def test_proportion_plot():
    g = proportion_plot(x = frame['target'])


def test_roc_plot():
    g = roc_plot(frame['B'], frame['target'])


def test_bin_plot():
    g = bin_plot(frame, x = 'B', target = 'target')
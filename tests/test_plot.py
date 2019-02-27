import pytest
import numpy as np
import pandas as pd

from toad.plot import badrate_plot, corr_plot

from generate_data import frame


def test_badrate_plot():
    g = badrate_plot(frame, x = 'A', target = 'target', return_counts = True)

def test_corr_plot():
    g = corr_plot(frame)

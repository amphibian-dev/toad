import pytest
import numpy as np
import pandas as pd

from .decorator import frame_exclude

np.random.seed(1)


def func():
    "This is a doc for method"


def test_decorator_doc():
    f = frame_exclude(func)

    assert f.__doc__ == 'This is a doc for method'
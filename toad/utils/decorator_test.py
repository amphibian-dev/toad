import pytest
import numpy as np
import pandas as pd

from .decorator import Decorator, frame_exclude

np.random.seed(1)


def func():
    "This is a doc for method"
    pass


def test_decorator_doc():
    f = frame_exclude(func)

    assert f.__doc__ == 'This is a doc for method'


def test_decorator_init_func():
    class a(Decorator):
        def setup_func(self, func):
            return sum
    
    f = a(func)

    assert f([10, 20]) == 30

def test_decorator_inherit():
    class a(Decorator):
        bias = 0
        def wrapper(self, *args, a = 0, **kwargs):
            return self.call(a + self.bias)
    
    class b(a):
        def wrapper(self, *args, b = 0, **kwargs):
            a = super().wrapper(*args, **kwargs)
            b = self.call(b)
            return a + b
    
    f = b(bias = 2)(lambda x: x+1)
    assert f(a = 1, b = 2) == 7


import pandas as pd
from .func import save_json
from functools import WRAPPER_ASSIGNMENTS



class Decorator:
    """base decorater class
    """
    _fn = None
    _cls = None
    is_class = False

    def __init__(self, *args, is_class = False, **kwargs):
        self.is_class = is_class

        if len(args) == 1 and callable(args[0]):
            self._fn = args[0]
        else:
            self.setup(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            self._fn = args[0]
            return self

        if self.is_class:
            self._cls = args[0]
            args = args[1:]
        
        return self.wrapper(*args, **kwargs)
    

    def __get__(self, instance, type = None):
        self.is_class = True

        def func(*args, **kwargs):
            return self.__call__(instance, *args, **kwargs)

        return func


    def __getattribute__(self, name):
        if name in WRAPPER_ASSIGNMENTS:
            return getattr(self._fn, name)
        
        return object.__getattribute__(self, name)


    def setup(self, *args, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
    def call(self, *args, **kwargs):
        if self._cls:
            args = (self._cls, *args)

        return self._fn(*args, **kwargs)

    def wrapper(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class frame_exclude(Decorator):
    """decorator for exclude columns
    """

    def wrapper(self, X, *args, exclude = None, **kwargs):
        if exclude is not None and isinstance(X, pd.DataFrame):
            X = X.drop(columns = exclude)

        return self.call(X, *args, **kwargs)


class select_dtypes(Decorator):
    """ decorator for select frame by dtypes
    """

    def wrapper(self, X, *args, select_dtypes = None, **kwargs):
        if select_dtypes is not None and isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include = select_dtypes)

        return self.call(X, *args, **kwargs)


class save_to_json(Decorator):
    """support save result to json file
    """
    def wrapper(self, *args, to_json = None, **kwargs):
        res = self.call(*args, **kwargs)

        if to_json is None:
            return res

        save_json(res, to_json)

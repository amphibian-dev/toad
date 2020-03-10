import numpy as np
import pandas as pd
from .func import save_json, read_json
from functools import wraps, WRAPPER_ASSIGNMENTS



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
        self._cls = instance

        @wraps(self._fn)
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
        if self._cls is not None:
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

        if to_json is not None:
            save_json(res, to_json)

        return res


class load_from_json(Decorator):
    """support load data from json file
    """
    require_first = False

    def wrapper(self, *args, from_json = None, **kwargs):
        if from_json is not None:
            obj = read_json(from_json)
            args = (obj, *args)
        
        elif self.require_first and len(args) > 0 and isinstance(args[0], str):
            obj = read_json(args[0])
            args = (obj, *args[1:])

        return self.call(*args, **kwargs)


class support_dataframe(Decorator):
    """decorator for supporting dataframe
    """
    require_target = True
    target = 'target'

    def wrapper(self, frame, *args, **kwargs):
        if not isinstance(frame, pd.DataFrame):
            return self.call(frame, *args, **kwargs)

        frame = frame.copy()
        if self.require_target and isinstance(args[0], str):
            target = frame.pop(args[0])
            args = (target,) + args[1:]
        elif self.target in kwargs and isinstance(kwargs[self.target], str):
            kwargs[self.target] = frame.pop(kwargs[self.target])

        res = dict()
        for col in frame:
            r = self.call(frame[col], *args, **kwargs)

            if not isinstance(r, np.ndarray):
                r = [r]

            res[col] = r
        return pd.DataFrame(res)

class proxy_docstring(Decorator):
    method_name = None
    
    def __get__(self, *args):
        func = super().__get__(*args)
        
        if self.method_name is not None and hasattr(self._cls, self.method_name):
            setattr(func, '__doc__', getattr(self._cls, self.method_name).__doc__)
        
        return func
import pandas as pd
from .func import save_json


class Decorator:
    """base decorater class
    """
    _fn = None
    _cls = None

    def __init__(self, *args, **kwargs):

        if len(args) == 1 and callable(args[0]):
            self._fn = args[0]
        else:
            self.setup(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # print('------------', self.a, args)
        if self._fn is None:
            self._fn = args[0]
            return self
        
        print('~~~~~', args)
        print('-----', args[0], self._cls)
        if len(args) > 0 and args[0] is self._cls:
            args = args[1:]
        
        print('-----', len(args))

        args, kwargs = self.before(*args, **kwargs)
        
        if self._cls:
            res = self._fn(self._cls, *args, **kwargs)
        else:
            res = self._fn(*args, **kwargs)
        
        return self.after(res, *args, **kwargs)
    
    def __get__(self, instance, type = None):
        self._cls = instance
        
        return self

    
    def setup(self, *args, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
    
    def before(self, *args, **kwargs):
        return args, kwargs
    
    def after(self, res, *args, **kwargs):
        return res


class frame_exclude(Decorator):
    """decorator for exclude columns
    """
    
    def before(self, X, *args, exclude = None, **kwargs):
        print('~~~~~ exclude')
        if exclude is not None and isinstance(X, pd.DataFrame):
            X = X.drop(columns = exclude)
        
        return (X, *args), kwargs


class select_dtypes(Decorator):
    """ decorator for select frame by dtypes
    """

    def before(self, X, *args, select_dtypes = None, **kwargs):
        print('~~~~~ dtypes')
        if select_dtypes is not None and isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include = select_dtypes)

        return (X, *args), kwargs


class save_to_json(Decorator):
    """support save result to json file
    """
    def before(self, *args, to_json = None, **kwargs):
        self.to_json = to_json

        return args, kwargs


    def after(self, res, *args, **kwargs):
        if self.to_json is None:
            return res

        save_json(res, to_json)
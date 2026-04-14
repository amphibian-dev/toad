import numpy as np
import pandas as pd
from time import time
from .func import save_json, read_json
from functools import wraps, WRAPPER_ASSIGNMENTS



class Decorator:
    """base decorater class
    """
    _cls = None
    is_class = False

    def __init__(self, *args, is_class = False, **kwargs):
        self.is_class = is_class
        self.args = []
        self.kwargs = {}

        if len(args) == 1 and callable(args[0]):
            self.fn = args[0]
        else:
            self.setup(*args, **kwargs)
    

    @property
    def fn(self):
        if hasattr(self, '__wrapped__'):
            return self.__wrapped__
        
        return None
    
    @fn.setter
    def fn(self, func):
        if hasattr(self, 'setup_func'):
            func = self.setup_func(func)
        
        self.__wrapped__ = func

    def __call__(self, *args, **kwargs):
        if self.fn is None:
            self.fn = args[0]
            return self

        if self.is_class:
            self._cls = args[0]
            args = args[1:]

        return self.wrapper(*args, **kwargs)


    def __get__(self, instance, type = None):
        self.is_class = True
        self._cls = instance

        @wraps(self.__wrapped__)
        def func(*args, **kwargs):
            return self.__call__(instance, *args, **kwargs)

        return func


    def __getattribute__(self, name):
        if name in WRAPPER_ASSIGNMENTS:
            return getattr(self.__wrapped__, name)

        return object.__getattribute__(self, name)


    def setup(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        for key in kwargs:
            setattr(self, key, kwargs[key])


    def call(self, *args, **kwargs):
        if self._cls is not None:
            args = (self._cls, *args)

        return self.fn(*args, **kwargs)

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


class support_numpy(Decorator):
    """decorator for supporting numpy array to use torch function
    """
    def wrapper(self, *args, **kwargs):
        import torch

        has_numpy = False
        l_args = []
        for a in args:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a)
                has_numpy = True
            
            l_args.append(a)

        res = self.call(*l_args, **kwargs)

        # only when arguments has numpy array, convert result to numpy array
        if has_numpy and isinstance(res, torch.Tensor):
            res = res.numpy()
        
        return res


class xgb_loss(Decorator):
    """decorator for converting function to xgb supported loss function

    Args:
        loss_func (callable): loss function
        **kwargs: other arguments for loss function except `pred` and `label`

    Examples:

    >>> @xgb_loss(**kwargs)
    >>> def loss_func(pred, label, **kwargs):
    >>>     ...
    >>>     return loss
    >>>
    >>> # or use `xgb_loss` directly
    >>> xgb_func = xgb_loss(**kwargs)(loss_func)
    >>>
    >>> # use in xgb
    >>> model = xgb.XGBClassifier(objective = xgb_func)
    """
    def wrapper(self, pred, label):
        from .func import derivative

        def partial_func(x):
            return self.call(x, label, **self.kwargs)
        
        grad = derivative(partial_func, pred, n=1, dx=1e-6)
        hess = derivative(partial_func, pred, n=2, dx=1e-6)

        return grad, hess


class performance(Decorator):
    """decorator for analysis code performance

    Args:
        loop (int): loop times, default `1`
    
    Examples:
    >>> @performance(loop = 100)
    >>> def func():
    >>>     ... # code
    >>>     return res
    >>>
    >>> func()
    >>> 
    >>> # or use `performance` in `with` statement
    >>> with performance():
    >>>     ... # code
    """
    loop = 1

    def wrapper(self, *args, **kwargs):
        costs = []
        for _ in range(self.loop):
            start = time()
            res = self.call(*args, **kwargs)
            end = time()
            costs.append(end - start)

        self.analysis(costs)
        return res
    

    def analysis(self, costs):
        import numpy as np

        print('total cost: {:.5f}s'.format(np.sum(costs)))
        print("-"*40)
        data = {
            "Mean": np.mean(costs),
            "Min": np.min(costs),
            "Max": np.max(costs),
            "90%": np.percentile(costs, 90),
            "95%": np.percentile(costs, 95),
            "99%": np.percentile(costs, 99),
        }
        HEADER = "{:>8}"*len(data)
        BODY = "{:>7.3f}s"*len(data)
        print(HEADER.format(*data.keys()))
        print(BODY.format(*data.values()))
    

    def __enter__(self):
        self.start = time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time()
        self.analysis([self.end - self.start])

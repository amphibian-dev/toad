import pandas as pd

from .progress import Progress


class ProgressAccessor:
    def __init__(self, obj):
        self.obj = obj
    
    def apply(self, func, *args, **kwargs):
        if isinstance(self.obj, pd.Series):
            l = len(self.obj)
        else:
            # dataframe
            axis = kwargs.get("axis", 0)
            if axis == 'index':
                axis = 0
            elif axis == 'columns':
                axis = 1
            
            l = self.obj.size // self.obj.shape[axis]
        
        p = iter(Progress(range(l)))
        
        def wrapper(*args, **kwargs):
            next(p)
            return func(*args, **kwargs)
        
        res = self.obj.apply(wrapper, *args, **kwargs)
        p.end()
        return res


class pandas_enable:
    def __init__(self):
        pd.api.extensions.register_dataframe_accessor("progress")(ProgressAccessor)
        pd.api.extensions.register_series_accessor("progress")(ProgressAccessor)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exce_type, exce_value, exce_trace):
        pandas_disable()


def pandas_disable():
    if hasattr(pd.DataFrame, 'progress'):
        delattr(pd.DataFrame, 'progress')
    
    if hasattr(pd.Series, 'progress'):
        delattr(pd.Series, 'progress')
